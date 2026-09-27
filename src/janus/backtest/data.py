"""Backtest için veri hazırlığı: NAV paneli, nakit vekili sepeti (P02), uygunluk maskesi, meta."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from loguru import logger

from janus.backtest.costs import FundMeta, build_fund_meta
from janus.data.quality import execution_status_masks
from janus.data.store import Store
from janus.portfolio.universe import bes_scope_mask, phase1_mask, tr_fold

_PIT_PROFILE_START = pd.Timestamp("2026-09-22")
_EXECUTION_FIELDS = (
    "buy_valor",
    "sell_valor",
    "entry_fee",
    "exit_fee",
    "can_buy",
    "can_sell",
    "tefas_status",
    "tax_category",
)


def _execution_profiles_asof(
    profiles: pd.DataFrame, asof: pd.Timestamp | str, morning_cutoff: str = "09:15"
) -> pd.DataFrame:
    """Return each fund's newest execution profile available by the local morning decision time."""
    if profiles.empty or "snapshot_date" not in profiles:
        return profiles.iloc[0:0].copy()
    decision = pd.Timestamp(asof)
    if decision.tzinfo is not None:
        decision = decision.tz_localize(None)
    decision = decision.normalize()
    hour, minute = (int(part) for part in morning_cutoff.split(":"))
    cutoff = decision + pd.Timedelta(hours=hour, minutes=minute)
    rows = profiles.copy()
    indexed_codes = "fund_code" not in rows.columns and rows.index.name == "fund_code"
    if indexed_codes:
        rows["_profile_fund_code"] = rows.index.astype(str)
    elif "fund_code" not in rows.columns:
        return rows.iloc[0:0].copy()
    rows["snapshot_date"] = pd.to_datetime(rows["snapshot_date"], errors="coerce").dt.normalize()
    code_column = "_profile_fund_code" if indexed_codes else "fund_code"
    rows[code_column] = rows[code_column].astype(str)
    eligible = rows["snapshot_date"].le(decision)
    if "source_published_at" in rows:
        published = pd.to_datetime(rows["source_published_at"], errors="coerce")
        if published.dt.tz is not None:
            published = published.dt.tz_localize(None)
        eligible &= published.isna() | published.le(cutoff)
    return rows.loc[eligible].sort_values([code_column, "snapshot_date"]).drop_duplicates(code_column, keep="last")


def _execution_meta_by_date(
    store: Store, decision_dates: pd.DatetimeIndex, codes: list[str], morning_cutoff: str = "09:15"
) -> pd.DataFrame:
    """Build execution-only date/fund profiles. This output is deliberately outside FEATURE_COLUMNS."""
    history_reader = getattr(store, "fund_master_history", None)
    history = history_reader() if callable(history_reader) else store.latest_fund_master()
    dates = pd.DatetimeIndex(decision_dates).normalize().unique().sort_values()
    grid = pd.MultiIndex.from_product([dates, codes], names=["decision_date", "fund_code"]).to_frame(index=False)
    empty_columns = [
        *_EXECUTION_FIELDS,
        "status",
        "execution_source",
        "source_snapshot_date",
        "last_success_at",
        "last_success_source",
        "published_at",
        "publication_time_assumption",
        "buy_reason",
        "sell_reason",
        "status_flag_mismatch",
    ]
    if not history.empty:
        history = history.copy()
        history["snapshot_date"] = pd.to_datetime(history["snapshot_date"], errors="coerce").dt.normalize()
        history["fund_code"] = history["fund_code"].astype(str)
        for field in _EXECUTION_FIELDS:
            if field not in history:
                history[field] = np.nan
        if "last_success_at" not in history:
            history["last_success_at"] = pd.NaT
        if "last_success_source" not in history:
            history["last_success_source"] = None
        if "source_published_at" not in history:
            history["source_published_at"] = pd.NaT
        history["source_published_at"] = pd.to_datetime(history["source_published_at"], errors="coerce")
        source = history.sort_values(["fund_code", "snapshot_date"])
        latest = source.dropna(subset=["snapshot_date"]).drop_duplicates("fund_code", keep="last")
        _, _, _, latest_mismatch = execution_status_masks(latest)
        mismatch_count = int(latest_mismatch.sum())
        if mismatch_count:
            logger.warning("fund_master status metni ile kayıtlı yön bayrakları uyuşmuyor: {} profil", mismatch_count)

        # A3 execution assumptions apply only before the first PIT date and are explicitly tagged.
        pre_mask = grid["decision_date"] < _PIT_PROFILE_START
        pre = grid.loc[pre_mask, ["decision_date", "fund_code"]].merge(
            latest[["fund_code", *_EXECUTION_FIELDS, "snapshot_date", "last_success_at", "last_success_source"]],
            on="fund_code",
            how="left",
            suffixes=("", "_profile"),
        )
        if not pre.empty:
            pre["can_buy"] = True
            pre["can_sell"] = True
            pre["tefas_status"] = "İşlem Görüyor"
            pre["status"] = "A3 varsayımlı, PIT kanıtı değil"
            pre["execution_source"] = "A3_CURRENT_PROFILE"
            pre["last_success_source"] = pre["last_success_source"].fillna("A3_current_profile")
            pre["source_snapshot_date"] = pre["snapshot_date"]
            pre["published_at"] = pre.get("source_published_at", pd.NaT)
            pre["publication_time_assumption"] = np.where(
                pd.notna(pre["published_at"]), "source_timestamp", "date_only_pit_assumption"
            )
            pre["buy_reason"] = np.where(pre["buy_valor"].notna(), "A3_assumption", "missing_buy_valor")
            pre["sell_reason"] = np.where(pre["sell_valor"].notna(), "A3_assumption", "missing_sell_valor")
            for col in ["buy_valor", "sell_valor", "entry_fee", "exit_fee"]:
                pre[col] = pd.to_numeric(pre[col], errors="coerce")
            for field in ("entry_fee", "exit_fee"):
                pre[field] = 0.0
            pre["status_flag_mismatch"] = False
            for col, valid_col, invalid_reason in (
                ("buy_reason", "buy_valor", "missing_buy_valor"),
                ("sell_reason", "sell_valor", "missing_sell_valor"),
            ):
                invalid = pre[valid_col].isna() | ~np.isfinite(pre[valid_col]) | pre[valid_col].lt(0)
                pre.loc[invalid, col] = invalid_reason
                pre.loc[invalid, "can_buy" if col == "buy_reason" else "can_sell"] = False
            pre = pre[grid.columns.tolist() + empty_columns]
        else:
            pre = pd.DataFrame(columns=[*grid.columns, *empty_columns])

        post_grid = grid.loc[~pre_mask].sort_values(["decision_date", "fund_code"])
        if not post_grid.empty:
            pit_history = source.dropna(subset=["snapshot_date"]).sort_values(["snapshot_date", "fund_code"])
            # Select each day's latest snapshot that was actually published by the configured
            # local morning decision cutoff. Missing publication timestamps retain date-only PIT.
            selected = []
            for decision_date, day_grid in post_grid.groupby("decision_date", sort=True):
                cutoff = decision_date + pd.Timedelta(
                    hours=int(morning_cutoff.split(":")[0]), minutes=int(morning_cutoff.split(":")[1])
                )
                day_history = pit_history[
                    pit_history["source_published_at"].isna() | pit_history["source_published_at"].le(cutoff)
                ]
                selected.append(
                    pd.merge_asof(
                        day_grid.sort_values(["decision_date", "fund_code"]),
                        day_history.sort_values(["snapshot_date", "fund_code"]),
                        left_on="decision_date",
                        right_on="snapshot_date",
                        by="fund_code",
                        direction="backward",
                        allow_exact_matches=True,
                    )
                )
            post = pd.concat(selected, ignore_index=True)
            post["execution_source"] = "PIT"
            post["source_snapshot_date"] = post["snapshot_date"]
            post["status"] = post["tefas_status"]
            post["last_success_at"] = pd.to_datetime(post["last_success_at"], errors="coerce")
            if "last_success_source" not in post:
                post["last_success_source"] = None
            success_local = post["last_success_at"].dt.tz_localize(None)
            cutoff_times = post["decision_date"] + pd.Timedelta(
                hours=int(morning_cutoff.split(":")[0]), minutes=int(morning_cutoff.split(":")[1])
            )
            age = cutoff_times - success_local
            reason = pd.Series("ok", index=post.index, dtype="object")
            reason = reason.mask(post["snapshot_date"].isna(), "no_pit_snapshot")
            reason = reason.mask(reason.eq("ok") & post["last_success_at"].isna(), "missing_last_success_at")
            reason = reason.mask(
                reason.eq("ok") & success_local.gt(cutoff_times),
                "future_last_success_at",
            )
            reason = reason.mask(reason.eq("ok") & age.gt(pd.Timedelta(days=7)), "stale_last_success_at")
            status_buy, status_sell, status_known, flag_mismatch = execution_status_masks(post)
            status_known_by_code = status_known.to_numpy(bool)
            reason = reason.mask(reason.eq("ok") & ~status_known_by_code, "unknown_tefas_status")
            status_buy_by_code = status_buy.to_numpy(bool)
            status_sell_by_code = status_sell.to_numpy(bool)
            missing = pd.Series(False, index=post.index)
            for field in ("entry_fee", "exit_fee"):
                post[field] = 0.0
            for field in _EXECUTION_FIELDS:
                if field not in {"tefas_status", "tax_category", "can_buy", "can_sell", "entry_fee", "exit_fee"}:
                    missing |= post[field].isna()
            reason = reason.mask(reason.eq("ok") & missing, "missing_execution_field")
            tax_category = post["tax_category"].astype("string").str.strip()
            missing_tax = tax_category.isna() | tax_category.eq("")
            reason = reason.mask(reason.eq("ok") & missing_tax, "missing_tax_category")
            for field in ("buy_valor", "sell_valor"):
                values = pd.to_numeric(post[field], errors="coerce")
                invalid = values.isna() | ~np.isfinite(values) | values.lt(0) | values.ne(np.floor(values))
                reason = reason.mask(reason.eq("ok") & invalid, f"invalid_{field}")
                post[field] = values
            valid = reason.eq("ok")
            post["can_buy"] = status_buy_by_code & valid.to_numpy(bool)
            post["can_sell"] = status_sell_by_code & valid.to_numpy(bool)
            post["status_flag_mismatch"] = flag_mismatch.to_numpy(bool)
            buy_reason = reason.mask(valid & ~status_buy_by_code, "status_buy_closed")
            sell_reason = reason.mask(valid & ~status_sell_by_code, "status_sell_closed")
            post["buy_reason"] = np.where(post["can_buy"], "ok", buy_reason)
            post["sell_reason"] = np.where(post["can_sell"], "ok", sell_reason)
            post["published_at"] = post["source_published_at"]
            post["publication_time_assumption"] = np.where(
                post["published_at"].notna(), "source_timestamp", "date_only_pit_assumption"
            )
            post = post[[*grid.columns, *empty_columns]]
        else:
            post = pd.DataFrame(columns=[*grid.columns, *empty_columns])
        result = pd.concat([pre, post], ignore_index=True)
    else:
        result = grid.copy()
        for col in _EXECUTION_FIELDS:
            result[col] = np.nan
        result["execution_source"] = "PIT"
        result["status"] = ""
        result["source_snapshot_date"] = pd.NaT
        result["last_success_at"] = pd.NaT
        result["last_success_source"] = None
        result["status_flag_mismatch"] = False
        result["published_at"] = pd.NaT
        result["publication_time_assumption"] = "date_only_pit_assumption"
        result["buy_reason"] = "no_pit_snapshot"
        result["sell_reason"] = "no_pit_snapshot"
        result["can_buy"] = False
        result["can_sell"] = False
    result["available_from"] = pd.to_datetime(result["source_snapshot_date"], errors="coerce")
    return result.set_index(["decision_date", "fund_code"]).sort_index()


def cash_proxy_codes(
    nav_wide: pd.DataFrame,
    fund_master: pd.DataFrame,
    cfg: dict,
    leg: str = "tefas",
    min_coverage: float = 0.99,
    top: int = 5,
    require_execution_profile: bool = False,
    profile_asof: pd.Timestamp | str | None = None,
) -> list[str]:
    """P02/Ekle-1: nakit vekili sepeti üyeleri — Para Piyasası şemsiyesi, pencere boyunca NAV doluluğu ≥ %99,
    kurucu kara listesinde değil; n_nav azalan (eşitlikte fon kodu) sıralı, en fazla `top` fon, eşit ağırlık.
    `leg`: "tefas" | "bes" — BES'te EMK para piyasası fonlarından sepet; yoksa boş (fallback yok, S5-5).
    TEFAS seçiminde founder_code kimliği zorunludur; portföy-geneli adet/ağırlık kısıtı emir yolunda uygulanır.
    `require_execution_profile` canlı/PIT TEFAS yolunda profili ve iki valörü zorunlu kılar. Valörler
    2026-09-22 öncesine taşınmaz. BES davranışı bu TEFAS kuralından etkilenmez. Bugünkü büyüklüğe göre seçim
    YAPILMAZ; tek vergi kategorisine filtre (Cevap-2). A7 (sağkalım) etiketi raporda."""
    fm = fund_master.drop_duplicates("fund_code").set_index("fund_code")
    umbrella_pat = "Para Piyasası" if leg == "tefas" else "Para Piyasası"
    mm = fm[fm["umbrella_type"].fillna("").str.contains(umbrella_pat, case=False)]
    if leg == "bes":
        # F13/F16: BES nakit vekili yalnız EMK yapısal kapsamından; TEFAS (YAT) fonu karışmaz.
        scope = bes_scope_mask(fm.reset_index(), cfg)
        mm = mm[mm.index.isin(scope.index[scope.to_numpy()])]
    bl = cfg["legs"][leg]["universe"].get("founder_blacklist", []) or []
    if leg == "bes":
        bl = list({*bl, *(cfg["legs"]["tefas"]["universe"].get("founder_blacklist") or [])})
    pat = "|".join(re.escape(tr_fold(x)) for x in bl)
    if pat and "founder" in mm.columns:
        mm = mm[~mm["founder"].fillna("").astype(str).map(tr_fold).str.contains(pat, regex=True)]
    cols = [c for c in mm.index if c in nav_wide.columns]
    if not cols:
        return []
    mm = mm.loc[cols]
    mm = mm.assign(_cov=nav_wide[cols].notna().mean().to_numpy())
    mm = mm[mm["_cov"] >= min_coverage]
    if "tax_category" in mm.columns and mm["tax_category"].notna().any():
        top_cat = mm["tax_category"].dropna().mode().iloc[0]
        mm = mm[mm["tax_category"].fillna(top_cat) == top_cat]
    if leg == "tefas":
        if "founder_code" not in mm.columns:
            return []
        founder_code = mm["founder_code"].astype("string").str.strip()
        mm = mm[founder_code.notna() & founder_code.ne("")].copy()
        mm["_founder_code"] = founder_code.loc[mm.index]
        if require_execution_profile:
            if profile_asof is None:
                raise ValueError("strict canlı B0 seçimi için profile_asof PIT tarihi zorunludur")
            snapshot_ts = pd.to_datetime(profile_asof, errors="coerce")
            if pd.isna(snapshot_ts):
                raise ValueError("strict canlı B0 seçimi için geçerli profile_asof PIT tarihi zorunludur")
            if snapshot_ts.normalize() < _PIT_PROFILE_START:
                return []
            if "snapshot_date" not in mm.columns:
                return []
            mm = _execution_profiles_asof(
                mm, snapshot_ts, cfg.get("project", {}).get("runs", {}).get("morning", "09:15")
            )
            mm = mm[pd.to_datetime(mm["snapshot_date"], errors="coerce").dt.normalize().ge(_PIT_PROFILE_START)]
            valor_cols = ["buy_valor", "sell_valor"]
            if any(col not in mm.columns for col in valor_cols):
                return []
            valor_ok = pd.Series(True, index=mm.index)
            for col in valor_cols:
                values = pd.to_numeric(mm[col], errors="coerce")
                valor_ok &= values.notna() & np.isfinite(values) & values.ge(0) & values.eq(np.floor(values))
            mm = mm[valor_ok]
            required_profile = ("tax_category", "tefas_status", "last_success_at")
            if any(col not in mm.columns for col in required_profile):
                return []
            tax = mm["tax_category"].astype("string").str.strip()
            mm = mm[tax.notna() & tax.ne("")]
            status_buy, status_sell, status_known, mismatch = execution_status_masks(mm.reset_index())
            mismatch_count = int(mismatch.sum())
            if mismatch_count:
                logger.warning("B0 status metni ile kayıtlı bayrak uyuşmuyor: {} profil", mismatch_count)
            mm = mm.loc[status_known.to_numpy() & status_buy.to_numpy() & status_sell.to_numpy()]
            success_at = pd.to_datetime(mm["last_success_at"], errors="coerce")
            if success_at.dt.tz is not None:
                success_at = success_at.dt.tz_localize(None)
            cutoff = snapshot_ts.normalize() + pd.Timedelta(
                hours=int(cfg.get("project", {}).get("runs", {}).get("morning", "09:15").split(":")[0]),
                minutes=int(cfg.get("project", {}).get("runs", {}).get("morning", "09:15").split(":")[1]),
            )
            age = cutoff - success_at
            mm = mm[success_at.notna() & success_at.le(cutoff) & age.ge(pd.Timedelta(0)) & age.le(pd.Timedelta(days=7))]
    if mm.empty:
        return []
    n_nav = (
        pd.to_numeric(mm["n_nav"], errors="coerce") if "n_nav" in mm.columns else mm["_cov"] * len(nav_wide)
    ).fillna(0.0)
    order = np.lexsort((mm.index.to_numpy(), -n_nav.to_numpy()))  # n_nav azalan, eşitlikte kod
    ranked = mm.iloc[order]
    return [str(code) for code in ranked.index[:top]]


def cash_proxy_returns(nav_wide: pd.DataFrame, codes: list[str]) -> pd.Series:
    """Sepetin eşit ağırlıklı günlük basit getirisi (sensitivite yolu ve B0 karşılaştırması için)."""
    if not codes:
        return pd.Series(0.0, index=nav_wide.index)
    r = nav_wide[codes].pct_change().clip(-0.01, 0.01)  # veri hatalarına karşı kırp
    return r.mean(axis=1).fillna(0.0)


def equity_index(
    nav_wide: pd.DataFrame, fund_master: pd.DataFrame, leg: str = "tefas", min_coverage: float = 0.8
) -> pd.Series:
    """Hisse fonlarının eşit ağırlıklı günlük getirisinden kurulan piyasa göstergesi (R0 gate için).

    `leg`: "tefas" | "bes" — BES'te EMK hisse fonlarından hesaplanır.
    """
    fm = fund_master.drop_duplicates("fund_code").set_index("fund_code")
    eq = fm[fm["umbrella_type"].fillna("").str.contains("Hisse", case=False)]
    cols = [c for c in eq.index if c in nav_wide.columns]
    if not cols:
        return pd.Series(np.nan, index=nav_wide.index)
    r = nav_wide[cols].pct_change().clip(-0.2, 0.2)
    ok = r.notna().mean(axis=1) >= min_coverage
    idx = (1 + r.mean(axis=1).where(ok, 0.0)).cumprod()
    return idx


def prepare(store: Store, cfg: dict, start: str | None = None) -> dict:
    """Depodan backtest girdileri."""
    fm = store.latest_fund_master()
    nav = store.nav_wide(start)
    # Status is date-varying execution metadata. The static model/backtest universe must not
    # apply today's status exclusion to pre-PIT decisions; execution_meta_by_date enforces
    # strict status at/after the first PIT snapshot and ADR-23 A3 status before it.
    tefas_cfg = cfg["legs"]["tefas"]
    universe_cfg = {
        **cfg,
        "legs": {
            **cfg["legs"],
            "tefas": {
                **tefas_cfg,
                "universe": {**tefas_cfg["universe"], "exclude_status_patterns": []},
            },
        },
    }
    elig = phase1_mask(fm, universe_cfg)
    codes = [c for c in nav.columns if c in elig.index]
    nav = nav[codes]
    meta: FundMeta = build_fund_meta(fm, codes, cfg)
    # Keep execution coverage independent of today's policy/blacklist-filtered model universe.
    morning_cutoff = cfg.get("project", {}).get("runs", {}).get("morning", "09:15")
    execution_meta = _execution_meta_by_date(store, nav.index, list(nav.columns), morning_cutoff)
    fee_scale = float(cfg["legs"]["tefas"].get("execution", {}).get("fee_scale", 0.01))
    for field in ("entry_fee", "exit_fee"):
        execution_meta[field] = pd.to_numeric(execution_meta[field], errors="coerce") * fee_scale
    cash_codes = cash_proxy_codes(nav, fm, cfg)
    cash_returns = cash_proxy_returns(nav, cash_codes)
    mm = fm.reindex(cash_codes) if cash_codes else fm.iloc[0:0]
    cats = mm["tax_category"].dropna() if "tax_category" in mm.columns else pd.Series(dtype=object)
    wr = (
        pd.to_numeric(mm["withholding_rate"], errors="coerce")
        if "withholding_rate" in mm.columns
        else pd.Series(dtype=float)
    )
    return {
        "nav": nav,
        "fund_master": fm,
        "meta": meta,
        "execution_meta_by_date": execution_meta,
        "eligible": elig.reindex(codes).fillna(False),
        "cash_codes": cash_codes,
        "cash_returns": cash_returns,
        "cash_nav": (1 + cash_returns).cumprod(),  # sepet indeksi (slot fiyatı)
        "cash_category": str(cats.mode().iloc[0]) if not cats.empty else None,
        "cash_rate": float(wr.mean()) if wr.notna().any() else 0.175,
        "equity_index": equity_index(nav, fm),
        # parquet anlık görüntüsünden okunuyorsa dolu (S3b-0b); normal Store'da None
        "snapshot_asof": getattr(store, "snapshot_asof", None),
    }
