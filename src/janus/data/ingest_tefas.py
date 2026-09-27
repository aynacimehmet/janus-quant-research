"""TEFAS ingest: fon listesi + ücretler + profil + NAV geçmişi → fund_master snapshot (PIT) ve fund_nav.

Akış: list_funds → management_fees → (fon başına) fund_info + history → build_fund_master → Store.
Hiçbir değer uydurulmaz: başarısız fonlar info_ok/hist_ok=False ile snapshot'ta kalır (PIT izi).
"""

from __future__ import annotations

import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from janus.data.quality import strict_trade_status_masks
from janus.data.store import Store
from janus.data.tefas_client import TefasClient

INFO_KEYS = [
    "isin",
    "category",
    "fund_size",
    "investor_count",
    "risk_value",
    "fund_class",
    "buy_valor",
    "sell_valor",
    "entry_fee",
    "exit_fee",
    "first_trading_time",
    "last_trading_time",
    "tefas_status",
    "kap_link",
]


# ---- saf yardımcılar (vektörize) ---------------------------------------------------------------
def derive_founder(name: pd.Series) -> pd.Series:
    """Fon adından kurucu adı: 'POYRAZ PORTFÖY ... FONU' → 'POYRAZ PORTFÖY'. 'Portföy' yoksa ilk iki kelime."""
    s = name.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    lead = s.str.extract(r"^(.*?portföy)", flags=re.IGNORECASE)[0]
    fallback = s.str.split().str[:2].str.join(" ")
    return lead.fillna(fallback).str.strip()


def _num(s: pd.Series) -> pd.Series:
    """'T+1', '%0,5', '1' gibi metinlerden ilk sayıyı çıkarır (virgül → nokta)."""
    txt = s.astype("string").str.replace(",", ".", regex=False)
    return pd.to_numeric(txt.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")


def _tax_fields(category: pd.Series, name: pd.Series, asof: datetime) -> tuple[pd.Series, pd.Series]:
    """borsapy.tax ile vergi kategorisi ve stopaj oranı; kütüphane yoksa None."""
    try:
        from borsapy.tax import classify_fund_tax_category, get_withholding_tax_rate  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return pd.Series([None] * len(category), index=category.index), pd.Series(np.nan, index=category.index)
    pairs = pd.DataFrame({"c": category.fillna(""), "n": name.fillna("")})
    uniq = pairs.drop_duplicates()
    cat_map = {(c, n): classify_fund_tax_category(c, n) for c, n in zip(uniq["c"], uniq["n"], strict=True)}
    cats = pd.Series([cat_map[(c, n)] for c, n in zip(pairs["c"], pairs["n"], strict=True)], index=category.index)
    rate_map = {c: (get_withholding_tax_rate(c, asof.date()) if c else np.nan) for c in cats.dropna().unique()}
    rates = cats.map(rate_map).astype(float)
    return cats, rates


def build_fund_master(
    list_df: pd.DataFrame,
    fees_df: pd.DataFrame,
    infos_df: pd.DataFrame,
    hist_stats: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    published_at: datetime,
) -> pd.DataFrame:
    """Kaynakları fund_code üzerinde birleştirip DATA_DICTIONARY şemasına çevirir."""
    fm = list_df[["fund_code", "name", "umbrella_type", "fund_class"]].drop_duplicates("fund_code").copy()
    fees = fees_df.drop_duplicates("fund_code")[["fund_code", "founder_code", "applied_fee", "max_expense_ratio"]]
    fm = fm.merge(fees, on="fund_code", how="left").rename(columns={"max_expense_ratio": "expense_ratio"})

    info = (
        infos_df.drop_duplicates("fund_code") if not infos_df.empty else pd.DataFrame(columns=["fund_code", *INFO_KEYS])
    )
    extra = [k for k in ("info_fetched", "last_success_at", "last_success_source") if k in info.columns]
    fm = fm.merge(
        info[["fund_code", *[k for k in INFO_KEYS if k in info.columns], *extra]],
        on="fund_code",
        how="left",
        suffixes=("", "_info"),
    )
    if "fund_class_info" in fm.columns:
        fm["fund_class"] = fm["fund_class_info"].fillna(fm["fund_class"])
        fm = fm.drop(columns="fund_class_info")

    hs = (
        hist_stats.drop_duplicates("fund_code")
        if not hist_stats.empty
        else pd.DataFrame(columns=["fund_code", "first_nav_date", "last_nav_date", "n_nav"])
    )
    fm = fm.merge(hs, on="fund_code", how="left")
    for c in ("first_nav_date", "last_nav_date"):
        fm[c] = pd.to_datetime(fm[c], errors="coerce").dt.date

    fm["founder"] = derive_founder(fm["name"])
    fm["manager"] = None  # borsapy 0.11: kaynakta yok; S3'te KAP'tan
    for col in ("buy_valor", "sell_valor"):
        fm[col] = _num(fm[col]).astype("Int64") if col in fm else pd.Series(pd.NA, index=fm.index, dtype="Int64")
    for col in ("entry_fee", "exit_fee"):
        fm[col] = _num(fm[col]) if col in fm else np.nan
    fm["aum_now"] = pd.to_numeric(fm.get("fund_size"), errors="coerce")
    fm["investor_count"] = pd.to_numeric(fm.get("investor_count"), errors="coerce").astype("Int64")
    fm["risk_value"] = pd.to_numeric(fm.get("risk_value"), errors="coerce").astype("Int64")
    fm["tax_category"], fm["withholding_rate"] = _tax_fields(fm["category"], fm["name"], published_at)
    fm["info_ok"] = fm["isin"].notna() | fm["category"].notna()
    if "info_fetched" not in fm.columns:
        fm["info_fetched"] = fm["info_ok"]
    fm["info_fetched"] = fm["info_fetched"].fillna(False).astype(bool)
    if "last_success_at" not in fm.columns:
        fm["last_success_at"] = pd.NaT
    if "last_success_source" not in fm.columns:
        fm["last_success_source"] = None
    strict_buy, strict_sell, _ = strict_trade_status_masks(fm)
    fm["can_buy"] = strict_buy.to_numpy(dtype=bool)
    fm["can_sell"] = strict_sell.to_numpy(dtype=bool)
    fm["hist_ok"] = fm["n_nav"].fillna(0).gt(0)
    fm["n_nav"] = fm["n_nav"].fillna(0).astype("Int64")
    fm["snapshot_date"] = snapshot_date.date()
    fm["source_published_at"] = published_at

    cols = [
        "snapshot_date",
        "fund_code",
        "isin",
        "name",
        "fund_class",
        "umbrella_type",
        "category",
        "founder_code",
        "founder",
        "manager",
        "buy_valor",
        "sell_valor",
        "entry_fee",
        "exit_fee",
        "first_trading_time",
        "last_trading_time",
        "tax_category",
        "withholding_rate",
        "applied_fee",
        "expense_ratio",
        "aum_now",
        "investor_count",
        "risk_value",
        "tefas_status",
        "kap_link",
        "first_nav_date",
        "last_nav_date",
        "n_nav",
        "info_ok",
        "hist_ok",
        "source_published_at",
        "info_fetched",
        "last_success_at",
        "last_success_source",
        "can_buy",
        "can_sell",
    ]
    for c in cols:
        if c not in fm.columns:
            fm[c] = None
    return fm[cols].reset_index(drop=True)


def history_to_long(code: str, hist: pd.DataFrame, published_at: datetime) -> pd.DataFrame:
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["fund_code", "date", "price", "published_at"])
    out = pd.DataFrame({"date": pd.to_datetime(hist.index).date, "price": hist["Price"].astype(float).to_numpy()})
    out = out[out["price"] > 0]
    out.insert(0, "fund_code", code)
    out["published_at"] = published_at
    return out


# ---- orkestrasyon ------------------------------------------------------------------------------
def ingest_tefas(
    store: Store,
    client: TefasClient,
    cfg: dict[str, Any],
    mode: str = "incremental",
    limit: int | None = None,
    snapshot_date: pd.Timestamp | None = None,
    codes: list[str] | None = None,
    only_failed: bool = False,
) -> dict[str, Any]:
    """Tam akış. `mode`: initial (5y) | incremental (kısa pencere). `codes`: yalnızca bu fonlar;
    `only_failed`: son ingest koşusunda hata veren fonlar. `ingest.workers` > 1 ise fonlar paralel çekilir."""
    icfg = cfg.get("ingest", {})
    fund_types = icfg.get("fund_types", ["YAT"])
    period = icfg.get("initial_period", "5y") if mode == "initial" else icfg.get("incremental_period", "1mo")
    started = datetime.now()
    published_at = started
    snap = pd.Timestamp(snapshot_date) if snapshot_date is not None else pd.Timestamp(started).normalize()
    run_id = f"ingest-{started:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    failures: dict[str, list[str]] = {"info": [], "history": []}

    lists, fees = [], []
    for ft in fund_types:
        lists.append(client.list_funds(ft))
        fees.append(client.management_fees(ft))
    list_df = pd.concat(lists, ignore_index=True)
    fees_df = pd.concat(fees, ignore_index=True)
    prev_fm = store.latest_fund_master()
    bes_patterns = cfg.get("legs", {}).get("bes", {}).get("exclude_umbrella_patterns", [])
    emk_codes: set[str] = set()
    if "EMK" in {str(ft).upper() for ft in fund_types}:
        from janus.portfolio.universe import bes_scope_excluded  # noqa: PLC0415

        # TEFAS list aşamasında yalnız umbrella_type kaynak verisidir; kategori ancak
        # profil sonrası bilinir. Son snapshot kategorisi varsa yeni profil çağrısını önle.
        excluded_umbrella = bes_scope_excluded(list_df, bes_patterns, include_category=False)
        prior_categories = (
            prev_fm.drop_duplicates("fund_code").set_index("fund_code")["category"]
            if not prev_fm.empty and {"fund_code", "category"}.issubset(prev_fm.columns)
            else pd.Series(dtype=object)
        )
        list_df["category"] = list_df["fund_code"].map(prior_categories)
        excluded_cached_category = bes_scope_excluded(
            list_df, bes_patterns, include_umbrella=False, include_category=True
        )
        list_df = list_df.loc[~(excluded_umbrella | excluded_cached_category)].copy()
        emk_codes = set(
            list_df.loc[list_df["fund_class"].fillna("").astype(str).str.upper().eq("EMK"), "fund_code"].astype(str)
        )
    all_codes = list_df["fund_code"].dropna().astype(str).unique().tolist()
    if only_failed:
        last = store.last_run("ingest_tefas")
        failed = (
            []
            if not last
            else [
                f.split(":")[0]
                for f in last["summary"].get("failures_info", []) + last["summary"].get("failures_history", [])
            ]
        )
        codes = [c for c in dict.fromkeys(failed) if c in all_codes]
    elif codes:
        wanted = {c.upper() for c in codes}
        codes = [c for c in all_codes if c in wanted]
    else:
        codes = all_codes
    if limit:
        codes = codes[:limit]
    subset_run = bool(only_failed or (codes is not None and len(codes) < len(all_codes)))
    logger.info("{} fon listelendi; {} fon işlenecek ({}, period={})", len(list_df), len(codes), mode, period)

    # Kalıcı profil hataları: yalnızca haftada bir (retry_failed_info_weekday, 0=Pzt … 6=Paz) yeniden denenir.
    # Gün, run/as-of tarihinden (snap) türetilir; duvar saati (datetime.now()) kullanılmaz.
    retry_day = int(icfg.get("retry_failed_info_weekday", 6))
    skip_info: set[str] = set()
    if not prev_fm.empty and "info_ok" in prev_fm.columns and not subset_run and snap.weekday() != retry_day:
        skip_info = set(prev_fm.loc[~prev_fm["info_ok"].fillna(False).astype(bool), "fund_code"].astype(str))

    def _one(code: str) -> tuple[dict | None, pd.DataFrame | None, list[tuple[str, str]]]:
        errs: list[tuple[str, str]] = []
        info = long = None
        if code in skip_info:
            errs.append(("info", f"{code}:SkippedUntilWeekly"))
        else:
            try:
                d = client.fund_info(code)
                ingested_at = datetime.now()
                info = {
                    "fund_code": code,
                    **{k: d.get(k) for k in INFO_KEYS},
                    "info_fetched": True,
                    "last_success_at": ingested_at,
                    "last_success_source": "fetch",
                }
                if (
                    code in emk_codes
                    and bes_patterns
                    and bes_scope_excluded(
                        pd.DataFrame({"fund_code": [code], "fund_class": ["EMK"], "category": [d.get("category")]}),
                        bes_patterns,
                        include_umbrella=False,
                        include_category=True,
                    ).iloc[0]
                ):
                    return info, None, errs
            except Exception as e:  # noqa: BLE001
                errs.append(("info", f"{code}:{type(e).__name__}"))
        try:
            long = history_to_long(code, client.history(code, period=period), published_at)
        except Exception as e:  # noqa: BLE001
            errs.append(("history", f"{code}:{type(e).__name__}"))
        return info, long, errs

    workers = max(1, int(icfg.get("workers", 1)))
    infos: list[dict] = []
    nav_frames: list[pd.DataFrame] = []
    results = ThreadPoolExecutor(max_workers=workers).map(_one, codes) if workers > 1 else map(_one, codes)
    for i, (info, long, errs) in enumerate(results, 1):
        if info:
            infos.append(info)
        if long is not None and not long.empty:
            nav_frames.append(long)
        for kind, msg in errs:
            failures[kind].append(msg)
        if i % 100 == 0:
            logger.info("… {}/{} fon", i, len(codes))

    infos_df = pd.DataFrame(infos)
    # Bugün profili alınamayan ama daha önce alınmış fonlar: son bilinen profil taşınır (info_fetched=False)
    if not prev_fm.empty and "info_ok" in prev_fm.columns:
        have = set(infos_df["fund_code"]) if not infos_df.empty else set()
        carry = prev_fm[
            prev_fm["info_ok"].fillna(False).astype(bool)
            & prev_fm["fund_code"].isin(codes)
            & ~prev_fm["fund_code"].isin(have)
        ]
        if not carry.empty:
            keep = [k for k in (*INFO_KEYS, "last_success_at", "last_success_source") if k in carry.columns]
            cf = carry[["fund_code", *keep]].copy()
            cf["fund_size"] = carry["aum_now"].to_numpy() if "aum_now" in carry else np.nan
            cf["info_fetched"] = False
            infos_df = pd.concat([infos_df, cf], ignore_index=True)
    nav_df = (
        pd.concat(nav_frames, ignore_index=True)
        if nav_frames
        else pd.DataFrame(columns=["fund_code", "date", "price", "published_at"])
    )
    n_nav = store.upsert_nav(nav_df)

    # NAV istatistikleri her zaman DEPODAN türetilir (artımlı modda pencere değil, toplam geçmiş)
    hist_stats = store.nav_stats(codes)

    fm = build_fund_master(list_df[list_df["fund_code"].isin(codes)], fees_df, infos_df, hist_stats, snap, published_at)
    n_fm = store.write_fund_master_rows(fm) if subset_run else store.write_fund_master(fm)

    summary = {
        "run_id": run_id,
        "mode": mode,
        "period": period,
        "snapshot_date": str(snap.date()),
        "n_listed": int(len(list_df)),
        "n_processed": len(codes),
        "n_info_ok": int(fm["info_ok"].sum()),
        "n_hist_ok": int(fm["hist_ok"].sum()),
        "n_nav_rows": int(n_nav),
        "n_fund_master_rows": int(n_fm),
        "failures_info": failures["info"][:50],
        "failures_history": failures["history"][:50],
        "n_failures": len(failures["info"]) + len(failures["history"]),
        "duration_s": round((datetime.now() - started).total_seconds(), 1),
    }
    status = "ok" if summary["n_failures"] == 0 else ("partial" if summary["n_hist_ok"] > 0 else "failed")
    store.log_run(run_id, "ingest_tefas", started, status, summary)
    logger.info(
        "ingest bitti: {} | info_ok={} hist_ok={} nav_rows={} failures={}",
        status,
        summary["n_info_ok"],
        summary["n_hist_ok"],
        n_nav,
        summary["n_failures"],
    )
    return {"status": status, **summary}
