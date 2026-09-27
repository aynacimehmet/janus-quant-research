"""Fon-gün özellik deposu (S3b-1): TEMPORAL_PROTOCOL §4/§8/§9 birebir.

Zaman alanları: feature_asof = t, decision_at = t+1, label_end = t+22,
label_available_at = cal[t+23]. Hedef (H13):
    y_t = log(NAV[t+22]/NAV[t+1]) - sum_{h=t+2..t+22} cash_h
cash_h = nakit vekili sepetin (janus.backtest.data.cash_proxy_returns, P02; brüt
günlük basit getiri) h. gün getirisi — tek kaynak, burada yeniden üretilmez.
y'de nakit terimi Σ log(1+cash_h) birikimidir (§4; NAV terimiyle aynı log uzayı);
cash_excess_* özellikleri de aynı log1p birikimi kullanır.
X ve y maskeleri ayrıdır (H03): feature_ready / label_ready; çıkarım satırları
(son 22 satır) y'siz tutulur. Tüm özellikler yalnızca NAV <= t ile hesaplanır;
meta yarı-sabit varsayılır (A3): umbrella_type_code, tax_rate.
aum_now / investor_count / tefas_status tarihsel özellik DEĞİLDİR (H02).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from janus.data.quality import next_business_day, trade_status_masks
from janus.portfolio.universe import tr_fold

LAMBDA_EWMA = 0.94  # RiskMetrics decay
MIN_HISTORY = 252  # eligible için gözlenen geçmiş alt sınırı
LABEL_HORIZON = 22  # y_t: NAV[t+22] / NAV[t+1]
DEFAULT_MAX_STALE = 2

LOG_RET_WINDOWS = (1, 5, 21, 63, 126, 252)
CASH_EXCESS_WINDOWS = (21, 63)
MACRO_COLUMNS = ("policy_rate", "d_policy_63", "cpi_yoy", "real_rate", "usdtry_ret63", "usdtry_vol21")
MARKET_COLUMNS = ("eq_trend63", "eq_vol21", "breadth200")

FEATURE_COLUMNS: list[str] = [
    *(f"log_ret_{k}" for k in LOG_RET_WINDOWS),
    *(f"cash_excess_{k}" for k in CASH_EXCESS_WINDOWS),
    "ewma_vol",
    "vol_63",
    "dd_from_peak_126",
    "dist_peak_252",
    "pct_rank_63",
    "pct_rank_252",
    "umbrella_type_code",
    "tax_rate",
    *MACRO_COLUMNS,
    *MARKET_COLUMNS,
]

TIME_COLUMNS = ("feature_asof", "fund_code", "decision_at", "label_end", "label_available_at")
MASK_COLUMNS = ("feature_ready", "label_ready", "eligible_at_decision", "policy_today_excluded")


def validate_columns(columns: list[str] | pd.Index) -> None:
    """Beyaz liste dışı özellik kolonu → ValueError (TEMPORAL §9.1)."""
    unknown = [c for c in columns if c not in FEATURE_COLUMNS]
    if unknown:
        msg = f"FEATURE_COLUMNS beyaz listesi dışı kolon(lar): {unknown}"
        raise ValueError(msg)


def _stack(df: pd.DataFrame) -> pd.Series:
    """Geniş (tarih × fon) → uzun Series, MultiIndex (tarih, fund_code); NaN korunur."""
    return df.stack()


def _policy_excluded(fund_master: pd.DataFrame, cfg: dict[str, Any] | None, leg: str = "tefas") -> pd.Series:
    """Bugünkü politika dışlaması (kara liste / durum metni / can_buy). fund_code indeksli bool.

    Bu kolon özellik DEĞİLDİR: yalnızca "bugünkü politika" koşularında kullanılır (V05);
    tarihsel deneyde eligible_at_decision kullanılır.
    """
    fm = fund_master.drop_duplicates("fund_code").set_index("fund_code")
    col = lambda c: fm[c].fillna("").astype(str).map(tr_fold) if c in fm.columns else pd.Series("", index=fm.index)  # noqa: E731
    excluded = pd.Series(False, index=fm.index)
    if cfg:
        u = cfg["legs"][leg]["universe"]
        blacklist = list(u.get("founder_blacklist", []) or [])
        if leg == "bes":
            # BES'te TEFAS kurucu kara listesi aynen uygulanır (S5-5)
            blacklist = list({*blacklist, *(cfg["legs"]["tefas"]["universe"].get("founder_blacklist") or [])})
        bl = "|".join(tr_fold(x) for x in blacklist)
        if bl:
            excluded |= col("founder").str.contains(bl, regex=True)
            excluded |= col("manager").str.contains(bl, regex=True)
            excluded |= col("name").str.contains(bl, regex=True)
        sp = "|".join(tr_fold(x) for x in (u.get("exclude_status_patterns", []) or []))
        if sp:
            excluded |= col("tefas_status").str.contains(sp, regex=True)
    can_buy, _ = trade_status_masks(fund_master)
    excluded |= ~can_buy.reindex(fm.index).fillna(True).astype(bool)
    return excluded


def _eligible_wide(nav: pd.DataFrame, max_stale: int) -> pd.DataFrame:
    """eligible_at_decision (o günkü bilgiyle): t'de NAV var, gözlenen geçmiş >= 252, data_stale değil."""
    mask = nav.notna().to_numpy()
    n = mask.shape[0]
    pos = np.arange(n)[:, None]
    last_pos = np.maximum.accumulate(np.where(mask, pos, -1), axis=0)
    count = np.cumsum(mask, axis=0)
    stale_ok = (pos - last_pos) <= max_stale  # t'de NAV varsa 0; yine de açık kontrol
    ok = mask & (count >= MIN_HISTORY) & stale_ok
    return pd.DataFrame(ok, index=nav.index, columns=nav.columns)


def _meta_features(fund_master: pd.DataFrame, funds: list[str]) -> pd.DataFrame:
    """Yarı-sabit meta (A3): umbrella_type_code (sıralı unique → deterministik kod), tax_rate."""
    fm = fund_master.drop_duplicates("fund_code").set_index("fund_code")
    ut = fm["umbrella_type"].fillna("") if "umbrella_type" in fm.columns else pd.Series("", index=fm.index)
    codes = pd.Series({v: i for i, v in enumerate(sorted(set(ut.map(tr_fold))))}, dtype=float)
    meta = pd.DataFrame(
        {
            "umbrella_type_code": ut.map(tr_fold).map(codes).reindex(funds),
            "tax_rate": pd.to_numeric(fm["withholding_rate"], errors="coerce").reindex(funds)
            if "withholding_rate" in fm.columns
            else np.nan,
        },
        index=pd.Index(funds, name="fund_code"),
    )
    return meta


def build_features(
    nav_wide: pd.DataFrame,
    fund_master: pd.DataFrame,
    cash_returns: pd.Series,
    macro_features: pd.DataFrame,
    market_features: pd.DataFrame,
    cfg: dict[str, Any] | None = None,
    asof: str | pd.Timestamp | None = None,
    leg: str = "tefas",
) -> pd.DataFrame:
    """Fon-gün paneli: satır = (feature_asof=t, fund_code); kolonlar TIME + MASK + FEATURE_COLUMNS + y.

    Girdiler: nav_wide (tarih × fon), fund_master (PIT snapshot), cash_returns (nakit vekili
    sepetin brüt günlük basit getirisi — janus.backtest.data.cash_proxy_returns), macro_features
    (PIT, janus.features.macro.macro_features çıktısı), market_features (janus.features.market).
    asof verilirse panel t <= asof'a kırpılır (tüm zaman alanları bu kırpılmış panele göre).
    """
    nav = nav_wide.sort_index()
    if asof is not None:
        nav = nav.loc[: pd.Timestamp(asof).normalize()]
    if nav.empty:
        raise ValueError("nav_wide boş")
    cal = pd.DatetimeIndex(nav.index)
    n = len(cal)
    funds = list(nav.columns)
    max_stale = (
        int((cfg.get("legs", {}).get(leg, {}).get("universe", {}) or {}).get("max_stale_days", DEFAULT_MAX_STALE))
        if cfg
        else DEFAULT_MAX_STALE
    )

    # ---- zaman alanları (işlem günü indeksi; TEMPORAL §1) -------------------------------------
    cal_s = pd.Series(cal, index=cal)
    time = pd.DataFrame(
        {
            "decision_at": cal_s.shift(-1),
            "label_end": cal_s.shift(-LABEL_HORIZON),
            "label_available_at": cal_s.shift(-LABEL_HORIZON - 1),
        }
    )
    terminal_decision = next_business_day(cal[-1], cfg)
    time.loc[cal[-1], "decision_at"] = terminal_decision
    # t+22 NAV is already in-panel for this last mature label; it becomes available on the
    # terminal row's next decision session. The terminal feature row itself keeps all labels NaT.
    mature_idx = n - LABEL_HORIZON - 1
    if mature_idx >= 0 and pd.isna(time.loc[cal[mature_idx], "label_available_at"]):
        time.loc[cal[mature_idx], "label_available_at"] = terminal_decision

    # ---- fon-bazlı geniş çerçeveler -------------------------------------------------------------
    navp = nav.where(nav > 0)
    ln = np.log(navp)
    r1 = ln.diff()
    cash_simple = cash_returns.reindex(cal).fillna(0.0)  # brüt basit getiri (cash_proxy_returns)
    cum_cash_log = np.log1p(cash_simple).cumsum()  # log uzay birikim (nakit-fazlası özellikleri için)
    cum_cash = cum_cash_log  # y için: TEMPORAL §4 Σ log(1+cash_h) — NAV terimiyle aynı log uzayı

    wide: dict[str, pd.DataFrame] = {}
    for k in LOG_RET_WINDOWS:
        wide[f"log_ret_{k}"] = ln.diff(k)
    for k in CASH_EXCESS_WINDOWS:
        wide[f"cash_excess_{k}"] = wide[f"log_ret_{k}"].sub(cum_cash_log - cum_cash_log.shift(k), axis=0)
    var = (r1**2).ewm(alpha=1 - LAMBDA_EWMA, min_periods=21).mean()
    wide["ewma_vol"] = np.sqrt(var * 252.0)
    wide["vol_63"] = r1.rolling(63, min_periods=63).std() * np.sqrt(252.0)
    wide["dd_from_peak_126"] = (
        1.0 - navp / navp.rolling(126, min_periods=126).max()
    )  # 126g kayan zirveden uzaklık (pencere içi max DD DEĞİL)
    wide["dist_peak_252"] = navp / navp.rolling(252, min_periods=252).max() - 1.0

    elig = _eligible_wide(nav, max_stale)
    for k in (63, 252):
        wide[f"pct_rank_{k}"] = wide[f"log_ret_{k}"].where(elig).rank(axis=1, pct=True)

    # ---- hedef ve olgunluk (TEMPORAL §4) --------------------------------------------------------
    y_wide = (ln.shift(-LABEL_HORIZON) - ln.shift(-1)).sub(
        cum_cash.shift(-LABEL_HORIZON) - cum_cash.shift(-1), axis=0
    )  # Σ_{h=t+2..t+22} log(1+cash_h) (TEMPORAL §4 birebir)
    label_ready = np.zeros((n, len(funds)), dtype=bool)
    label_ready[: max(n - LABEL_HORIZON, 0)] = True  # label_end = cal[t+22] <= son NAV

    # ---- uzun form --------------------------------------------------------------------------------
    X = pd.DataFrame({name: _stack(df) for name, df in wide.items()})
    X["y"] = _stack(y_wide)
    X["eligible_at_decision"] = _stack(elig.astype(float))
    X["label_ready"] = _stack(pd.DataFrame(label_ready, index=cal, columns=funds).astype(float))
    X = X.reset_index()
    X.columns = ["feature_asof", "fund_code", *X.columns[2:]]

    out = X.merge(time.reset_index(names="feature_asof"), on="feature_asof", how="left")
    meta = _meta_features(fund_master, funds)
    out = out.merge(meta.reset_index(), on="fund_code", how="left")
    for cols, frame in ((MACRO_COLUMNS, macro_features), (MARKET_COLUMNS, market_features)):
        have = (
            frame.reindex(cal)[[c for c in cols if c in frame.columns]] if not frame.empty else pd.DataFrame(index=cal)
        )
        for c in cols:  # eksik kolon → NaN (feature_ready düşer; uydurma yok)
            if c not in have:
                have[c] = np.nan
        out = out.merge(have.reset_index(names="feature_asof"), on="feature_asof", how="left")
    pol = _policy_excluded(fund_master, cfg, leg=leg)
    out["policy_today_excluded"] = out["fund_code"].map(pol).fillna(True).astype(bool)

    feature_block = out[FEATURE_COLUMNS]
    validate_columns(FEATURE_COLUMNS)
    out["feature_ready"] = feature_block.notna().all(axis=1)
    out["label_ready"] = out["label_ready"].astype(bool)
    out["eligible_at_decision"] = out["eligible_at_decision"].astype(bool)

    cols = [*TIME_COLUMNS, *MASK_COLUMNS, *FEATURE_COLUMNS, "y"]
    return out[cols].sort_values(["feature_asof", "fund_code"], kind="stable").reset_index(drop=True)
