"""Veri kalitesi (S3b-0c-1): NAV eskiliği ≠ resmî askı; can_buy/can_sell; tazelik referansı bağımsız takvimden.

Sözleşme: docs/LEDGER_EXECUTION_SPEC.md §1, docs/TEMPORAL_PROTOCOL.md §2.
- data_stale: NAV, beklenen tarihe göre > max_stale_days iş günü eski (yalnızca veri durumu).
- suspended: tefas_status "alımına kapalı" VE "bozumuna kapalı" (resmî askı) — "işlem görmüyor" askı DEĞİLDİR.
- can_buy / can_sell: durum metninden; boş/NaN → ikisi True.
- fresh_ratio paydası eligible evren; referans = karar gününden önceki iş günü (panel ekseni değil).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from janus.portfolio.universe import tr_fold

_BUY_CLOSED = ("alımına kapalı", "işlem görmüyor")
_SELL_CLOSED = ("bozumuna kapalı", "işlem görmüyor")


# ---- takvim ----------------------------------------------------------------------------------
def _holidays(cfg: dict[str, Any] | None) -> np.ndarray:
    hol = (cfg or {}).get("calendar", {}).get("holidays", []) or []
    return np.array([np.datetime64(pd.Timestamp(h).date()) for h in hol], dtype="datetime64[D]")


def is_business_day(date: str | pd.Timestamp, cfg: dict[str, Any] | None = None) -> bool:
    """Configured BIST takviminde tarih iş günü mü? (Hafta sonu + `calendar.holidays`)."""
    day = np.datetime64(pd.Timestamp(date).date())
    return bool(np.is_busday(day, holidays=_holidays(cfg)))


def business_days(
    start: str | pd.Timestamp, end: str | pd.Timestamp, cfg: dict[str, Any] | None = None
) -> pd.DatetimeIndex:
    """Dahilî [start, end] aralığındaki configured BIST iş günleri."""
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if first > last:
        return pd.DatetimeIndex([])
    days = pd.date_range(first, last, freq="D")
    valid = np.is_busday(days.to_numpy(dtype="datetime64[D]"), holidays=_holidays(cfg))
    return days[valid]


def next_business_day(date: str | pd.Timestamp, cfg: dict[str, Any] | None = None) -> pd.Timestamp:
    """Tarihten kesinlikle sonraki configured BIST iş günü; takvim gününü değil session indeksini kullanır."""
    day = np.datetime64(pd.Timestamp(date).date())
    nxt = np.busday_offset(day, 1, roll="backward", holidays=_holidays(cfg))
    return pd.Timestamp(nxt)


def expected_last_nav_date(asof: str | pd.Timestamp | None = None, cfg: dict[str, Any] | None = None) -> pd.Timestamp:
    """Koşu anı `asof` için beklenen en yeni NAV etiketi (TEFAS etiketi = ilan günü, A1-rev):
    iş günü ve saat ≥ `calendar.nav_publish_time` (varsayılan 10:00) ise bugün; aksi hâlde önceki iş günü."""
    t = pd.Timestamp(asof) if asof is not None else pd.Timestamp.now()
    hol = _holidays(cfg)
    pub = str((cfg or {}).get("calendar", {}).get("nav_publish_time", "10:00"))
    hh, mm = (int(x) for x in pub.split(":"))
    today = np.datetime64(t.date())
    is_bday = bool(np.is_busday(today, holidays=hol))
    if is_bday and (t.hour, t.minute) >= (hh, mm):
        return pd.Timestamp(today)
    if is_bday:  # iş günü ama yayın saatinden önce → önceki iş günü
        return pd.Timestamp(np.busday_offset(today, -1, roll="backward", holidays=hol))
    return pd.Timestamp(np.busday_offset(today, 0, roll="backward", holidays=hol))  # hafta sonu/tatil → son iş günü


def business_days_behind(
    dates: pd.Series | pd.DatetimeIndex, reference: pd.Timestamp, cfg: dict[str, Any] | None = None
) -> np.ndarray:
    """reference − date, iş günü cinsinden (NaT → inf). Vektörize (np.busday_count)."""
    arr = pd.to_datetime(pd.Series(dates)).dt.normalize()
    out = np.full(len(arr), np.inf)
    ok = arr.notna().to_numpy()
    if ok.any():
        out[ok] = np.busday_count(
            arr[ok].to_numpy(dtype="datetime64[D]"), np.datetime64(reference.date()), holidays=_holidays(cfg)
        ).astype(float)
    return out


# ---- NAV paneli -------------------------------------------------------------------------------
def stale_days(nav_wide: pd.DataFrame) -> pd.Series:
    """Panelin son tarihine göre son geçerli NAV'dan bu yana geçen satır sayısı (iç kullanım; referans takvimi değil)."""
    mask = nav_wide.notna().to_numpy()
    n = mask.shape[0]
    has_any = mask.any(axis=0)
    last_idx = (n - 1) - np.argmax(mask[::-1], axis=0)
    out = np.where(has_any, (n - 1) - last_idx, np.inf).astype(float)
    return pd.Series(out, index=nav_wide.columns, name="stale_days")


def last_valid_dates(nav_wide: pd.DataFrame) -> pd.Series:
    mask = nav_wide.notna().to_numpy()
    n = mask.shape[0]
    has_any = mask.any(axis=0)
    last_idx = (n - 1) - np.argmax(mask[::-1], axis=0)
    idx = pd.to_datetime(nav_wide.index)
    vals = np.where(has_any, idx.to_numpy()[np.clip(last_idx, 0, n - 1)], np.datetime64("NaT", "ns"))
    return pd.Series(vals, index=nav_wide.columns, name="last_valid_date")


def flag_data_stale(
    nav_wide: pd.DataFrame,
    max_stale_days: int = 2,
    reference: pd.Timestamp | None = None,
    cfg: dict[str, Any] | None = None,
) -> pd.Series:
    """NAV, referans güne (verilmezse panelin son günü) göre > max_stale_days iş günü eski → data_stale=True. Askı değildir."""
    ref = (
        pd.Timestamp(reference).normalize() if reference is not None else pd.Timestamp(nav_wide.index.max()).normalize()
    )
    behind = business_days_behind(last_valid_dates(nav_wide), ref, cfg)
    return pd.Series(behind > max_stale_days, index=nav_wide.columns, name="data_stale")


def flag_bad_nav(nav_wide: pd.DataFrame, jump_threshold: float = 0.25) -> pd.DataFrame:
    """NaN/0 ve |günlük getiri| > eşik olan hücreleri bayraklar (doldurmaz)."""
    nonpos = (nav_wide <= 0).fillna(False)
    ret = np.log(nav_wide.where(nav_wide > 0)).diff()
    jump = ret.abs() > jump_threshold
    flags = pd.DataFrame({"nonpositive": nonpos.stack(), "jump": jump.stack()})
    return flags[flags["nonpositive"] | flags["jump"]]


# ---- durum metni --------------------------------------------------------------------------------
def _status_series(fund_master: pd.DataFrame) -> pd.Series:
    fm = fund_master.drop_duplicates("fund_code").set_index("fund_code")
    s = fm["tefas_status"] if "tefas_status" in fm.columns else pd.Series("", index=fm.index)
    return s.fillna("").astype(str).map(tr_fold)


def trade_status_masks(fund_master: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(can_buy, can_sell) — fund_code indeksli bool. Boş metin → ikisi True."""
    s = _status_series(fund_master)
    can_buy = ~s.str.contains("|".join(_BUY_CLOSED), regex=True)
    can_sell = ~s.str.contains("|".join(_SELL_CLOSED), regex=True)
    return can_buy.rename("can_buy"), can_sell.rename("can_sell")


def strict_trade_status_masks(fund_master: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Strict execution parser: recognized TEFAS statuses only; unknown values fail closed.

    Returns (can_buy, can_sell, recognized), indexed by fund_code. Legacy
    ``trade_status_masks`` intentionally retains its historical blank-is-open rule.
    """
    status = _status_series(fund_master).str.strip()
    buy_closed = status.str.contains(r"alımına kapalı", regex=True, na=False)
    sell_closed = status.str.contains(r"bozumuna kapalı", regex=True, na=False)
    both_closed = status.str.contains(r"alımına ve bozumuna kapalı|bozumuna ve alımına kapalı", regex=True, na=False)
    buy_closed |= both_closed
    sell_closed |= both_closed
    not_trading = status.str.contains(r"işlem görmüyor", regex=True, na=False)
    known_open = status.str.contains(r"işlem görüyor|işlem yapılabilir", regex=True, na=False)
    buy_open = status.str.contains(r"alımına açık", regex=True, na=False)
    sell_open = status.str.contains(r"bozumuna açık", regex=True, na=False)
    recognized = buy_closed | sell_closed | not_trading | known_open | buy_open | sell_open
    can_buy = recognized & (known_open | buy_open | sell_closed) & ~buy_closed & ~not_trading
    can_sell = recognized & (known_open | sell_open | buy_closed) & ~sell_closed & ~not_trading
    return can_buy.rename("can_buy"), can_sell.rename("can_sell"), recognized.rename("status_known")


def execution_status_masks(
    fund_master: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Status-text execution truth plus an audit mask for stale stored direction flags.

    The legacy text parser supplies direction-specific semantics; unknown status remains fail-closed
    via the strict recognizer. Stored ``can_buy``/``can_sell`` values are diagnostic only.
    """
    # Give each row a unique temporary code so date×fund panels retain date-specific status.
    rows = fund_master.reset_index(drop=True).copy()
    rows["fund_code"] = np.arange(len(rows)).astype(str)
    buy_by_row, sell_by_row = trade_status_masks(rows)
    strict_buy_by_row, strict_sell_by_row, known_by_row = strict_trade_status_masks(rows)
    can_buy = buy_by_row & strict_buy_by_row & known_by_row
    can_sell = sell_by_row & strict_sell_by_row & known_by_row

    mismatch = pd.Series(False, index=rows.index, dtype=bool)
    for column, expected in (("can_buy", can_buy), ("can_sell", can_sell)):
        if column in fund_master:
            stored = fund_master.reset_index(drop=True)[column]
            mismatch |= stored.isna() | stored.ne(expected)
        else:
            mismatch |= True
    return (
        can_buy.rename("can_buy"),
        can_sell.rename("can_sell"),
        known_by_row.rename("status_known"),
        mismatch.rename("status_flag_mismatch"),
    )


def flag_suspended_status(fund_master: pd.DataFrame) -> pd.Series:
    """Resmî askı: alıma VE bozuma kapalı. 'İşlem görmüyor' (platformda satılmıyor) askı sayılmaz."""
    s = _status_series(fund_master)
    return (s.str.contains("alımına kapalı") & s.str.contains("bozumuna kapalı")).rename("suspended")


# ---- özet ----------------------------------------------------------------------------------------
def quality_summary(
    nav_wide: pd.DataFrame,
    fund_master: pd.DataFrame,
    cfg: dict[str, Any],
    asof: str | pd.Timestamp | None = None,
    eligible: pd.Series | None = None,
    held: pd.Series | list[str] | None = None,
    lookback_days: int = 30,
) -> dict[str, Any]:
    """Sabah mesajı ve runs kaydı için özet. Yalnızca sayılar ve fon kodları; tutar yok."""
    u = cfg["legs"]["tefas"]["universe"]
    if nav_wide.empty:
        return {"ok": False, "reason": "NAV verisi yok", "source_stale": True}
    max_stale = int(u.get("max_stale_days", 2))
    src_stale_days = int(u.get("source_stale_days", 2))
    reference = expected_last_nav_date(asof, cfg)
    recent = nav_wide.tail(lookback_days)
    elig_idx = eligible[eligible.astype(bool)].index if eligible is not None else recent.columns
    elig_cols = [c for c in recent.columns if c in set(elig_idx)] or list(recent.columns)
    held_codes = set(held.index[held.astype(bool)] if isinstance(held, pd.Series) else (held or []))

    panel_last = pd.Timestamp(recent.index.max()).normalize()
    fresh_days_behind = float(business_days_behind(pd.Series([panel_last]), reference, cfg)[0])
    source_stale = fresh_days_behind >= src_stale_days

    # Fon bazlı: referans = panelin çoğunluk tarihi (askı/gecikme tespiti); kaynak bazlı: referans = beklenen etiket
    behind = pd.Series(business_days_behind(last_valid_dates(recent[elig_cols]), panel_last, cfg), index=elig_cols)
    fresh_ratio = float((behind <= 0).mean()) if len(behind) else 0.0
    data_stale = behind > max_stale
    bad = flag_bad_nav(recent)

    susp_all = flag_suspended_status(fund_master)
    scope = [c for c in susp_all.index if c in set(elig_cols) or c in held_codes]
    suspended = susp_all.reindex(scope).fillna(False).astype(bool)
    can_buy, can_sell = trade_status_masks(fund_master)

    source_stale = source_stale or fresh_ratio < float(u.get("min_universe_fresh_ratio", 0.95))
    ok = not source_stale
    return {
        "ok": ok,
        "reference_date": str(reference.date()),
        "latest_date": str(panel_last.date()),
        "fresh_days_behind": fresh_days_behind,
        "source_stale": bool(source_stale),
        "n_funds_nav": int(recent.shape[1]),
        "n_eligible": int(len(elig_cols)),
        "fresh_ratio": round(fresh_ratio, 4),
        "n_stale": int((behind > 0).sum()),
        "n_data_stale": int(data_stale.sum()),
        "data_stale_codes": data_stale[data_stale].index.tolist()[:20],
        "n_suspended": int(suspended.sum()),
        "suspended_codes": suspended[suspended].index.tolist()[:20],
        "n_bad_cells": int(len(bad)),
        "n_status_closed": int((~can_buy).sum()),
        "n_can_buy": int(can_buy.sum()),
        "n_can_sell": int(can_sell.sum()),
        "n_fund_master": int(len(fund_master)),
        "n_info_ok": int(fund_master["info_ok"].sum()) if "info_ok" in fund_master else None,
        "n_hist_ok": int(fund_master["hist_ok"].sum()) if "hist_ok" in fund_master else None,
    }
