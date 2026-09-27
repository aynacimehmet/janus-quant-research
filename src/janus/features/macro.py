"""PIT makro özellikleri: her gözlem yalnızca `available_from` gününden itibaren görünür (as-of disiplini)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from janus.data.store import Store


def macro_long(store: Store) -> pd.DataFrame:
    return store.con.execute("SELECT series, date, value, available_from FROM macro ORDER BY series, date").df()


def pit_daily(long: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Uzun makro tablosu → takvime hizalı geniş tablo; değer available_from'dan itibaren ileri doldurulur."""
    out = {}
    for name, g in long.groupby("series"):
        s = pd.Series(g["value"].to_numpy(float), index=pd.to_datetime(g["available_from"])).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        out[name] = s.reindex(calendar.union(s.index)).ffill().reindex(calendar)
    return pd.DataFrame(out, index=calendar)


def cpi_yoy_pit(long: pd.DataFrame, calendar: pd.DatetimeIndex, series: str = "cpi_index") -> pd.Series:
    """Aylık endeksten yıllık enflasyon; her ayın değeri o ayın available_from tarihinden itibaren görünür."""
    g = long[long["series"] == series].sort_values("date")
    if g.empty:
        return pd.Series(np.nan, index=calendar)
    idx = pd.Series(g["value"].to_numpy(float), index=pd.to_datetime(g["date"]))
    yoy = idx / idx.shift(12) - 1.0
    yoy.index = pd.to_datetime(g["available_from"])
    yoy = yoy[~yoy.index.duplicated(keep="last")].sort_index()
    return yoy.reindex(calendar.union(yoy.index)).ffill().reindex(calendar)


def macro_features(store: Store, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    long = macro_long(store)
    if long.empty:
        return pd.DataFrame(index=calendar)
    d = pit_daily(long, calendar)
    f = pd.DataFrame(index=calendar)
    if "policy_rate" in d:
        f["policy_rate"] = d["policy_rate"] / 100.0
        f["d_policy_63"] = f["policy_rate"] - f["policy_rate"].shift(63)
    f["cpi_yoy"] = cpi_yoy_pit(long, calendar)
    if "policy_rate" in f:
        f["real_rate"] = f["policy_rate"] - f["cpi_yoy"]
    if "usdtry" in d:
        lr = np.log(d["usdtry"]).diff()
        f["usdtry_ret63"] = np.log(d["usdtry"]).diff(63)
        f["usdtry_vol21"] = lr.rolling(21).std() * np.sqrt(252)
    return f
