"""Piyasa/stres özellikleri: hisse fonu endeks vekili, genişlik, askıdaki fon sayısı. Tümü D'ye kadar olan NAV'lardan; kullanım D-1."""

from __future__ import annotations

import numpy as np
import pandas as pd


def market_features(
    nav_wide: pd.DataFrame, equity_index: pd.Series, equity_cols: list[str], stale: np.ndarray | None = None
) -> pd.DataFrame:
    f = pd.DataFrame(index=nav_wide.index)
    li = np.log(equity_index.replace(0, np.nan))
    f["eq_trend63"] = li.diff(63)
    f["eq_trend252"] = li.diff(252)
    r = li.diff()
    f["eq_vol21"] = r.rolling(21).std() * np.sqrt(252)
    f["eq_vol63"] = r.rolling(63).std() * np.sqrt(252)
    eq = nav_wide[[c for c in equity_cols if c in nav_wide.columns]].ffill()
    if eq.shape[1] > 0:
        ma200 = eq.rolling(200, min_periods=120).mean()
        f["breadth200"] = (eq > ma200).sum(axis=1) / eq.notna().sum(axis=1).replace(0, np.nan)
    if stale is not None:
        f["n_suspended"] = stale.sum(axis=1)
    return f


def regime_feature_matrix(market: pd.DataFrame, macro: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    X = pd.concat([market, macro.reindex(market.index)], axis=1)
    if cols:
        X = X[[c for c in cols if c in X.columns]]
    return X
