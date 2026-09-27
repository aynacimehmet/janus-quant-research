"""Performans metrikleri (vergi ve maliyet sonrası) ve deflated Sharpe."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

ANN = 252


def summary(
    equity: pd.Series, cash_index: pd.Series | None = None, taxes: float = 0.0, fees: float = 0.0, turnover: float = 0.0
) -> dict:
    r = np.log(equity).diff().dropna()
    if cash_index is not None:
        rc = np.log(cash_index.reindex(equity.index)).diff().reindex(r.index).fillna(0.0)
    else:
        rc = pd.Series(0.0, index=r.index)
    ex = r - rc
    years = len(r) / ANN
    cagr = float(equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    vol = float(r.std(ddof=1) * math.sqrt(ANN))
    sharpe = float(ex.mean() / ex.std(ddof=1) * math.sqrt(ANN)) if ex.std(ddof=1) > 0 else np.nan
    down = ex[ex < 0]
    sortino = float(ex.mean() / down.std(ddof=1) * math.sqrt(ANN)) if len(down) > 1 and down.std(ddof=1) > 0 else np.nan
    dd = 1 - equity / equity.cummax()
    mdd = float(dd.max())
    cash_cagr = (
        float(cash_index.iloc[-1] / cash_index.iloc[0]) ** (1 / years) - 1
        if cash_index is not None and years > 0
        else np.nan
    )
    return {
        "cagr": cagr,
        "cash_cagr": cash_cagr,
        "excess_cagr": cagr - cash_cagr if cash_index is not None else np.nan,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "calmar": cagr / mdd if mdd > 0 else np.nan,
        "turnover_per_year": turnover / years if years > 0 else np.nan,
        "taxes_pct_of_final": taxes / float(equity.iloc[-1]),
        "fees_pct_of_final": fees / float(equity.iloc[-1]),
        "n_days": int(len(r)),
        "skew": float(skew(ex)) if len(ex) > 3 else np.nan,
        "kurt": float(kurtosis(ex, fisher=False)) if len(ex) > 3 else np.nan,
    }


def deflated_sharpe(
    sr: float, n_trials: int, sr_var: float, n_obs: int, skew_: float = 0.0, kurt_: float = 3.0
) -> float:
    """Bailey & López de Prado (2014): SR'nin, N deneme arasından seçilmiş olmanın etkisi düşüldükten sonra
    anlamlı olma olasılığı (0–1). sr: yıllık değil, GÖZLEM başına Sharpe. sr_var: denemeler arası SR varyansı."""
    if n_trials <= 1 or n_obs <= 1:
        return float("nan")
    gamma = 0.5772156649
    e = math.e
    sr0 = math.sqrt(max(sr_var, 1e-12)) * (
        (1 - gamma) * norm.ppf(1 - 1 / n_trials) + gamma * norm.ppf(1 - 1 / (n_trials * e))
    )
    denom = math.sqrt(max(1 - skew_ * sr + (kurt_ - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr0) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))
