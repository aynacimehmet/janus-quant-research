"""Teşhis: yıllık getiri kırılımı, tutulan fon türü dağılımı, vergi/komisyon sürtünmesi."""

from __future__ import annotations

import pandas as pd

from janus.backtest.engine import BacktestResult


def yearly_returns(equity: pd.Series) -> pd.Series:
    """Takvim yılı getirileri (yıl içindeki ilk → son değer)."""
    g = equity.groupby(equity.index.year)
    return (g.last() / g.first() - 1.0).rename("return")


def yearly_table(results: list[BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame({r.name: yearly_returns(r.equity) for r in results})


def holdings_by_type(result: BacktestResult, fund_master: pd.DataFrame, col: str = "umbrella_type") -> pd.Series:
    """Rebalance günlerindeki hedef ağırlıkların fon türüne göre ortalaması (nakit = 1 − toplam)."""
    if result.weights.empty:
        return pd.Series(dtype=float)
    fm = (
        fund_master.drop_duplicates("fund_code").set_index("fund_code")[col].reindex(result.weights.columns).fillna("?")
    )
    by = result.weights.T.groupby(fm).sum().T.mean()
    by["nakit"] = 1.0 - float(result.weights.sum(axis=1).mean())
    return by.sort_values(ascending=False)


def friction_decomposition(res_net: BacktestResult, res_gross: BacktestResult) -> dict:
    """Aynı strateji vergisiz/komisyonsuz koşuyla karşılaştırma → CAGR puanı cinsinden sürtünme."""
    years = len(res_net.equity) / 252

    def cagr(s: pd.Series) -> float:
        return float(s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1

    return {
        "cagr_net": cagr(res_net.equity),
        "cagr_gross": cagr(res_gross.equity),
        "friction_pts": cagr(res_gross.equity) - cagr(res_net.equity),
        "taxes_pct_final": res_net.taxes_paid / float(res_net.equity.iloc[-1]),
        "fees_pct_final": res_net.fees_paid / float(res_net.equity.iloc[-1]),
    }


def realized_gain_by_tax(result: BacktestResult) -> pd.DataFrame:
    if result.fills.empty:
        return pd.DataFrame()
    s = result.fills[result.fills["side"] == "SELL"]
    return s.groupby(s["tax"] > 0).agg(n=("code", "size"), gain=("realized_gain", "sum"), tax=("tax", "sum"))
