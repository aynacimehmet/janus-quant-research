"""Baseline zinciri (S2): B0 nakit fonu · B1 eşit ağırlık top-N momentum · gate R0 (S2b'de HRP ile B2/B3).

Tüm stratejiler yalnızca ctx.nav (≤ D-1) görür — as-of disiplini motor tarafından garanti edilir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from janus.backtest.engine import StrategyContext


@dataclass
class CashOnly:
    """B0: her şey nakit fonunda."""

    name: str = "B0_cash"

    def __call__(self, ctx: StrategyContext) -> tuple[pd.Series, float]:
        return pd.Series(dtype=float), 0.0


def momentum_scores(nav: pd.DataFrame, lookback: int = 252, skip: int = 21, min_history: int = 252) -> pd.Series:
    """12-1 momentum: log(NAV[-skip] / NAV[-lookback]); yetersiz geçmiş → NaN. Vektörize."""
    if len(nav) < lookback + 1:
        return pd.Series(np.nan, index=nav.columns)
    p_end = nav.iloc[-skip - 1] if skip > 0 else nav.iloc[-1]
    p_start = nav.iloc[-lookback - 1]
    valid = nav.iloc[-min_history:].notna().mean() >= 0.9
    score = np.log(p_end / p_start)
    return score.where(valid)


@dataclass
class TopNMomentum:
    """B1: momentum sıralamasında ilk N fon, eşit ağırlık; opsiyonel uygunluk maskesi (evren)."""

    n: int = 10
    lookback: int = 252
    skip: int = 21
    eligible: pd.Series | None = None  # fund_code → bool (Faz-1 evreni)
    max_weight: float = 0.25
    buffer_mult: float = 1.0  # > 1: elde tutulan fon sıralamada ≤ buffer_mult·n ise tutulur (turnover azalır)
    tax_aware: bool = False  # skor × (1 − stopaj)
    name: str = "B1_topN_momentum"

    def __call__(self, ctx: StrategyContext) -> tuple[pd.Series, float]:
        score = momentum_scores(ctx.nav, self.lookback, self.skip)
        if self.eligible is not None:
            score = score.where(self.eligible.reindex(score.index).fillna(False).astype(bool))
        if self.tax_aware:
            tax = pd.Series(ctx.meta.tax_rate, index=ctx.meta.index).reindex(score.index).fillna(0.175)
            score = score * (1.0 - tax)
        ranked = score.dropna().sort_values(ascending=False)
        if ranked.empty:
            return pd.Series(dtype=float), 0.0
        if self.buffer_mult > 1.0:
            rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
            held = [
                c
                for c in ctx.weights_now[ctx.weights_now > 1e-6].index
                if c in rank.index and rank[c] <= self.buffer_mult * self.n
            ]
            fill = [c for c in ranked.index if c not in held][: max(self.n - len(held), 0)]
            chosen = (held + fill)[: self.n]
        else:
            chosen = list(ranked.head(self.n).index)
        w = pd.Series(1.0 / len(chosen), index=chosen).clip(upper=self.max_weight)
        return w, 1.0


@dataclass
class RuleGate:
    """R0: endeks 200g ortalamasının üstünde VE gerçekleşen vol 80. yüzdeliğin altında → tam; biri → orta; hiçbiri → düşük.
    `index_nav`: piyasa göstergesi (örn. hisse fonlarının eşit ağırlıklı NAV'ı) — as-of için ctx tarihine kadar kesilir."""

    index_nav: pd.Series
    inner: object  # sarmalanan strateji (ağırlıkları üretir)
    levels: dict[str, float]
    ma_days: int = 200
    vol_days: int = 21
    vol_pct: float = 0.80
    name: str = "B3_rule_gate"

    def __call__(self, ctx: StrategyContext) -> tuple[pd.Series, float]:
        w, _ = self.inner(ctx)
        s = self.index_nav[self.index_nav.index < ctx.date].dropna()
        if len(s) < self.ma_days + self.vol_days:
            return w, self.levels["medium"]
        above = s.iloc[-1] > s.iloc[-self.ma_days :].mean()
        r = np.log(s).diff().dropna()
        vol = r.iloc[-self.vol_days :].std()
        hist_vol = r.rolling(self.vol_days).std().dropna()
        calm = vol < hist_vol.quantile(self.vol_pct)
        level = "full" if (above and calm) else ("medium" if (above or calm) else "low")
        return w, self.levels[level]
