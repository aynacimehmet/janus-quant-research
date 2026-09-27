"""B2: momentum ile seçim (sıralama tamponu, kurucu/küme sınırı) + HRP ağırlık + tavanlar. B3 = B2 + kapı."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from janus.backtest.engine import StrategyContext
from janus.strategies.baselines import momentum_scores
from janus.strategies.hrp import apply_caps, cluster_labels, hrp_weights, ledoit_wolf_cov, select_with_limits


@dataclass
class MomentumHRP:
    n: int = 10
    lookback: int = 252
    skip: int = 21
    cov_days: int = 126
    buffer_mult: float = 2.0  # elde tutulan fon, sıralamada ≤ buffer_mult·n ise tutulur
    tax_aware: bool = True  # skor × (1 − stopaj)
    max_fund: float = 0.25
    max_founder: float = 0.30
    max_per_founder: int = 3
    max_per_cluster: int = 3
    cluster_distance: float = 0.4
    linkage: str = "single"
    eligible: pd.Series | None = None
    founders: pd.Series | None = None  # fund_code → kurucu
    refresh_every: int = 1  # kaç rebalance'ta bir seçim + HRP yeniden hesaplanır (3 = çeyreklik)
    smooth_lambda: float = 0.0  # hedef ağırlık EWMA: w = λ·w_önceki + (1−λ)·w_yeni (turnover azaltır)
    ks2: bool = (
        False  # kill-switch v2 (S3b-5-3): seçilen fonların olgunlaşmış son 63g ortalama nakit-fazlası < 0 → NAKİT
    )
    ks2_features: pd.DataFrame | None = None  # cash_excess_21 kolonunu içeren features çerçevesi
    name: str = "B2_momentum_hrp"
    _last: dict = field(default_factory=dict, repr=False)
    _calls: int = field(default=0, repr=False)
    _prev: pd.Series | None = field(default=None, repr=False)
    _ks2_mean: pd.DataFrame | None = field(default=None, repr=False)
    _ks2_count: int = field(default=0, repr=False)

    def scores(self, ctx: StrategyContext) -> pd.Series:
        s = momentum_scores(ctx.nav, self.lookback, self.skip)
        if self.eligible is not None:
            s = s.where(self.eligible.reindex(s.index).fillna(False).astype(bool))
        if self.tax_aware:
            tax = pd.Series(ctx.meta.tax_rate, index=ctx.meta.index).reindex(s.index).fillna(0.175)
            s = s * (1.0 - tax)
        return s

    def __call__(self, ctx: StrategyContext) -> tuple[pd.Series, float]:
        self._calls += 1
        if self._prev is not None and self.refresh_every > 1 and (self._calls - 1) % self.refresh_every != 0:
            return self._prev.copy(), 1.0  # ara aylarda hedef sabit; motor yalnızca sapma eşiğini uygular
        w = self._compute(ctx)
        if self._prev is not None and self.smooth_lambda > 0 and not w.empty:
            idx = w.index.union(self._prev.index)
            w = self.smooth_lambda * self._prev.reindex(idx).fillna(0.0) + (1 - self.smooth_lambda) * w.reindex(
                idx
            ).fillna(0.0)
            w = apply_caps(w[w > 1e-4], self.max_fund, self.max_founder, self.founders)
        self._prev = w.copy()
        return w, 1.0

    @property
    def ks2_days(self) -> int:
        """Kill-switch v2'nin nakde düşürdüğü rebalance gün sayısı."""
        return self._ks2_count

    def _ks2_table(self) -> pd.DataFrame:
        """Olgunlaşmış 63g ortalama nakit-fazlası: rolling(63).mean().shift(23) — D gününde
        satır pos_D, değer t ∈ [pos_D−86, pos_D−23] penceresinin ortalaması (label_available_at ≤ D)."""
        if self._ks2_mean is None:
            w = self.ks2_features.pivot_table(
                index="feature_asof", columns="fund_code", values="cash_excess_21", aggfunc="last"
            ).sort_index()
            self._ks2_mean = w.rolling(63, min_periods=63).mean().shift(23)
        return self._ks2_mean

    def _ks2_to_cash(self, ctx: StrategyContext, chosen: list[str]) -> bool:
        """Kill-switch v2: referans nakit — seçilen fonların olgunlaşmış son 63 iş günü ortalama
        nakit-fazlası < 0 → momentum/conformal değil NAKİTe düş."""
        if not self.ks2 or not chosen:
            return False
        t = self._ks2_table()
        d = pd.Timestamp(ctx.date)
        if d not in t.index:
            return False
        vals = t.loc[d, [c for c in chosen if c in t.columns]].dropna()
        if vals.empty:
            return False
        return bool(float(vals.mean()) < 0.0)

    def _compute(self, ctx: StrategyContext) -> pd.Series:
        s = self.scores(ctx)
        ranked = s.dropna().sort_values(ascending=False)
        if ranked.empty:
            return pd.Series(dtype=float)
        held = set(ctx.weights_now[ctx.weights_now > 1e-6].index)
        rank = pd.Series(np.arange(1, len(ranked) + 1), index=ranked.index)
        keep = {c for c in held if c in rank.index and rank[c] <= self.buffer_mult * self.n}
        pool = ranked.head(int(max(3 * self.n, self.n + len(keep)))).index.union(list(keep))
        rets = np.log(ctx.nav[pool]).diff().iloc[-self.cov_days :]
        rets = rets.loc[:, rets.notna().mean() >= 0.8]
        if rets.shape[1] == 0:
            return pd.Series(dtype=float)
        cov = ledoit_wolf_cov(rets)
        clusters = pd.Series(cluster_labels(cov, self.cluster_distance, self.linkage), index=rets.columns)
        chosen = select_with_limits(
            ranked.reindex(rets.columns),
            self.n,
            self.founders,
            clusters,
            self.max_per_founder,
            self.max_per_cluster,
            keep,
        )
        if not chosen:
            return pd.Series(dtype=float)
        if self._ks2_to_cash(ctx, chosen):  # kill-switch v2: momentum değil NAKİTe düş
            self._ks2_count += 1
            self._last = {"date": ctx.date, "chosen": [], "clusters": []}
            return pd.Series(dtype=float)
        idx = [list(rets.columns).index(c) for c in chosen]
        w = pd.Series(hrp_weights(cov[np.ix_(idx, idx)], self.linkage), index=chosen)
        w = apply_caps(w, self.max_fund, self.max_founder, self.founders)
        self._last = {"date": ctx.date, "chosen": chosen, "clusters": clusters.reindex(chosen).tolist()}
        return w
