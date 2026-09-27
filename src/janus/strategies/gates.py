"""Kapılar (maruziyet): R0 kural (baselines.RuleGate), R1-jump, R1-HMM, R2 nakit-hisse makası. Hepsi as-of: yalnızca ctx.date öncesi veri."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from janus.backtest.engine import StrategyContext
from janus.models.regime import JumpModel, Scaler, rank_states, rank_to_level


@dataclass
class SpreadGate:
    """R2: son `window` günde hisse endeksi getirisi − nakit getirisi > +marj → tam; < −marj → düşük; arası → orta."""

    inner: object
    equity_index: pd.Series
    cash_index: pd.Series
    levels: dict[str, float]
    window: int = 63
    margin: float = 0.02
    name: str = "B3_spread_gate"

    def __call__(self, ctx: StrategyContext) -> tuple[pd.Series, float]:
        w, _ = self.inner(ctx)
        e = self.equity_index[self.equity_index.index < ctx.date].dropna()
        c = self.cash_index[self.cash_index.index < ctx.date].dropna()
        if len(e) <= self.window or len(c) <= self.window:
            return w, self.levels["medium"]
        spread = float(np.log(e.iloc[-1] / e.iloc[-self.window - 1]) - np.log(c.iloc[-1] / c.iloc[-self.window - 1]))
        level = "full" if spread > self.margin else ("low" if spread < -self.margin else "medium")
        return w, self.levels[level]


@dataclass
class RegimeGate:
    """R1: özellik matrisi (günlük, as-of) üzerinde jump model (veya HMM) — her rebalance'ta D-1'e kadar yeniden tahmin,
    durumlar eğitim penceresindeki nakit-fazlası getiriye göre sıralanır, çevrimiçi tahminle güncel durum → seviye."""

    inner: object
    features: pd.DataFrame  # takvim × özellik (D'ye kadar hesaplı; kullanımda < ctx.date kesilir)
    excess_returns: pd.Series  # günlük hisse endeksi − nakit (log)
    levels: dict[str, float]
    model: str = "jump"  # jump | hmm
    k: int = 3
    jump_penalty: float = 50.0
    min_train: int = 252
    refit_every: int = 1  # kaç rebalance'ta bir yeniden tahmin
    name: str = "B3_jump_gate"
    _calls: int = field(default=0, repr=False)
    _state: dict = field(default_factory=dict, repr=False)
    _cols: list = field(default_factory=list, repr=False)

    def _fit(self, upto: pd.Timestamp) -> None:
        X = self.features[self.features.index < upto].dropna(how="all").dropna(axis=1, how="all")
        X = X.iloc[-2520:]  # ≤ 10 yıl
        self._cols = list(X.columns)
        ex = self.excess_returns.reindex(X.index).fillna(0.0).to_numpy()
        if len(X) < self.min_train:
            self._state = {}
            return
        sc = Scaler.fit(X.to_numpy(float))
        Z = sc.transform(X.to_numpy(float))
        if self.model == "hmm":
            from hmmlearn.hmm import GaussianHMM  # noqa: PLC0415

            hmm = GaussianHMM(n_components=self.k, covariance_type="diag", n_iter=100, random_state=0).fit(Z)
            labels = hmm.predict(Z)
            self._state = {"scaler": sc, "hmm": hmm, "ranks": rank_states(labels, ex, self.k), "last": int(labels[-1])}
        else:
            jm = JumpModel(k=self.k, jump_penalty=self.jump_penalty).fit(Z)
            self._state = {
                "scaler": sc,
                "jm": jm,
                "ranks": rank_states(jm.labels_, ex, self.k),
                "last": int(jm.labels_[-1]),
            }

    def current_level(self, ctx: StrategyContext) -> str:
        if not self._state:
            return "medium"
        X = self.features[self.features.index < ctx.date].dropna(how="all")
        if X.empty:
            return "medium"
        z = self._state["scaler"].transform(X[self._cols].iloc[[-1]].to_numpy(float))
        if "hmm" in self._state:
            s = int(self._state["hmm"].predict(z)[0])
        else:
            s = int(self._state["jm"].predict_online(z, s_prev=self._state["last"])[0])
        self._state["last"] = s
        return rank_to_level(self._state["ranks"][s], self.k)

    def __call__(self, ctx: StrategyContext) -> tuple[pd.Series, float]:
        w, _ = self.inner(ctx)
        if (self._calls % self.refit_every) == 0:
            self._fit(ctx.date)
        self._calls += 1
        return w, self.levels[self.current_level(ctx)]
