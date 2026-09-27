"""Rejim modelleri (S3a): kendi jump model uygulamamız (Nystrup ve ark.; DP ile kesin durum dizisi), HMM yedeği,
"nakit-hisse makası" kuralı. Anchored walk-forward: her rebalance'ta D-1'e kadar veriyle yeniden tahmin.

Jump model: min Σ‖x_t − μ_{s_t}‖² + λ·1[s_t ≠ s_{t−1}]; koordinat inişi (durum dizisi DP, merkezler ortalama).
Çevrimiçi tahmin: s_t = argmin_s ‖x_t − μ_s‖² + λ·1[s ≠ s_{t−1}].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> Scaler:
        return cls(np.nanmean(X, axis=0), np.nanstd(X, axis=0) + 1e-9)

    def transform(self, X: np.ndarray, clip: float = 3.0) -> np.ndarray:
        Z = (X - self.mean) / self.std
        return np.clip(np.nan_to_num(Z, nan=0.0), -clip, clip)


def _viterbi(C: np.ndarray, lam: float) -> np.ndarray:
    """C: T×k maliyet. Geçişte λ cezası (durum değişimi). En ucuz yolu döner."""
    T, k = C.shape
    V = np.empty((T, k))
    back = np.zeros((T, k), dtype=int)
    V[0] = C[0]
    ar = np.arange(k)
    for t in range(1, T):
        prev = V[t - 1]
        j = int(prev.argmin())
        switch = prev[j] + lam
        stay_better = prev <= switch
        V[t] = C[t] + np.where(stay_better, prev, switch)
        back[t] = np.where(stay_better, ar, j)
    s = np.empty(T, dtype=int)
    s[-1] = int(V[-1].argmin())
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s


@dataclass
class JumpModel:
    k: int = 2
    jump_penalty: float = 50.0
    n_init: int = 5
    max_iter: int = 30
    seed: int = 0
    centers_: np.ndarray | None = field(default=None, repr=False)
    labels_: np.ndarray | None = field(default=None, repr=False)
    objective_: float = np.inf

    def fit(self, Z: np.ndarray) -> JumpModel:
        rng = np.random.default_rng(self.seed)
        T = Z.shape[0]
        best = (np.inf, None, None)
        for _ in range(self.n_init):
            mu = Z[rng.choice(T, self.k, replace=False)]
            s_prev = None
            for _ in range(self.max_iter):
                C = ((Z[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
                s = _viterbi(C, self.jump_penalty)
                for j in range(self.k):
                    if (s == j).any():
                        mu[j] = Z[s == j].mean(axis=0)
                if s_prev is not None and np.array_equal(s, s_prev):
                    break
                s_prev = s
            C = ((Z[:, None, :] - mu[None, :, :]) ** 2).sum(-1)
            obj = float(C[np.arange(T), s].sum() + self.jump_penalty * (np.diff(s) != 0).sum())
            if obj < best[0]:
                best = (obj, mu.copy(), s.copy())
        self.objective_, self.centers_, self.labels_ = best
        return self

    def predict_online(self, Z: np.ndarray, s_prev: int | None = None) -> np.ndarray:
        """Sıralı çevrimiçi tahmin (geleceğe bakmaz)."""
        C = ((Z[:, None, :] - self.centers_[None, :, :]) ** 2).sum(-1)
        out = np.empty(len(Z), dtype=int)
        for t in range(len(Z)):
            if s_prev is None:
                s_prev = int(C[t].argmin())
            else:
                pen = np.full(self.k, self.jump_penalty)
                pen[s_prev] = 0.0
                s_prev = int((C[t] + pen).argmin())
            out[t] = s_prev
        return out


def rank_states(labels: np.ndarray, excess: np.ndarray, k: int) -> dict[int, int]:
    """Durumları eğitim penceresindeki ortalama nakit-fazlası getiriye göre sıralar: 0 = en kötü … k−1 = en iyi."""
    means = np.array([excess[labels == j].mean() if (labels == j).any() else -np.inf for j in range(k)])
    order = np.argsort(means)
    return {int(state): int(rank) for rank, state in enumerate(order)}


def rank_to_level(rank: int, k: int) -> str:
    if k <= 2:
        return "full" if rank == k - 1 else "low"
    if rank == k - 1:
        return "full"
    return "medium" if rank == k - 2 else "low"
