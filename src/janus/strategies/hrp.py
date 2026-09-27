"""Hierarchical Risk Parity (López de Prado 2016) + Janus kısıtları (fon %25, kurucu %30 / ≤ 3 fon, küme ≤ 3 fon).

Kovaryans: Ledoit–Wolf büzme (sklearn). Mesafe: sqrt(0.5·(1−ρ)). Linkage: single (klasik) veya ward.
Tavanlar iteratif: aşan fon/kurucu kırpılır, fazla diğerlerine oransal dağıtılır; dağıtılamayan kısım nakitte kalır (toplam < 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    from sklearn.covariance import LedoitWolf  # noqa: PLC0415

    x = returns.fillna(0.0).to_numpy(float)
    if x.shape[0] < 3:
        return np.cov(x, rowvar=False) if x.shape[0] > 1 else np.eye(x.shape[1])
    return LedoitWolf().fit(x).covariance_


def corr_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    c = cov / np.outer(d, d)
    return np.clip((c + c.T) / 2, -1.0, 1.0)


def hrp_linkage(cov: np.ndarray, method: str = "single") -> np.ndarray:
    corr = corr_from_cov(cov)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    return linkage(squareform(dist, checks=False), method=method)


def _ivp(cov: np.ndarray, idx: np.ndarray) -> np.ndarray:
    ivp = 1.0 / np.clip(np.diag(cov)[idx], 1e-16, None)
    return ivp / ivp.sum()


def _cluster_var(cov: np.ndarray, idx: np.ndarray) -> float:
    w = _ivp(cov, idx)
    sub = cov[np.ix_(idx, idx)]
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray, method: str = "single") -> np.ndarray:
    """Quasi-diagonalization (yaprak sırası) + recursive bisection. Toplam = 1."""
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    order = leaves_list(hrp_linkage(cov, method))
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            k = len(c) // 2
            a, b = c[:k], c[k:]
            va, vb = _cluster_var(cov, a), _cluster_var(cov, b)
            alpha = 1.0 - va / (va + vb) if (va + vb) > 0 else 0.5
            w[a] *= alpha
            w[b] *= 1.0 - alpha
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def apply_caps(
    w: pd.Series, max_fund: float = 0.25, max_group: float = 0.30, groups: pd.Series | None = None, iters: int = 100
) -> pd.Series:
    """Fon ve grup (kurucu) tavanları. Tavana çarpan fon/grup dondurulur (bir daha pay almaz), fazla serbest fonlara
    oransal dağıtılır; alıcı kalmazsa kalan nakitte kalır (toplam < 1). Dondurma kümeleri yalnızca büyüdüğü için yakınsar."""
    w = w.astype(float).clip(lower=0.0).copy()
    total = float(w.sum())
    if total <= 0:
        return w
    w = w / total
    g = groups.reindex(w.index).fillna("_") if groups is not None else pd.Series("_", index=w.index)
    frozen = pd.Series(False, index=w.index)
    for _ in range(iters):
        excess = 0.0
        over = w > max_fund + 1e-12
        if over.any():
            excess += float((w[over] - max_fund).sum())
            w[over] = max_fund
            frozen |= over
        gs = w.groupby(g).sum()
        for grp, tot in gs[gs > max_group + 1e-12].items():
            m = (g == grp).to_numpy()
            excess += float(tot - max_group)
            w[m] *= max_group / tot
            frozen |= pd.Series(m, index=w.index)
        if excess <= 1e-12:
            break
        recv = (~frozen) & (w > 0)
        if not recv.any():
            break  # kalan nakit
        w[recv] += excess * w[recv] / w[recv].sum()
    return w


def constrain_founder_targets(
    target: pd.Series,
    current: pd.Series,
    founders: pd.Series,
    max_weight: float = 0.30,
    max_funds: int = 3,
    protected: pd.Series | None = None,
) -> pd.Series:
    """Apply one portfolio-wide founder cap, preserving unsellable current exposure.

    `target` and `current` are whole-portfolio weights. Protected positions (for
    example min-hold or can_sell=False) are floors; new exposure is scaled or
    removed when founder weight/count capacity is exhausted. Unallocated weight
    intentionally remains cash.
    """
    codes = target.index.union(current.index)
    desired = target.reindex(codes).fillna(0.0).astype(float).clip(lower=0.0)
    held = current.reindex(codes).fillna(0.0).astype(float).clip(lower=0.0)
    groups = founders.reindex(codes).astype("string").str.strip()
    locked = (
        protected.reindex(codes).fillna(False).astype(bool) if protected is not None else pd.Series(False, index=codes)
    )
    result = desired.copy()
    # Missing founder identity fails closed for additions but does not erase existing holdings.
    unknown = groups.isna() | groups.eq("")
    unknown_existing = bool((unknown & held.gt(1e-12)).any())
    result.loc[unknown & held.le(0)] = 0.0
    result.loc[unknown & held.gt(0)] = np.minimum(desired.loc[unknown & held.gt(0)], held.loc[unknown & held.gt(0)])
    known_groups = groups.loc[~unknown].dropna().unique()
    for founder in known_groups:
        members = groups.index[groups.eq(founder)]
        held_group = held.reindex(members)
        desired_group = desired.reindex(members)
        is_held = held_group > 1e-12
        # Assume no proposed sale has filled yet; all live positions consume weight/count capacity.
        base_weight = float(held_group.sum())
        base_count = int(is_held.sum())
        result.loc[members] = desired_group.where(is_held, 0.0).clip(upper=held_group)
        result.loc[members] = result.loc[members].where(~locked.reindex(members), np.maximum(desired_group, held_group))
        increments = (desired_group - held_group).clip(lower=0.0)
        new_funds = ~is_held
        new_candidates = increments.loc[new_funds & (increments > 1e-12)]
        slots = 0 if unknown_existing else max(0, int(max_funds) - base_count)
        if len(new_candidates) > slots:
            keep = new_candidates.sort_values(ascending=False, kind="stable").index[:slots]
            increments.loc[new_candidates.index.difference(keep)] = 0.0
        room = 0.0 if unknown_existing else max(0.0, float(max_weight) - base_weight)
        total = float(increments.sum())
        if total > room + 1e-12 and total > 0:
            increments *= room / total
        result.loc[members] += increments
        result.loc[members] = result.loc[members].where(~locked.reindex(members), held_group)
    return result


def cluster_labels(cov: np.ndarray, distance_threshold: float = 0.4, method: str = "single") -> np.ndarray:
    """Korelasyon kümeleri (mesafe eşiği: 0.4 ≈ ρ ≥ 0.68 aynı küme)."""
    if cov.shape[0] < 2:
        return np.zeros(cov.shape[0], dtype=int)
    return fcluster(hrp_linkage(cov, method), t=distance_threshold, criterion="distance")


def select_with_limits(
    scores: pd.Series,
    n: int,
    groups: pd.Series | None,
    clusters: pd.Series | None,
    max_per_group: int = 3,
    max_per_cluster: int = 3,
    keep: set[str] | None = None,
) -> list[str]:
    """Skora göre açgözlü seçim: kurucu ve küme başına adet sınırı; `keep` (elde tutulanlar) önce değerlendirilir."""
    ranked = scores.dropna().sort_values(ascending=False)
    if keep:
        held = [c for c in ranked.index if c in keep]
        ranked = pd.concat([ranked[held], ranked.drop(held)])
    chosen: list[str] = []
    gcount: dict = {}
    ccount: dict = {}
    for code in ranked.index:
        g = groups.get(code, "_") if groups is not None else "_"
        c = clusters.get(code, -1) if clusters is not None else -1
        if gcount.get(g, 0) >= max_per_group or ccount.get(c, 0) >= max_per_cluster:
            continue
        chosen.append(code)
        gcount[g] = gcount.get(g, 0) + 1
        ccount[c] = ccount.get(c, 0) + 1
        if len(chosen) >= n:
            break
    return chosen
