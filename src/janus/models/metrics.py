"""Tahmin metrikleri (S3b-2): pinball (α başına), günlük Spearman rank IC, FVA.

FVA = pinball_baseline − pinball_model (her baseline için ayrı; pozitif = iyileşme).
Aynı satır kümesi, aynı ufuk, aynı α (VALIDATION §1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pinball_loss(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    """Ortalama pinball: α(y−q) if y≥q else (1−α)(q−y)."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(q))
    if not ok.any():
        return float("nan")
    d = y[ok] - q[ok]
    return float(np.mean(np.where(d >= 0, alpha * d, (1 - alpha) * -d)))


def daily_rank_ic(pred: pd.Series, y: pd.Series, decision_at: pd.Series, min_funds: int = 5) -> float:
    """Günlük Spearman rank IC (q50 vs y, fonlar arası) ortalaması; yetersiz gün atlanır."""
    df = pd.DataFrame({"d": decision_at.to_numpy(), "p": pred.to_numpy(), "y": y.to_numpy()}).dropna()
    ics = []
    for _, g in df.groupby("d"):
        if len(g) < min_funds:
            continue
        rp, ry = g["p"].rank(), g["y"].rank()
        if rp.std() == 0 or ry.std() == 0:
            continue
        ics.append(np.corrcoef(rp, ry)[0, 1])
    return float(np.mean(ics)) if ics else float("nan")


def fva(pinball_baseline: float, pinball_model: float) -> float:
    """Forecast value added: baseline − model pinball; pozitif = model iyileşmesi."""
    return float(pinball_baseline - pinball_model)
