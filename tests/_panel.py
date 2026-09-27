"""Test yardımcısı: sentetik NAV paneli (paket değil; pytest tests/ dizinini sys.path'e ekler)."""

import numpy as np
import pandas as pd


def make_panel(n_funds=6, days=800, seed=0, drift=0.0004):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=days)
    r = rng.normal(drift, 0.01, size=(days, n_funds))
    r[:, 0] += 0.002  # 0 numaralı fon belirgin momentum
    return pd.DataFrame(10 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"F{i}" for i in range(n_funds)])
