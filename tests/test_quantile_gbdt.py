"""S3b-2 testleri: QuantileGBDT beyaz liste + tahmin şeması + metrikler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _features import make_features
from janus.features.fund_features import FEATURE_COLUMNS
from janus.models.metrics import daily_rank_ic, fva, pinball_loss
from janus.models.quantile_gbdt import QuantileGBDT


def test_quantile_fit_predict_and_whitelist():
    f = make_features()
    tr = f["feature_ready"] & f["eligible_at_decision"] & f["y"].notna()
    X, y = f.loc[tr, FEATURE_COLUMNS], f.loc[tr, "y"]
    m = QuantileGBDT(num_boost_round=20).fit(X, y)
    preds = m.predict(X.head(50))
    assert list(preds.columns) == ["q10", "q50", "q90"]
    with pytest.raises(ValueError, match="beyaz liste"):
        m.fit(X.assign(aum_now=1.0), y)
    with pytest.raises(ValueError, match="beyaz liste"):
        m.predict(X.head(5).assign(aum_now=1.0))


def test_pinball_manual():
    y = np.array([1.0, -1.0, 2.0])
    q = np.array([0.5, 0.5, 0.5])
    assert pinball_loss(y, q, 0.1) == pytest.approx((0.05 + 1.35 + 0.15) / 3)
    assert fva(0.2, 0.15) == pytest.approx(0.05)  # pozitif = iyileşme


def test_rank_ic_perfect():
    d = pd.Series(["2024-01-02"] * 5)
    pred = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    assert daily_rank_ic(pred, y, d) == pytest.approx(1.0)
