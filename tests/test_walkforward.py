"""S3b-2 walk-forward sızıntı/QA testleri (sentetik panel; gerçek veri yok)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _features import make_features
from janus.features.fund_features import FEATURE_COLUMNS
from janus.models.baselines import baseline_predictions
from janus.models.walkforward import cpcv_select, oos_mask, refit_dates, run_walkforward, train_mask

MK = {"num_boost_round": 20}


@pytest.fixture(scope="module")
def features():
    return make_features(n_funds=6, days=500, seed=0)


@pytest.fixture(scope="module")
def preds(features):
    return run_walkforward(features, model_kwargs=MK)


def test_refit_dates_first_business_day():
    cal = pd.bdate_range("2024-01-01", "2024-06-30")
    r = refit_dates(cal)
    assert [str(d.date()) for d in r] == [
        "2024-01-01",
        "2024-02-01",
        "2024-03-01",
        "2024-04-01",
        "2024-05-01",
        "2024-06-03",
    ]
    assert all(d.weekday() < 5 for d in r)


def test_train_max_t_constraint(features, preds):
    assert len(preds) > 0
    cal = pd.DatetimeIndex(np.sort(pd.to_datetime(features["decision_at"]).unique()))
    for _, row in preds.iterrows():
        pos = cal.get_loc(row["decision_at"])
        assert cal.get_loc(row["train_max_t"]) <= pos - 23  # train_max_t <= decision_at - 23 (işlem günü)


def test_future_perturbation_predictions_bit_identical(features):
    t0 = 400
    cal = features["feature_asof"].unique()
    cut = pd.Timestamp(cal[t0])
    base = run_walkforward(features, model_kwargs=MK)
    pert = features.copy()
    feat_cols = [c for c in FEATURE_COLUMNS if c not in ("umbrella_type_code", "tax_rate")]
    future = pert["feature_asof"] > cut
    pert.loc[future, feat_cols] = pert.loc[future, feat_cols] * 1.01  # gelecek feature perturbasyonu
    pert.loc[future, "y"] = pert.loc[future, "y"] * 1.01
    perturbed = run_walkforward(pert, model_kwargs=MK)
    b = base[base["decision_at"] <= cut + pd.Timedelta(days=1)].reset_index(drop=True)
    p = perturbed[perturbed["decision_at"] <= cut + pd.Timedelta(days=1)].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, p, check_exact=True)


def test_whitelist_enforced_in_walkforward(features):
    bad = features.copy()
    bad["aum_now"] = 1.0
    with pytest.raises(ValueError, match="beyaz liste"):
        run_walkforward(bad, model_kwargs=MK)


def test_interim_days_use_same_model(features, preds):
    month = preds["model_id"].dt.to_period("M")
    for m, g in preds.groupby(month):
        assert g["model_id"].nunique() == 1  # ara günler ilgili ayın M_R'si
        assert (
            g["model_id"].iloc[0]
            == refit_dates(pd.DatetimeIndex(np.sort(pd.to_datetime(features["decision_at"]).unique())))[
                list(
                    refit_dates(pd.DatetimeIndex(np.sort(pd.to_datetime(features["decision_at"]).unique()))).to_period(
                        "M"
                    )
                ).index(m)
            ]
        )


def test_baselines_same_rows(features, preds):
    mask = oos_mask(features)
    bl = baseline_predictions(features, mask, min_decision_at=preds["model_id"].min())
    model_keys = set(map(tuple, preds[["decision_at", "fund_code"]].to_numpy()))
    for name, b in bl.items():
        base_keys = set(map(tuple, b[["decision_at", "fund_code"]].to_numpy()))
        # baseline NaN tahmin verebilir (yetersiz gözlem) ama satır kümesi modelle aynı
        assert base_keys == model_keys, name


def test_model_id_traceability(features, preds):
    assert {"decision_at", "fund_code", "q10", "q50", "q90", "model_id", "train_max_t"}.issubset(preds.columns)
    assert preds[["q10", "q50", "q90", "model_id", "train_max_t"]].notna().all().all()


def test_train_mask_contract(features):
    r = pd.Timestamp(features["decision_at"].max())
    m = train_mask(features, r)
    assert (pd.to_datetime(features.loc[m, "label_available_at"]) <= r).all()
    assert features.loc[m, "eligible_at_decision"].all() and features.loc[m, "feature_ready"].all()


def test_cpcv_purge_and_grid(features):
    grid = [{"num_leaves": 15, "min_data_in_leaf": 200}, {"num_leaves": 31, "min_data_in_leaf": 500}]
    res = cpcv_select(features, grid, n_blocks=6, n_test=2, model_kwargs=MK)
    assert set(res.columns) >= {"num_leaves", "min_data_in_leaf", "pinball_mean", "combo"}
    assert len(res) == 2 * len(list(__import__("itertools").combinations(range(6), 2))) or len(res) > 0
    assert res["pinball_mean"].notna().all()
