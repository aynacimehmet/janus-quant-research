import numpy as np
import pandas as pd

from janus.features.macro import cpi_yoy_pit, pit_daily
from janus.models.regime import JumpModel, Scaler, rank_states, rank_to_level


def test_jump_model_recovers_two_regimes():
    rng = np.random.default_rng(0)
    T = 600
    true = np.r_[np.zeros(200), np.ones(250), np.zeros(150)].astype(int)
    X = rng.normal(0, 1, (T, 3)) + np.where(true[:, None] == 1, 2.5, -2.5)
    Z = Scaler.fit(X).transform(X)
    jm = JumpModel(k=2, jump_penalty=20.0, n_init=3).fit(Z)
    lab = jm.labels_
    agree = max((lab == true).mean(), (lab != true).mean())
    assert agree > 0.95
    assert (np.diff(lab) != 0).sum() <= 4  # ceza: az geçiş
    online = jm.predict_online(Z[-50:], s_prev=int(lab[-51]))
    assert (online == lab[-50:]).mean() > 0.9


def test_rank_states_and_levels():
    labels = np.array([0, 0, 1, 1, 2, 2])
    excess = np.array([0.01, 0.02, -0.02, -0.01, 0.0, 0.0])
    ranks = rank_states(labels, excess, 3)
    assert ranks[0] == 2 and ranks[1] == 0 and ranks[2] == 1
    assert (
        rank_to_level(ranks[0], 3) == "full"
        and rank_to_level(ranks[1], 3) == "low"
        and rank_to_level(ranks[2], 3) == "medium"
    )
    assert rank_to_level(1, 2) == "full" and rank_to_level(0, 2) == "low"


def test_pit_daily_respects_available_from():
    cal = pd.bdate_range("2026-01-01", "2026-03-31")
    long = pd.DataFrame(
        {
            "series": ["cpi_index"] * 3,
            "date": pd.to_datetime(["2025-12-01", "2026-01-01", "2026-02-01"]),
            "value": [100.0, 103.0, 106.0],
            "available_from": pd.to_datetime(["2026-01-05", "2026-02-05", "2026-03-06"]),
        }
    )
    d = pit_daily(long, cal)
    assert np.isnan(d.loc["2026-01-02", "cpi_index"]) and d.loc["2026-01-05", "cpi_index"] == 100.0
    assert d.loc["2026-02-04", "cpi_index"] == 100.0 and d.loc["2026-02-05", "cpi_index"] == 103.0
    long12 = pd.DataFrame(
        {
            "series": "cpi_index",
            "date": pd.date_range("2025-01-01", periods=14, freq="MS"),
            "value": 100 * 1.03 ** np.arange(14),
            "available_from": pd.date_range("2025-02-05", periods=14, freq="MS"),
        }
    )
    yoy = cpi_yoy_pit(long12, pd.bdate_range("2026-01-01", "2026-04-01"))
    assert abs(yoy.loc["2026-03-02"] - (1.03**12 - 1)) < 1e-9
