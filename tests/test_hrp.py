import numpy as np
import pandas as pd
import pytest

from janus.strategies.hrp import (
    apply_caps,
    cluster_labels,
    constrain_founder_targets,
    hrp_weights,
    ledoit_wolf_cov,
    select_with_limits,
)


def _returns(seed=0, n=8, t=300):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, 0.01, (t, 2))
    load = np.array([[1, 0]] * 4 + [[0, 1]] * 4)  # iki blok (küme)
    r = f @ load.T + rng.normal(0, 0.004, (t, n))
    r[:, 0] *= 3  # F0 çok oynak → düşük HRP ağırlığı
    return pd.DataFrame(r, columns=[f"F{i}" for i in range(n)])


def test_hrp_weights_sum_and_risk_parity_direction():
    rets = _returns()
    cov = ledoit_wolf_cov(rets)
    w = hrp_weights(cov)
    assert abs(w.sum() - 1) < 1e-9 and (w > 0).all()
    assert w[0] < w[1:].mean()  # en oynak fon en küçük ağırlığı alır


def test_cluster_labels_two_blocks():
    lab = cluster_labels(ledoit_wolf_cov(_returns()), distance_threshold=0.6)
    assert len(set(lab[:4])) == 1 and len(set(lab[4:])) == 1 and lab[0] != lab[4]


def test_apply_caps_fund_and_founder():
    w = pd.Series([0.5, 0.2, 0.15, 0.1, 0.05], index=list("ABCDE"))
    g = pd.Series(["X", "X", "X", "Y", "Z"], index=list("ABCDE"))
    out = apply_caps(w, max_fund=0.25, max_group=0.30, groups=g)
    assert (out <= 0.25 + 1e-9).all()
    assert out.groupby(g).sum().max() <= 0.30 + 1e-9
    assert out.sum() <= 1 + 1e-9


def test_select_with_limits_group_and_cluster():
    scores = pd.Series({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0.5})
    groups = pd.Series({"A": "X", "B": "X", "C": "X", "D": "X", "E": "Y", "F": "Y"})
    clusters = pd.Series({"A": 1, "B": 1, "C": 1, "D": 2, "E": 2, "F": 3})
    chosen = select_with_limits(scores, n=5, groups=groups, clusters=clusters, max_per_group=3, max_per_cluster=2)
    assert chosen == ["A", "B", "D", "E", "F"]  # C: küme 1 dolu; D: X'in 3.'sü; F: küme 3
    kept = select_with_limits(scores, n=2, groups=None, clusters=None, keep={"F"})
    assert kept[0] == "F"  # elde tutulan önce değerlendirilir


def test_s5_6c2_combined_founder_cap_allows_three_and_rejects_fourth():
    codes = ["B0A", "B0B", "R1", "R2"]
    groups = pd.Series("PYSA", index=codes)
    target = pd.Series([0.10, 0.10, 0.10, 0.10], index=codes)
    result = constrain_founder_targets(target, pd.Series(dtype=float), groups)

    assert result["B0A"] > 0 and result["B0B"] > 0 and result["R1"] > 0
    assert result["R2"] == 0
    assert result.sum() == pytest.approx(0.30)


def test_s5_6c2_locked_min_hold_counts_and_unallocated_weight_is_cash():
    groups = pd.Series({"LOCK": "PYSA", "RISK": "PYSA", "OTHER": "PYSB"})
    current = pd.Series({"LOCK": 0.25})
    target = pd.Series({"LOCK": 0.25, "RISK": 0.10, "OTHER": 0.20})
    protected = pd.Series({"LOCK": True})
    result = constrain_founder_targets(target, current, groups, protected=protected)

    assert result["LOCK"] == pytest.approx(0.25)
    assert result["RISK"] == pytest.approx(0.05)
    assert result["OTHER"] == pytest.approx(0.20)
    assert result["LOCK"] + result["RISK"] <= 0.30 + 1e-12
    assert 1.0 - result.sum() == pytest.approx(0.50)


def test_s5_6c2_locked_overweight_is_not_increased_by_new_targets():
    groups = pd.Series({"LOCK": "PYSA", "NEW": "PYSA"})
    current = pd.Series({"LOCK": 0.32})
    target = pd.Series({"LOCK": 0.40, "NEW": 0.10})
    result = constrain_founder_targets(target, current, groups, protected=pd.Series({"LOCK": True}))

    assert result["LOCK"] == pytest.approx(0.32)
    assert result["NEW"] == pytest.approx(0.0)


def test_s5_6c2_unknown_founder_fails_closed_for_new_buy():
    target = pd.Series({"UNKNOWN": 0.20})
    result = constrain_founder_targets(target, pd.Series({"UNKNOWN": 0.10}), pd.Series(dtype="string"))
    assert result["UNKNOWN"] == pytest.approx(0.10)


def test_s5_6c2_unknown_existing_founder_blocks_other_new_buy():
    current = pd.Series({"UNKNOWN_HELD": 0.10})
    target = pd.Series({"UNKNOWN_HELD": 0.10, "KNOWN_NEW": 0.10})
    founders = pd.Series({"KNOWN_NEW": "PYSA"})
    result = constrain_founder_targets(target, current, founders)

    assert result["UNKNOWN_HELD"] == pytest.approx(0.10)
    assert result["KNOWN_NEW"] == pytest.approx(0.0)


def test_s5_6c2_unrelated_founders_preserve_targets():
    target = pd.Series({"A": 0.25, "B": 0.25, "C": 0.20})
    founders = pd.Series({"A": "PYSA", "B": "PYSB", "C": "PYSC"})

    result = constrain_founder_targets(target, pd.Series(dtype=float), founders)

    pd.testing.assert_series_equal(result, target)
