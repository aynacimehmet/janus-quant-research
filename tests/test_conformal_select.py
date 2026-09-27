"""S3b-4 testleri: ConformalHRP (lower skoru, kill-switch, tax_penalty) + seçim invariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _features import make_features
from _panel import make_panel
from janus.backtest.costs import meta_for_synthetic
from janus.backtest.engine import BacktestConfig, StrategyContext, run_backtest
from janus.data.quality import next_business_day
from janus.models.conformal import calibrate_predictions, coverage_report
from janus.models.walkforward import run_walkforward
from janus.strategies.conformal_select import (
    ConformalHRP,
    candidate_counts,
    kill_switch_flags,
    selected_mask_from_history,
)
from janus.strategies.portfolio import MomentumHRP

MK = {"num_boost_round": 20}


@pytest.fixture(scope="module")
def s3b():
    """Sentetik panelde WF + CQR. Bu panelde y medyanı negatif → lower hep < 0 (kanıt yok → nakit;
    ADR-18 davranışı). Seçim testleri için cal_pos (hepsi pozitif) ve cal_mixed (F0–F3 pozitif) türetilir."""
    f = make_features(n_funds=8, days=600, seed=1)
    preds = run_walkforward(f, model_kwargs=MK)
    cal = calibrate_predictions(preds, f, miscoverage_target=0.2, gamma=0.05)
    nav = make_panel(n_funds=8, days=600, seed=1)
    cal_pos = cal.copy()
    cal_pos["lower"] = cal_pos["lower"].abs() + 0.01
    cal_mixed = cal.copy()
    neg = cal_mixed["fund_code"].isin(["F4", "F5", "F6", "F7"])
    cal_mixed.loc[neg, "lower"] = -0.01
    cal_mixed.loc[~neg, "lower"] = cal_mixed.loc[~neg, "lower"].abs() + 0.01
    return f, preds, cal, cal_pos, cal_mixed, nav


def _ctx(nav, meta, date):
    d = pd.Timestamp(date)
    if d in nav.index:
        pos = nav.index.get_loc(d)
    elif d == next_business_day(nav.index[-1]):
        pos = len(nav.index) - 1
    else:
        raise ValueError(f"test context tarihi NAV/terminal karar takviminde değil: {d.date()}")
    return StrategyContext(
        idx=pos,
        date=d,
        nav=nav.iloc[:pos],
        meta=meta,
        weights_now=pd.Series(0.0, index=meta.index),
        equity=100.0,
        drawdown=0.0,
        calendar=nav.index,
    )


def test_scores_lower_positive_only(s3b):
    f, _, _, _, cal_mixed, nav = s3b
    meta = meta_for_synthetic(list(nav.columns))
    d = pd.Timestamp(cal_mixed["decision_at"].max())
    s = ConformalHRP(calibrated=cal_mixed, kill_switch=False).scores(_ctx(nav, meta, d))
    day = cal_mixed[pd.to_datetime(cal_mixed["decision_at"]) == d]
    assert (s > 0).all()  # lower <= 0 aday değil (ADR-18)
    assert set(s.index) == set(day.loc[day["lower"] > 0, "fund_code"]) == {"F0", "F1", "F2", "F3"}


def test_no_candidates_all_cash(s3b):
    _, _, cal, _, _, nav = s3b
    cal0 = cal.copy()
    cal0["lower"] = -0.01  # kanıt yok → nakit
    cash_r = pd.Series(0.0003, index=nav.index)
    res = run_backtest(
        nav,
        meta_for_synthetic(list(nav.columns)),
        ConformalHRP(calibrated=cal0, kill_switch=False),
        BacktestConfig(initial_capital=100, warmup_days=260),
        cash_returns=cash_r,
        cash_nav=(1 + cash_r).cumprod(),
    )
    assert not res.weights.empty
    risky = res.weights.drop(columns=["CASH_PROXY"], errors="ignore")
    assert (risky.sum(axis=1) < 1e-9).all()
    assert np.allclose(res.equity.to_numpy(), res.cash_index.to_numpy(), rtol=1e-9)


def test_kill_switch_flags_and_fallback(s3b):
    f, _, cal, _, _, nav = s3b
    dec = pd.DatetimeIndex(np.sort(pd.to_datetime(cal["decision_at"].unique())))
    mid = dec[len(dec) // 2]
    # kötü model: sabit dev q'lar → pinball(model) > baseline → True
    bad = pd.DataFrame(
        {
            "decision_at": cal["decision_at"],
            "fund_code": cal["fund_code"],
            "q10": 10.0,
            "q50": 10.0,
            "q90": 10.0,
            "model_id": cal["model_id"],
            "train_max_t": cal["train_max_t"],
        }
    )
    flags_bad = kill_switch_flags(bad, f)
    assert bool(flags_bad.loc[mid]) is True
    # iyi model: q'lar y'ye çok yakın → False
    good = cal[["decision_at", "fund_code", "model_id", "train_max_t"]].merge(
        f[["decision_at", "fund_code", "y"]], on=["decision_at", "fund_code"], how="left"
    )
    good["q10"], good["q50"], good["q90"] = good["y"] - 0.001, good["y"], good["y"] + 0.001
    good = good.drop(columns=["y"])
    flags_good = kill_switch_flags(good, f)
    assert bool(flags_good.loc[mid]) is False
    # strateji: bayraklı günde momentum fallback + sayaç
    meta = meta_for_synthetic(list(nav.columns))
    st = ConformalHRP(calibrated=cal, predictions=bad, features=f)
    ref = MomentumHRP(n=st.n, eligible=st.eligible, founders=st.founders, tax_aware=st.tax_aware)
    ctx = _ctx(nav, meta, mid)
    pd.testing.assert_series_equal(st.scores(ctx), ref.scores(ctx), check_names=False)
    assert st.fallback_days == 1  # scores() çağrısı bayraklı günü sayar
    st(ctx)
    assert st.fallback_days == 2


def test_future_perturbation_selection(s3b):
    _, _, _, cal, _, nav = s3b
    meta = meta_for_synthetic(list(nav.columns))
    t0 = pd.Timestamp(cal["decision_at"].max()) - pd.Timedelta(days=90)
    cal2 = cal.copy()
    cal2.loc[pd.to_datetime(cal2["decision_at"]) > t0, "lower"] *= 1.5
    d = pd.Timestamp(cal["decision_at"].min()) + pd.Timedelta(days=60)
    if d >= t0:
        d = pd.Timestamp(cal["decision_at"].min())
    s1 = ConformalHRP(calibrated=cal, kill_switch=False).scores(_ctx(nav, meta, d))
    s2 = ConformalHRP(calibrated=cal2, kill_switch=False).scores(_ctx(nav, meta, d))
    pd.testing.assert_series_equal(s1, s2, check_exact=True)


def test_tax_penalty_is_ranking_only(s3b):
    _, _, _, cal, _, nav = s3b
    meta = meta_for_synthetic(list(nav.columns), tax_rate=0.175)
    d = pd.Timestamp(cal["decision_at"].max())
    ctx = _ctx(nav, meta, d)
    s0 = ConformalHRP(calibrated=cal, kill_switch=False).scores(ctx)
    s1 = ConformalHRP(calibrated=cal, kill_switch=False, tax_penalty=True).scores(ctx)
    common = s0.index.intersection(s1.index)
    assert np.allclose(s1[common].to_numpy(), s0[common].to_numpy() * 0.825)
    # ağırlık toplamı cezayla değişmez (tavanlar aynı); yalnız sıralama/aday kümesi değişebilir
    w0 = ConformalHRP(calibrated=cal, kill_switch=False)(ctx)[0]
    w1 = ConformalHRP(calibrated=cal, kill_switch=False, tax_penalty=True)(ctx)[0]
    assert w0.sum() <= 1 + 1e-9 and w1.sum() <= 1 + 1e-9


def test_history_and_selected_coverage(s3b):
    _, _, _, cal, _, nav = s3b
    meta = meta_for_synthetic(list(nav.columns))
    cash_r = pd.Series(0.0003, index=nav.index)
    st = ConformalHRP(calibrated=cal, kill_switch=False)
    run_backtest(
        nav,
        meta,
        st,
        BacktestConfig(initial_capital=100, warmup_days=260),
        cash_returns=cash_r,
        cash_nav=(1 + cash_r).cumprod(),
    )
    assert st._history
    mask = selected_mask_from_history(cal, st._history)
    assert mask.any()
    cov = coverage_report(cal, mask)
    assert len(cov) and (cov["n_rows"] > 0).all()


def test_candidate_counts(s3b):
    _, _, _, cal, _, _ = s3b
    c = candidate_counts(cal)
    assert (c > 0).all()
    assert c.index.is_monotonic_increasing


def test_conformal_hrp_validation():
    with pytest.raises(ValueError):
        ConformalHRP(calibrated=pd.DataFrame({"decision_at": [], "fund_code": [], "lower": []}), kill_switch=False)
    with pytest.raises(ValueError):
        ConformalHRP(calibrated=pd.DataFrame({"a": [1]}), kill_switch=False)
    with pytest.raises(ValueError, match="predictions"):
        ConformalHRP(
            calibrated=pd.DataFrame({"decision_at": [1], "fund_code": ["F"], "lower": [0.1]}), kill_switch=True
        )


def test_calibrated_whitelist_no_leak(s3b):
    """F-06: ConformalHRP calibrated içindeki y/upper stratejiye sızmaz; yalnız lower kullanılır."""
    _, _, _, cal, _, nav = s3b
    cal_extra = cal.copy()
    cal_extra["y"] = 999.0
    cal_extra["upper"] = 999.0
    st = ConformalHRP(calibrated=cal_extra, kill_switch=False)
    d = pd.Timestamp(cal["decision_at"].max())
    s = st.scores(_ctx(nav, meta_for_synthetic(list(nav.columns)), d))
    assert (s > 0).all() or s.empty
    assert set(s.index) <= set(cal_extra.loc[cal_extra["lower"] > 0, "fund_code"])
