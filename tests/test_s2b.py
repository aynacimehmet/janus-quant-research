import numpy as np
import pandas as pd

from _panel import make_panel  # noqa: E402
from janus.backtest.costs import meta_for_synthetic
from janus.backtest.diagnostics import holdings_by_type, yearly_returns
from janus.backtest.engine import BacktestConfig, rebalance_days, run_backtest
from janus.backtest.robustness import parameter_sweep, pbo_cscv, start_date_sweep, stressed_meta, suspension_scenario
from janus.strategies.baselines import TopNMomentum
from janus.strategies.portfolio import MomentumHRP


def test_quarterly_schedule():
    cal = pd.bdate_range("2024-01-01", "2024-12-31")
    d = rebalance_days(cal, "quarterly_first_business_day")
    assert [cal[i].strftime("%m-%d") for i in d] == ["01-01", "04-01", "07-01", "10-01"]


def test_buffer_reduces_turnover():
    nav = make_panel(n_funds=8, days=700, drift=0.0)
    meta = meta_for_synthetic(list(nav.columns))
    cfg = BacktestConfig(warmup_days=260)
    plain = run_backtest(nav, meta, TopNMomentum(n=3, max_weight=0.5), cfg)
    buff = run_backtest(nav, meta, TopNMomentum(n=3, max_weight=0.5, buffer_mult=2.0), cfg)
    assert buff.turnover < plain.turnover


def test_momentum_hrp_runs_and_respects_caps():
    nav = make_panel(n_funds=12, days=700, drift=0.0003)
    founders = pd.Series({c: f"K{i % 3}" for i, c in enumerate(nav.columns)})
    strat = MomentumHRP(n=6, founders=founders, max_fund=0.25, max_founder=0.30, max_per_founder=3)
    res = run_backtest(nav, meta_for_synthetic(list(nav.columns)), strat, BacktestConfig(warmup_days=260))
    assert res.n_rebalances > 10 and not res.weights.empty
    assert (res.weights.max(axis=1) <= 0.25 + 1e-9).all()
    per_founder = res.weights.T.groupby(founders).sum().T
    assert (per_founder.max(axis=1) <= 0.30 + 1e-6).all()
    assert ((res.weights > 0).T.groupby(founders).sum().T.max(axis=1) <= 3).all()


def test_yearly_and_holdings():
    nav = make_panel(n_funds=4, days=600, drift=0.0005)
    res = run_backtest(
        nav, meta_for_synthetic(list(nav.columns)), TopNMomentum(n=2, max_weight=0.5), BacktestConfig(warmup_days=260)
    )
    yr = yearly_returns(res.equity)
    assert len(yr) >= 2
    fm = pd.DataFrame({"fund_code": nav.columns, "umbrella_type": ["Hisse", "Hisse", "Borç", "Borç"]})
    hb = holdings_by_type(res, fm)
    assert "nakit" in hb.index and abs(hb.sum() - 1) < 1e-6


def test_start_sweep_and_stress():
    nav = make_panel(n_funds=6, days=800, drift=0.0003)
    meta = meta_for_synthetic(list(nav.columns))
    cfg = BacktestConfig(warmup_days=260, sensitivity=True)
    cash = pd.Series(0.0004, index=nav.index)
    sd = start_date_sweep(nav, meta, lambda: TopNMomentum(n=3, max_weight=0.5), cfg, cash, n_starts=3)
    assert len(sd) == 3 and {"cagr", "mdd"} <= set(sd.columns)
    m2 = stressed_meta(meta, extra_fee=0.001, valor_plus=1)
    assert (m2.sell_valor == meta.sell_valor + 1).all() and (m2.entry_fee > meta.entry_fee).all()
    founders = pd.Series({c: f"K{i % 2}" for i, c in enumerate(nav.columns)})
    sc = suspension_scenario(
        nav, meta, lambda: TopNMomentum(n=3, max_weight=0.5), cfg, cash, founders, months=3, seeds=2
    )
    assert len(sc) == 3 and sc.iloc[0]["scenario"] == "baz"


def test_pbo_random_is_high_and_sweep_runs():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (400, 10))  # gürültü: seçilen en iyi OOS'ta rastgele → PBO ≈ 0.5
    out = pbo_cscv(R, n_blocks=8)
    assert 0.2 <= out["pbo"] <= 0.8 and out["n_combos"] == 70
    nav = make_panel(n_funds=6, days=700, drift=0.0003)
    df, info = parameter_sweep(
        nav,
        meta_for_synthetic(list(nav.columns)),
        lambda **p: TopNMomentum(max_weight=0.5, **p),
        {"n": [2, 3], "lookback": [126, 252]},
        BacktestConfig(warmup_days=260, sensitivity=True),
        pd.Series(0.0004, index=nav.index),
    )
    assert len(df) == 4 and 0 <= info["deflated_sharpe_prob"] <= 1 and "pbo" in info
