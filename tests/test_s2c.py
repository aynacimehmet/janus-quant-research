import pandas as pd

from _panel import make_panel  # noqa: E402
from janus.backtest.costs import meta_for_synthetic
from janus.backtest.engine import BacktestConfig, run_backtest
from janus.backtest.robustness import parameter_sweep
from janus.strategies.baselines import TopNMomentum
from janus.strategies.portfolio import MomentumHRP


def test_quarterly_refresh_and_smoothing_cut_turnover():
    nav = make_panel(n_funds=12, days=800, drift=0.0003)
    meta = meta_for_synthetic(list(nav.columns))
    cfg = BacktestConfig(warmup_days=260)
    founders = pd.Series({c: f"K{i % 4}" for i, c in enumerate(nav.columns)})
    base = run_backtest(nav, meta, MomentumHRP(n=6, founders=founders), cfg)
    slow = run_backtest(nav, meta, MomentumHRP(n=6, founders=founders, refresh_every=3, smooth_lambda=0.5), cfg)
    assert slow.turnover < base.turnover
    assert (slow.weights.max(axis=1) <= 0.25 + 1e-9).all()
    assert (slow.weights.T.groupby(founders).sum().T.max(axis=1) <= 0.30 + 1e-6).all()


def test_sweep_with_cfg_schedule():
    nav = make_panel(n_funds=6, days=700, drift=0.0003)
    df, info = parameter_sweep(
        nav,
        meta_for_synthetic(list(nav.columns)),
        lambda **p: TopNMomentum(max_weight=0.5, **p),
        {"n": [3], "cfg:schedule": ["monthly_first_business_day", "quarterly_first_business_day"]},
        BacktestConfig(warmup_days=260, sensitivity=True),
        pd.Series(0.0004, index=nav.index),
    )
    assert len(df) == 2 and set(df["cfg:schedule"]) == {"monthly_first_business_day", "quarterly_first_business_day"}
    assert (
        df.loc[df["cfg:schedule"].str.startswith("quarterly"), "turnover"].item()
        < df.loc[df["cfg:schedule"].str.startswith("monthly"), "turnover"].item()
    )
