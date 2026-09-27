import pandas as pd

from _panel import make_panel  # noqa: E402
from janus.backtest.costs import meta_for_synthetic
from janus.backtest.engine import BacktestConfig, run_backtest
from janus.backtest.gate_suite import run_gate_suite
from janus.data.ingest_evds import FakeMacro, ingest_macro
from janus.data.ingest_tefas import ingest_tefas
from janus.data.store import Store
from janus.data.tefas_client import FakeTefasClient
from janus.strategies.baselines import TopNMomentum
from janus.strategies.gates import RegimeGate, SpreadGate


def test_spread_gate_levels():
    nav = make_panel(n_funds=4, days=600, drift=0.0006)
    eq = nav.mean(axis=1)
    cash = pd.Series((1.0004) ** pd.RangeIndex(len(nav)), index=nav.index)
    levels = {"low": 0.3, "medium": 0.65, "full": 1.0}
    g = SpreadGate(inner=TopNMomentum(n=2, max_weight=0.5), equity_index=eq, cash_index=cash, levels=levels)
    res = run_backtest(nav, meta_for_synthetic(list(nav.columns)), g, BacktestConfig(warmup_days=260))
    assert set(res.events.loc[res.events["event"] == "rebalance", "exposure"].round(2)) <= {0.3, 0.65, 1.0}


def test_regime_gate_jump_runs_as_of():
    nav = make_panel(n_funds=4, days=700, drift=0.0004)
    eq = nav.mean(axis=1)
    X = pd.DataFrame({"t63": eq.pct_change(63), "v21": eq.pct_change().rolling(21).std()}, index=nav.index)
    excess = eq.pct_change().fillna(0.0) - 0.0004
    g = RegimeGate(
        inner=TopNMomentum(n=2, max_weight=0.5),
        features=X,
        excess_returns=excess,
        levels={"low": 0.3, "medium": 0.65, "full": 1.0},
        k=2,
        jump_penalty=10.0,
    )
    res = run_backtest(nav, meta_for_synthetic(list(nav.columns)), g, BacktestConfig(warmup_days=300))
    assert res.n_rebalances > 10 and set(res.events.loc[res.events["event"] == "rebalance", "exposure"].round(2)) <= {
        0.3,
        0.65,
        1.0,
    }


def test_gate_suite_quick(tmp_path, cfg):
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=14, days=800), cfg, mode="initial", snapshot_date="2026-09-22")
    ingest_macro(st, FakeMacro(), cfg)
    cfg["backtest"] = {"initial_capital": 100, "warmup_days": 260, "cash_tax_rate": 0.175}
    cfg.setdefault("regime", {})["hmm"] = False
    md, path = run_gate_suite(st, cfg, top_n=3, out_dir=tmp_path / "r", quick=True)
    assert "B3_R1_jump" in md and "B3_R2_spread" in md and "PBO" in md and path.exists()
