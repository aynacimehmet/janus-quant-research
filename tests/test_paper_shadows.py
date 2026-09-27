from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from janus.data.store import Store
from janus.paper.core import _ensure_paper_tables
from janus.paper.shadows import run_shadows, snapshot_shadow_equity


def test_shadow_failure_is_recorded_without_blocking_other_snapshot(tmp_path, monkeypatch):
    import janus.paper.shadows as shadows

    store = Store(tmp_path / "shadows.duckdb")
    _ensure_paper_tables(store)
    monkeypatch.setattr(
        shadows,
        "prepare",
        lambda *_args: {
            "nav": pd.DataFrame(),
            "meta": pd.DataFrame(),
            "cash_returns": pd.Series(dtype=float),
            "cash_nav": pd.Series(dtype=float),
            "cash_category": None,
            "cash_rate": None,
        },
    )
    monkeypatch.setattr(shadows, "SHADOWS", {"broken": lambda *_args: "broken", "healthy": lambda *_args: "healthy"})
    monkeypatch.setattr(shadows.BacktestConfig, "from_cfg", lambda _cfg: object())

    def backtest(*args, **kwargs):
        if args[2] == "broken":
            raise ValueError("synthetic portfolio failure")
        return SimpleNamespace(equity=pd.Series([100.0], index=pd.to_datetime(["2026-09-24"])))

    monkeypatch.setattr(shadows, "run_backtest", backtest)
    result = run_shadows(store, {}, names=["broken", "healthy"])
    summary = snapshot_shadow_equity(store, result, pd.Timestamp("2026-09-24"))

    statuses = dict(
        store.con.execute("SELECT portfolio_name, status FROM paper_shadow_runs WHERE date='2026-09-24'").fetchall()
    )
    assert statuses == {"broken": "failed", "healthy": "ok"}
    assert summary["failed"] == 1
    assert store.con.execute(
        "SELECT equity FROM paper_equity WHERE portfolio_name='healthy' AND date='2026-09-24'"
    ).fetchone() == (100.0,)
    assert store.con.execute(
        "SELECT cash, receivables, risky_value, slot_value FROM paper_equity "
        "WHERE portfolio_name='healthy' AND date='2026-09-24'"
    ).fetchone() == (None, None, None, None)
    assert store.con.execute("SELECT count(*) FROM paper_equity WHERE portfolio_name='broken'").fetchone() == (0,)
    store.close()
