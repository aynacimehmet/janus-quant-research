from janus.backtest.runner import run_baselines
from janus.data.ingest_tefas import ingest_tefas
from janus.data.store import Store
from janus.data.tefas_client import FakeTefasClient


def test_runner_end_to_end(tmp_path, cfg):
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=700), cfg, mode="initial", snapshot_date="2026-09-22")
    cfg["backtest"] = {"initial_capital": 100, "warmup_days": 260, "cash_tax_rate": 0.175}
    rows, path = run_baselines(st, cfg, top_n=3, out_dir=tmp_path / "reports")
    names = [r["strategy"] for r in rows]
    assert names[0] == "B0_cash" and names[-1] == "B3_hrp_gate" and len(names) == 8
    assert path is not None and "Yıllık getiriler" in path.read_text(encoding="utf-8")
