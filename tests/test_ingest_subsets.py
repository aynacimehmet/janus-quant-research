from janus.data.ingest_tefas import ingest_tefas
from janus.data.store import Store
from janus.data.tefas_client import FakeTefasClient


def test_only_failed_and_codes(tmp_path, cfg):
    cfg["ingest"]["fund_types"] = ["YAT"]
    st = Store(tmp_path / "t.duckdb")
    client = FakeTefasClient(n=12, days=300)
    r1 = ingest_tefas(st, client, cfg, mode="initial", snapshot_date="2026-09-22")
    assert r1["failures_info"] == ["F05:RuntimeError"]
    r2 = ingest_tefas(st, client, cfg, mode="incremental", snapshot_date="2026-09-22", only_failed=True)
    assert r2["n_processed"] == 1 and r2["failures_info"] == ["F05:RuntimeError"]
    assert len(st.latest_fund_master()) == 12  # alt küme koşusu snapshot'ı silmedi
    r3 = ingest_tefas(st, client, cfg, mode="incremental", snapshot_date="2026-09-22", codes=["F00", "f01"])
    assert r3["n_processed"] == 2 and r3["n_failures"] == 0
    assert len(st.latest_fund_master()) == 12


def test_parallel_workers_equivalent(tmp_path, cfg):
    cfg["ingest"]["fund_types"] = ["YAT"]
    cfg["ingest"]["workers"] = 3
    st = Store(tmp_path / "t.duckdb")
    r = ingest_tefas(st, FakeTefasClient(n=12, days=300), cfg, mode="initial", snapshot_date="2026-09-22")
    assert r["n_hist_ok"] == 12 and r["n_info_ok"] == 11
