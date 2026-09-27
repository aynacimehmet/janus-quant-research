from janus.backtest.runner import run_suite
from janus.data.ingest_tefas import ingest_tefas
from janus.data.store import Store
from janus.data.tefas_client import FakeTefasClient


def test_suite_quick(tmp_path, cfg):
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=700), cfg, mode="initial", snapshot_date="2026-09-22")
    cfg["backtest"] = {"initial_capital": 100, "warmup_days": 260, "cash_tax_rate": 0.175}
    md, path = run_suite(st, cfg, top_n=3, out_dir=tmp_path / "reports", root=None, quick=True)
    for section in (
        "Yıllık getiriler",
        "Sürtünme ayrışımı",
        "TEFAS/BES işlem komisyonu ADR-27 ile 0",
        "Tutulan fon türleri",
        "12 başlangıç tarihi",
        "Maliyet / valör stresi",
        "PBO",
    ):
        assert section in md
    assert path.exists()
