"""S3b-0: salt-okunur Store — okuma çalışır, yazma engellenir, açılış yan etkisiz ve kilit paylaşılır."""

from datetime import datetime
from pathlib import Path

import pytest

from janus.cli import _store
from janus.data.ingest_tefas import ingest_tefas
from janus.data.store import Store
from janus.data.tefas_client import FakeTefasClient


def _populate(path, cfg) -> None:
    cfg["ingest"]["fund_types"] = ["YAT"]
    st = Store(path)
    ingest_tefas(st, FakeTefasClient(n=6, days=300), cfg, mode="initial", snapshot_date="2026-09-22")
    st.close()


def test_read_only_store_reads_and_blocks_writes(tmp_path, cfg):
    p = tmp_path / "t.duckdb"
    _populate(p, cfg)

    ro = Store(p, read_only=True)
    try:
        assert ro.read_only is True
        fm = ro.latest_fund_master()
        assert len(fm) == 6
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(write_fund_master\)"):
            ro.write_fund_master(fm)
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(write_fund_master_rows\)"):
            ro.write_fund_master_rows(fm)
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(upsert_nav\)"):
            ro.upsert_nav(ro.nav_long())
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(upsert_macro\)"):
            ro.upsert_macro(ro.nav_long())
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(log_run\)"):
            ro.log_run("r1", "test", datetime.now(), "ok", {})
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(export_parquet\)"):
            ro.export_parquet(tmp_path / "out")
    finally:
        ro.close()


def test_read_only_missing_file_is_side_effect_free(tmp_path):
    parent = tmp_path / "yeni"
    with pytest.raises(FileNotFoundError):
        Store(parent / "yok.duckdb", read_only=True)
    assert not parent.exists()  # salt-okunur açılış dizin oluşturmaz


def test_two_read_only_stores_coexist(tmp_path, cfg):
    """İki analiz komutu (ör. gate-suite + backtest) yazma kilidi almadan aynı anda okur."""
    p = tmp_path / "t.duckdb"
    _populate(p, cfg)

    a = Store(p, read_only=True)
    b = Store(p, read_only=True)
    try:
        assert len(a.latest_fund_master()) == len(b.latest_fund_master()) == 6
    finally:
        a.close()
        b.close()


def test_cli_store_passes_read_only_to_store(monkeypatch, tmp_path):
    """`janus.cli._store(read_only=True)` bayrağı Store'a iletir; dry_run=False yol değiştirmez."""
    target = tmp_path / "janus.duckdb"
    calls: list[dict] = []

    class FakeStore:
        def __init__(self, path, read_only=False):
            calls.append({"path": Path(path), "read_only": read_only})

    monkeypatch.setattr("janus.cli.store_path", lambda cfg: target)
    # boş parquet dizini → tam set yok → fallback (gerçek data/curated'dan bağımsız)
    monkeypatch.setattr("janus.cli.load_config", lambda: {"store": {"parquet_dir": str(tmp_path)}})
    monkeypatch.setattr("janus.data.store.Store", FakeStore)

    st = _store(dry_run=False, read_only=True)

    assert isinstance(st, FakeStore)
    assert calls == [{"path": target, "read_only": True}]
    # dry_run=False → gerçek depo yolu aynen geçer; `_dryrun` dosya adı uygulanmaz
    assert calls[0]["path"].name == "janus.duckdb"
    assert not calls[0]["path"].name.endswith("_dryrun.duckdb")
