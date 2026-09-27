"""S3b-0b: parquet anlık görüntüsünden salt-okunur Store — okuma birebir, yazma engelli, CLI bağlantısı."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from loguru import logger
from pandas.testing import assert_frame_equal

from janus.cli import _store
from janus.data.ingest_tefas import ingest_tefas
from janus.data.store import Store
from janus.data.tefas_client import FakeTefasClient

_TABLES = ("fund_master", "fund_nav", "macro", "runs")


def _populate_and_export(tmp_path, cfg) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Sentetik depo doldur → export → (curated dizini, orijinal nav_wide, orijinal latest_fund_master)."""
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=6, days=300), cfg, mode="initial", snapshot_date="2026-09-22")
    st.log_run("r1", "nightly", datetime(2026, 9, 22, 23, 30), "ok", {"n": 1})
    curated = tmp_path / "curated"
    st.export_parquet(curated)
    nav, fm = st.nav_wide(), st.latest_fund_master()
    st.close()
    return curated, nav, fm


def test_from_parquet_reads_same_data(tmp_path, cfg):
    curated, nav, fm = _populate_and_export(tmp_path, cfg)
    snap = Store.from_parquet(curated)
    try:
        assert snap.read_only is True
        assert isinstance(snap.snapshot_asof, pd.Timestamp)  # datetime, None değil
        assert_frame_equal(snap.nav_wide(), nav)
        assert_frame_equal(snap.latest_fund_master(), fm)
    finally:
        snap.close()


def test_export_writes_snapshot_asof_iso_in_runs_summary(tmp_path, cfg):
    """0c-5: export sonrası runs.summary.snapshot_asof ISO; diğer summary alanları korunur."""
    curated, _, _ = _populate_and_export(tmp_path, cfg)
    st = Store(tmp_path / "t.duckdb")
    try:
        run = st.last_run("nightly")
        assert run["summary"]["n"] == 1  # mevcut alan korunur
        ts = pd.Timestamp(run["summary"]["snapshot_asof"])
        assert ts == Store.from_parquet(curated).snapshot_asof or ts.tz_convert(None) is not None
    finally:
        st.close()


def test_from_parquet_blocks_all_writes(tmp_path, cfg):
    curated, nav, fm = _populate_and_export(tmp_path, cfg)
    snap = Store.from_parquet(curated)
    try:
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(write_fund_master\)"):
            snap.write_fund_master(fm)
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(write_fund_master_rows\)"):
            snap.write_fund_master_rows(fm)
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(upsert_nav\)"):
            snap.upsert_nav(nav.reset_index().rename(columns={"index": "date"}))
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(upsert_macro\)"):
            snap.upsert_macro(pd.DataFrame(columns=["series", "date", "value", "available_from", "published_at"]))
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(log_run\)"):
            snap.log_run("r2", "test", datetime.now(), "ok", {})
        with pytest.raises(RuntimeError, match=r"yazma engellendi \(export_parquet\)"):
            snap.export_parquet(tmp_path / "out")
    finally:
        snap.close()


def test_from_parquet_missing_file_raises_with_name(tmp_path, cfg):
    curated, _, _ = _populate_and_export(tmp_path, cfg)
    (curated / "fund_nav.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="fund_nav.parquet"):
        Store.from_parquet(curated)


def test_snapshot_asof_is_max_mtime(tmp_path, cfg):
    curated, _, _ = _populate_and_export(tmp_path, cfg)
    import os

    old = datetime(2020, 1, 1, 12, 0).timestamp()
    for t in _TABLES:
        os.utime(curated / f"{t}.parquet", (old, old))
    os.utime(curated / "fund_nav.parquet", (old + 100, old + 100))
    snap = Store.from_parquet(curated)
    try:
        assert snap.snapshot_asof == pd.Timestamp.fromtimestamp(old + 100)
    finally:
        snap.close()


def test_cli_store_uses_parquet_snapshot(monkeypatch, tmp_path, cfg):
    """Tam parquet seti varsa _store(read_only=True) → from_parquet."""
    curated, _, _ = _populate_and_export(tmp_path, cfg)
    monkeypatch.setattr("janus.cli.store_path", lambda c: tmp_path / "janus.duckdb")
    monkeypatch.setattr("janus.cli.load_config", lambda: {"store": {"parquet_dir": str(curated)}})
    st = _store(dry_run=False, read_only=True)
    try:
        assert isinstance(st, Store) and st.read_only is True and st.snapshot_asof is not None
    finally:
        st.close()


def test_export_snapshot_cli_refreshes_all_parquet_views(monkeypatch, tmp_path, cfg):
    from typer.testing import CliRunner

    from janus.cli import app

    db = tmp_path / "janus.duckdb"
    curated = tmp_path / "curated"
    st = Store(db)
    st.upsert_nav(
        pd.DataFrame(
            {
                "fund_code": ["SYN"],
                "date": [pd.Timestamp("2026-09-24")],
                "price": [1.0],
                "published_at": [pd.Timestamp("2026-09-24 08:00")],
            }
        )
    )
    st.log_run("export-test", "test", datetime(2026, 9, 25), "ok", {})
    st.export_parquet_atomic(curated)
    st.upsert_nav(
        pd.DataFrame(
            {
                "fund_code": ["SYN"],
                "date": [pd.Timestamp("2026-09-25")],
                "price": [1.01],
                "published_at": [pd.Timestamp("2026-09-25 08:00")],
            }
        )
    )
    st.close()
    monkeypatch.setattr("janus.cli._store", lambda dry_run=False, read_only=False: Store(db))
    monkeypatch.setattr("janus.cli.load_config", lambda: cfg)
    monkeypatch.setattr("janus.cli._parquet_root", lambda _cfg, dry_run=False: curated)

    result = CliRunner().invoke(app, ["export-snapshot"])

    assert result.exit_code == 0, result.output
    from janus.cli import _parquet_dir

    active = _parquet_dir(cfg)
    assert all((active / f"{table}.parquet").is_file() for table in _TABLES)
    snapshot = Store.from_parquet(active)
    source = Store(db, read_only=True)
    assert snapshot.last_run("test") is not None
    assert snapshot.last_run("test")["summary"] == source.last_run("test")["summary"]
    assert_frame_equal(snapshot.nav_long(), source.nav_long())
    snapshot.close()
    source.close()


def test_export_snapshot_cli_failure_preserves_previous_generation(monkeypatch, tmp_path, cfg):
    from typer.testing import CliRunner

    from janus.cli import app

    db = tmp_path / "janus.duckdb"
    curated = tmp_path / "curated"
    st = Store(db)
    st.log_run("export-failure", "test", datetime(2026, 9, 25), "ok", {"keep": True})
    st.upsert_nav(
        pd.DataFrame(
            {
                "fund_code": ["SYN"],
                "date": [pd.Timestamp("2026-09-24")],
                "price": [1.0],
                "published_at": [pd.Timestamp("2026-09-24 08:00")],
            }
        )
    )
    st.export_parquet_atomic(curated)
    old_summary = st.last_run("test")["summary"]
    st.close()
    monkeypatch.setattr("janus.cli._store", lambda dry_run=False, read_only=False: Store(db))
    monkeypatch.setattr("janus.cli.load_config", lambda: cfg)
    monkeypatch.setattr("janus.cli._parquet_root", lambda _cfg, dry_run=False: curated)
    original_export = Store.export_parquet

    def partial_then_fail(store, stage):
        original_export(store, stage)
        raise RuntimeError("synthetic incomplete export")

    monkeypatch.setattr(Store, "export_parquet", partial_then_fail)
    result = CliRunner().invoke(app, ["export-snapshot"])

    assert result.exit_code == 1
    assert "SNAPSHOT EXPORT FAILED" in result.output
    from janus.cli import _parquet_dir

    active = _parquet_dir(cfg)
    assert all((active / f"{table}.parquet").is_file() for table in _TABLES)
    snapshot = Store.from_parquet(active)
    assert len(snapshot.nav_long()) == 1
    snapshot.close()
    source = Store(db, read_only=True)
    assert source.last_run("test")["summary"] == old_summary
    source.close()


def test_cli_store_falls_back_without_parquet(monkeypatch, tmp_path):
    """Parquet yok/yarım → uyarı + DuckDB salt-okunur fallback (from_parquet çağrılmaz)."""
    calls: list[dict] = []
    pq_calls: list[Path] = []

    class FakeStore:
        def __init__(self, path, read_only=False):
            calls.append({"path": Path(path), "read_only": read_only})

        @classmethod
        def from_parquet(cls, d):
            pq_calls.append(Path(d))
            raise AssertionError("from_parquet çağrılmamalı")

    monkeypatch.setattr("janus.cli.store_path", lambda c: tmp_path / "janus.duckdb")
    monkeypatch.setattr("janus.cli.load_config", lambda: {"store": {"parquet_dir": str(tmp_path / "bos")}})
    monkeypatch.setattr("janus.data.store.Store", FakeStore)

    records: list[str] = []
    hid = logger.add(records.append, level="WARNING")
    try:
        st = _store(dry_run=False, read_only=True)
    finally:
        logger.remove(hid)
    assert isinstance(st, FakeStore)
    assert calls == [{"path": tmp_path / "janus.duckdb", "read_only": True}]
    assert not pq_calls
    assert any("parquet anlık görüntüsü" in m for m in records)


def test_from_parquet_snapshot_immutable_while_writer_appends(tmp_path, cfg):
    """Kabul (S3b-0b): yazıcı süreç depoyu güncellerken parquet anlık görüntüsü değişmez."""
    cfg["ingest"]["fund_types"] = ["YAT"]
    p = tmp_path / "t.duckdb"
    st = Store(p)
    ingest_tefas(st, FakeTefasClient(n=6, days=300), cfg, mode="initial", snapshot_date="2026-09-22")
    st.export_parquet(tmp_path / "curated")
    st.close()
    snap = Store.from_parquet(tmp_path / "curated")
    before = snap.nav_wide()
    writer = Store(p)
    try:
        writer.upsert_nav(
            pd.DataFrame(
                {
                    "fund_code": ["F00"],
                    "date": [pd.Timestamp("2026-09-23")],
                    "price": [1.23],
                    "published_at": [pd.Timestamp("2026-09-23 09:00")],
                }
            )
        )
    finally:
        writer.close()
    assert snap.nav_wide().equals(before)
    assert len(snap.latest_fund_master()) == 6
    snap.close()
