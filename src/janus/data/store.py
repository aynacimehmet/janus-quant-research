"""DuckDB tabanlı depo: PIT snapshot'lar (fund_master), NAV upsert, koşu kaydı, Parquet arşivi."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

_DDL = {
    "fund_master": """
        CREATE TABLE IF NOT EXISTS fund_master (
            snapshot_date DATE, fund_code VARCHAR, isin VARCHAR, name VARCHAR, fund_class VARCHAR,
            umbrella_type VARCHAR, category VARCHAR, founder_code VARCHAR, founder VARCHAR, manager VARCHAR,
            buy_valor INTEGER, sell_valor INTEGER, entry_fee DOUBLE, exit_fee DOUBLE,
            first_trading_time VARCHAR, last_trading_time VARCHAR,
            tax_category VARCHAR, withholding_rate DOUBLE, applied_fee DOUBLE, expense_ratio DOUBLE,
            aum_now DOUBLE, investor_count BIGINT, risk_value INTEGER, tefas_status VARCHAR, kap_link VARCHAR,
            first_nav_date DATE, last_nav_date DATE, n_nav INTEGER, info_ok BOOLEAN, hist_ok BOOLEAN,
            source_published_at TIMESTAMP, info_fetched BOOLEAN, last_success_at TIMESTAMP,
            last_success_source VARCHAR,
            can_buy BOOLEAN, can_sell BOOLEAN
        )""",
    "fund_nav": """
        CREATE TABLE IF NOT EXISTS fund_nav (
            fund_code VARCHAR, date DATE, price DOUBLE, published_at TIMESTAMP
        )""",
    "macro": """
        CREATE TABLE IF NOT EXISTS macro (
            series VARCHAR, date DATE, value DOUBLE, available_from DATE, published_at TIMESTAMP
        )""",
    "runs": """
        CREATE TABLE IF NOT EXISTS runs (
            run_id VARCHAR, kind VARCHAR, started_at TIMESTAMP, finished_at TIMESTAMP, status VARCHAR, summary VARCHAR
        )""",
}

_TABLES = tuple(_DDL)


def parquet_snapshot_complete(curated_dir: str | Path) -> bool:
    """`Store.from_parquet` için 4 tablo parquet'i de mevcut mu? (hızlı kontrol; hata fırlatmaz)"""
    d = Path(curated_dir)
    return all((d / f"{t}.parquet").is_file() for t in _TABLES)


class Store:
    """Tek dosyalık DuckDB deposu. Yazma işlemleri idempotent (aynı gün yeniden koşulabilir)."""

    snapshot_asof: pd.Timestamp | None = None  # yalnızca from_parquet örneklerinde dolu (S3b-0b)

    def __init__(self, path: str | Path, read_only: bool = False):
        """read_only=True: mevcut depoyu yazma kilidi olmadan açar; DDL/migration atlanır (şema hazır varsayılır)."""
        self.path = Path(path)
        self.read_only = read_only
        if read_only and not self.path.exists():
            raise FileNotFoundError(f"salt-okunur Store: depo dosyası bulunamadı: {self.path}")
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path), read_only=read_only)
        if not read_only:
            for ddl in _DDL.values():
                self.con.execute(ddl)
            self._migrate()

    @classmethod
    def from_parquet(cls, curated_dir: str | Path) -> Store:
        """Parquet anlık görüntüsünden salt-okunur Store (S3b-0b): 4 tablo in-memory DuckDB'ye VIEW.

        Nesne read_only=True'dur (tüm yazma metodları RuntimeError fırlatır); __init__ bypass edilir,
        dosya açma yan etkisi olmaz. snapshot_asof = 4 parquet'in max mtime'ı (yerel saat).
        """
        d = Path(curated_dir)
        missing = [f"{t}.parquet" for t in _TABLES if not (d / f"{t}.parquet").is_file()]
        if missing:
            raise FileNotFoundError(f"parquet anlık görüntüsü eksik: {', '.join(missing)} ({d})")
        self = cls.__new__(cls)
        self.path = d
        self.read_only = True
        self.snapshot_asof = max(pd.Timestamp.fromtimestamp((d / f"{t}.parquet").stat().st_mtime) for t in _TABLES)
        self.con = duckdb.connect(":memory:")
        for t in _TABLES:
            self.con.execute(f"CREATE VIEW {t} AS SELECT * FROM '{d / (t + '.parquet')}'")
        return self

    def _ensure_writable(self, operation: str) -> None:
        """Salt-okunur Store'da yazma denemelerini engeller; hata hangi işlemin engellendiğini söyler."""
        if self.read_only:
            raise RuntimeError(f"salt-okunur Store: yazma engellendi ({operation})")

    def _migrate(self) -> None:
        """Var olan depoya sonradan eklenen kolonlar (idempotent)."""
        self.con.execute("ALTER TABLE fund_master ADD COLUMN IF NOT EXISTS info_fetched BOOLEAN")
        self.con.execute("ALTER TABLE fund_master ADD COLUMN IF NOT EXISTS last_success_at TIMESTAMP")
        self.con.execute("ALTER TABLE fund_master ADD COLUMN IF NOT EXISTS last_success_source VARCHAR")
        self.con.execute("ALTER TABLE fund_master ADD COLUMN IF NOT EXISTS can_buy BOOLEAN")
        self.con.execute("ALTER TABLE fund_master ADD COLUMN IF NOT EXISTS can_sell BOOLEAN")

    def backfill_profile_success_at(self, dry_run: bool = False) -> int:
        """Fill missing success times from proxy ingest time, recording that provenance explicitly."""
        self._ensure_writable("backfill_profile_success_at")
        predicate = "info_fetched IS TRUE AND last_success_at IS NULL AND source_published_at IS NOT NULL"
        count = int(self.con.execute(f"SELECT count(*) FROM fund_master WHERE {predicate}").fetchone()[0])
        if not dry_run and count:
            self.con.execute(
                f"""UPDATE fund_master SET last_success_at = source_published_at,
                last_success_source = 'proxy_ingest_time' WHERE {predicate}"""
            )
        return count

    # ---- yazma -----------------------------------------------------------------
    def write_fund_master(self, df: pd.DataFrame) -> int:
        """Aynı snapshot_date için önce siler, sonra ekler (idempotent)."""
        self._ensure_writable("write_fund_master")
        if df.empty:
            return 0
        self.con.register("df_", df)
        self.con.execute("DELETE FROM fund_master WHERE snapshot_date IN (SELECT DISTINCT snapshot_date FROM df_)")
        self.con.execute("INSERT INTO fund_master BY NAME SELECT * FROM df_")
        self.con.unregister("df_")
        return len(df)

    def write_fund_master_rows(self, df: pd.DataFrame) -> int:
        """Alt küme koşusu: aynı snapshot_date içinde yalnızca verilen fonların satırlarını değiştirir."""
        self._ensure_writable("write_fund_master_rows")
        if df.empty:
            return 0
        self.con.register("df_", df)
        self.con.execute(
            "DELETE FROM fund_master USING df_ WHERE fund_master.snapshot_date = df_.snapshot_date AND fund_master.fund_code = df_.fund_code"
        )
        self.con.execute("INSERT INTO fund_master BY NAME SELECT * FROM df_")
        self.con.unregister("df_")
        return len(df)

    def upsert_nav(self, df: pd.DataFrame) -> int:
        """(fund_code, date) anahtarında upsert; published_at ile as-of izi tutulur."""
        self._ensure_writable("upsert_nav")
        if df.empty:
            return 0
        df = df[["fund_code", "date", "price", "published_at"]].drop_duplicates(["fund_code", "date"], keep="last")
        self.con.register("df_", df)
        self.con.execute(
            "DELETE FROM fund_nav USING df_ WHERE fund_nav.fund_code = df_.fund_code AND fund_nav.date = df_.date"
        )
        self.con.execute("INSERT INTO fund_nav BY NAME SELECT * FROM df_")
        self.con.unregister("df_")
        return len(df)

    def upsert_macro(self, df: pd.DataFrame) -> int:
        """(series, date) anahtarında upsert."""
        self._ensure_writable("upsert_macro")
        if df.empty:
            return 0
        df = df[["series", "date", "value", "available_from", "published_at"]].drop_duplicates(
            ["series", "date"], keep="last"
        )
        self.con.register("df_", df)
        self.con.execute("DELETE FROM macro USING df_ WHERE macro.series = df_.series AND macro.date = df_.date")
        self.con.execute("INSERT INTO macro BY NAME SELECT * FROM df_")
        self.con.unregister("df_")
        return len(df)

    def macro_wide(self, asof: str | None = None) -> pd.DataFrame:
        """Seriler sütun; `asof` verilirse yalnızca o tarihte KULLANILABİLİR (available_from ≤ asof) gözlemler."""
        q = "SELECT series, date, value FROM macro"
        if asof:
            q += f" WHERE available_from <= DATE '{asof}'"
        long = self.con.execute(q).df()
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot(index="date", columns="series", values="value").sort_index()
        wide.index = pd.to_datetime(wide.index)
        return wide

    def log_run(self, run_id: str, kind: str, started_at: datetime, status: str, summary: dict) -> None:
        self._ensure_writable("log_run")
        self.con.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
            [run_id, kind, started_at, datetime.now(), status, json.dumps(summary, ensure_ascii=False, default=str)],
        )

    # ---- okuma -----------------------------------------------------------------
    def latest_fund_master(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT * FROM fund_master WHERE snapshot_date = (SELECT max(snapshot_date) FROM fund_master)"
        ).df()

    def fund_master_history(self) -> pd.DataFrame:
        """Return dated fund metadata snapshots; callers must apply their own point-in-time cutoff."""
        return self.con.execute("SELECT * FROM fund_master ORDER BY snapshot_date, fund_code").df()

    def nav_long(self, start: str | None = None) -> pd.DataFrame:
        q = "SELECT fund_code, date, price FROM fund_nav"
        if start:
            q += f" WHERE date >= DATE '{start}'"
        return self.con.execute(q + " ORDER BY date, fund_code").df()

    def nav_wide(self, start: str | None = None) -> pd.DataFrame:
        long = self.nav_long(start)
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot(index="date", columns="fund_code", values="price").sort_index()
        wide.index = pd.to_datetime(wide.index)
        return wide

    def nav_stats(self, codes: list[str] | None = None) -> pd.DataFrame:
        """fund_code başına first_nav_date, last_nav_date, n_nav (depodaki toplam geçmiş)."""
        q = "SELECT fund_code, min(date) AS first_nav_date, max(date) AS last_nav_date, count(*) AS n_nav FROM fund_nav"
        if codes:
            self.con.register("codes_", pd.DataFrame({"fund_code": list(codes)}))
            q += " WHERE fund_code IN (SELECT fund_code FROM codes_)"
        df = self.con.execute(q + " GROUP BY fund_code").df()
        if codes:
            self.con.unregister("codes_")
        return df

    def last_nav_dates(self) -> pd.Series:
        df = self.con.execute("SELECT fund_code, max(date) AS last_date FROM fund_nav GROUP BY fund_code").df()
        return df.set_index("fund_code")["last_date"]

    def last_run(self, kind: str | None = None) -> dict | None:
        q = "SELECT run_id, kind, started_at, finished_at, status, summary FROM runs"
        if kind:
            q += f" WHERE kind = '{kind}'"
        row = self.con.execute(q + " ORDER BY finished_at DESC LIMIT 1").fetchone()
        if not row:
            return None
        d = dict(zip(["run_id", "kind", "started_at", "finished_at", "status", "summary"], row, strict=True))
        d["summary"] = json.loads(d["summary"]) if d["summary"] else {}
        return d

    # ---- arşiv -----------------------------------------------------------------
    def export_parquet(self, out_dir: str | Path) -> list[Path]:
        self._ensure_writable("export_parquet")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Stamp the source before copying so DuckDB and the Parquet generation carry identical runs rows.
        iso = pd.Timestamp.now(tz="UTC").isoformat()
        for run_id, summary in self.con.execute("SELECT run_id, summary FROM runs").fetchall():
            d = json.loads(summary) if summary else {}
            d["snapshot_asof"] = iso
            self.con.execute(
                "UPDATE runs SET summary = ? WHERE run_id = ?",
                [json.dumps(d, ensure_ascii=False, default=str), run_id],
            )
        paths = []
        for t in _DDL:
            p = out / f"{t}.parquet"
            self.con.execute(f"COPY (SELECT * FROM {t}) TO '{p}' (FORMAT PARQUET)")
            paths.append(p)
        return paths

    def export_parquet_atomic(self, out_dir: str | Path) -> list[Path]:
        """Export/validate a generation, then atomically switch a small current-generation pointer.

        Immutable generations are retained. Readers resolve the pointer once, so an export never
        exposes a missing or mixed four-Parquet snapshot while it is being published.
        """
        self._ensure_writable("export_parquet_atomic")
        out = Path(out_dir)
        out.parent.mkdir(parents=True, exist_ok=True)
        generations = out.parent / f".{out.name}.generations"
        generations.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="stage-", dir=generations))
        pointer = out.with_name(f".{out.name}.current")
        pointer_tmp: Path | None = None
        published = False
        old_pointer = pointer.read_text(encoding="utf-8") if pointer.is_file() else None
        transaction_open = False
        try:
            self.con.execute("BEGIN TRANSACTION")
            transaction_open = True
            paths = self.export_parquet(stage)
            expected = {f"{table}.parquet" for table in _TABLES}
            if {path.name for path in paths} != expected or not parquet_snapshot_complete(stage):
                raise RuntimeError("Parquet stage eksik tablo içeriyor")
            snapshot = Store.from_parquet(stage)
            try:
                for table in _TABLES:
                    columns = [row[1] for row in self.con.execute(f"PRAGMA table_info('{table}')").fetchall()]
                    columns_sql = ", ".join(f'"{column}"' for column in columns)
                    query = f"SELECT count(*), bit_xor(hash({columns_sql})) FROM {table}"
                    source_fingerprint = self.con.execute(query).fetchone()
                    snapshot_fingerprint = snapshot.con.execute(query).fetchone()
                    if source_fingerprint != snapshot_fingerprint:
                        raise RuntimeError(f"Parquet stage içeriği DuckDB ile uyuşmuyor: {table}")
            finally:
                snapshot.close()
            relative_generation = os.path.relpath(stage, out.parent)
            pointer_tmp = pointer.with_name(f"{pointer.name}.{uuid.uuid4().hex}.tmp")
            pointer_tmp.write_text(relative_generation, encoding="utf-8")
            os.replace(pointer_tmp, pointer)
            published = True
            self.con.execute("COMMIT")
            transaction_open = False
            return [stage / f"{table}.parquet" for table in _TABLES]
        except Exception:
            if transaction_open:
                self.con.execute("ROLLBACK")
                transaction_open = False
            if published:
                if old_pointer is None:
                    pointer.unlink(missing_ok=True)
                else:
                    restore_tmp = pointer.with_name(f"{pointer.name}.{uuid.uuid4().hex}.restore")
                    restore_tmp.write_text(old_pointer, encoding="utf-8")
                    os.replace(restore_tmp, pointer)
                published = False
            raise
        finally:
            if pointer_tmp is not None and pointer_tmp.exists():
                pointer_tmp.unlink()
            if stage.exists() and not published:
                shutil.rmtree(stage)

    def close(self) -> None:
        self.con.close()
