"""Atomically archive a paper-trading pilot before starting a fresh epoch."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime

MANIFEST_TABLE = "janus_paper_archive_manifest"
EPOCH_TABLE = "janus_pilot_epoch"


def _table_exists(con, name: str) -> bool:
    return bool(
        con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='main' AND table_name=?",
            [name],
        ).fetchone()[0]
    )


def archive_paper_v1(store, archived_at: datetime | None = None, ensure_schema: Callable | None = None) -> list[str]:
    """Archive literal ``paper_`` tables and initialize their new schema atomically.

    A manifest written in the same transaction is the sole evidence that a prior
    archive completed. Unmanifested ``_v1`` tables are treated as an incomplete
    archive and fail closed.
    """
    con = store.con
    names = [
        str(row[0])
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' AND table_type='BASE TABLE' ORDER BY table_name"
        ).fetchall()
    ]
    live = [name for name in names if name.startswith("paper_") and not name.endswith("_v1")]
    versioned = [name for name in names if name.startswith("paper_") and name.endswith("_v1")]

    # An empty schema may be retried only when the completed-archive manifest
    # proves the archived set, epoch, and all archived tables are still present.
    if _table_exists(con, MANIFEST_TABLE):
        if not _table_exists(con, EPOCH_TABLE):
            raise ValueError("Pilot arşiv manifesti var ancak epoch yok; hiçbir tablo değiştirilmedi")
        rows = con.execute(f'SELECT archived_tables FROM "{MANIFEST_TABLE}"').fetchall()
        archived_names = json.loads(rows[0][0]) if len(rows) == 1 else []
        if (
            not archived_names
            or any(name not in versioned for name in archived_names)
            or set(versioned) != set(archived_names)
            or any(int(con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]) != 0 for name in live)
        ):
            raise ValueError("Önceki arşiv doğrulanamadı; hiçbir tablo değiştirilmedi")
        if ensure_schema is not None:
            con.execute("BEGIN TRANSACTION")
            try:
                ensure_schema(store)
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        return []

    if versioned:
        raise ValueError("Manifest'siz/yarım _v1 arşivi var; hiçbir tablo değiştirilmedi")
    if not live:
        raise ValueError("Arşivlenecek paper_ tabloları yok; hiçbir tablo değiştirilmedi")

    targets = {name: f"{name}_v1" for name in live}
    collisions = [target for target in targets.values() if _table_exists(con, target)]
    if collisions:
        raise ValueError(f"Arşiv hedefi zaten var: {collisions[0]}; hiçbir tablo taşınmadı")

    stamp = archived_at or datetime.now()
    con.execute("BEGIN TRANSACTION")
    try:
        for source, target in targets.items():
            con.execute(f'ALTER TABLE "{source}" RENAME TO "{target}"')
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {EPOCH_TABLE} "
            "(epoch_id INTEGER PRIMARY KEY, started_at TIMESTAMP, initialized_at TIMESTAMP, "
            "published_at TIMESTAMP, available_from TIMESTAMP)"
        )
        con.execute(f"DELETE FROM {EPOCH_TABLE}")
        con.execute(f"INSERT INTO {EPOCH_TABLE} VALUES (1, ?, NULL, ?, ?)", [stamp, stamp, stamp])
        con.execute(
            f"CREATE TABLE {MANIFEST_TABLE} "
            "(archived_tables VARCHAR, archived_at TIMESTAMP, published_at TIMESTAMP, available_from TIMESTAMP)"
        )
        con.execute(
            f"INSERT INTO {MANIFEST_TABLE} VALUES (?, ?, ?, ?)",
            [json.dumps(sorted(targets.values())), stamp, stamp, stamp],
        )
        if ensure_schema is not None:
            ensure_schema(store)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return list(targets.values())
