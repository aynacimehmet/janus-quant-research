"""S5-4: operasyon KPI hesaplamaları."""

from __future__ import annotations

import json
import re
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from janus.data.quality import business_days


def _kpi_thresholds(cfg: dict) -> dict:
    kpi = cfg.get("paper", {}).get("kpi", {})
    return {
        "nightly_success_min": float(kpi.get("nightly_success_min", 0.95)),
        "morning_deadline": kpi.get("morning_deadline", "09:30"),
        "reconcile_max_pp": float(kpi.get("reconcile_max_pp", 0.5)),
        "identity_tol": float(kpi.get("identity_tol", 1e-6)),
    }


def _parse_time(t: str) -> time:
    h, m = t.split(":")
    return time(int(h), int(m))


def _load_runs(store, kind: str) -> pd.DataFrame:
    df = store.con.execute(
        "SELECT run_id, kind, started_at, finished_at, status, summary FROM runs WHERE kind = ? ORDER BY finished_at",
        [kind],
    ).df()
    if df.empty:
        return df
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col])
    return df


def _window(asof: str | pd.Timestamp | None, lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.now().normalize()
    return end - pd.Timedelta(days=lookback_days - 1), end


def _pilot_window(store, asof: str | pd.Timestamp | None, lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = _window(asof, lookback_days)
    exists = store.con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='main' AND table_name='janus_pilot_epoch'"
    ).fetchone()[0]
    if exists:
        epoch = store.con.execute("SELECT min(started_at) FROM janus_pilot_epoch").fetchone()[0]
        if epoch is not None:
            start = max(start, pd.Timestamp(epoch).normalize())
    return start, end


def _epoch_started_at(store) -> pd.Timestamp | None:
    exists = store.con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='main' AND table_name='janus_pilot_epoch'"
    ).fetchone()[0]
    if not exists:
        return None
    value = store.con.execute("SELECT min(started_at) FROM janus_pilot_epoch").fetchone()[0]
    return pd.Timestamp(value) if value is not None else None


def _after_epoch(df: pd.DataFrame, store) -> pd.DataFrame:
    epoch = _epoch_started_at(store)
    return df if epoch is None or df.empty else df.loc[df["started_at"] >= epoch]


def _nightly_kpis(store, asof: str | pd.Timestamp | None = None, lookback_days: int = 28) -> dict:
    """Günlük schedule paydasına göre başarıyı hesapla; eksik nightly günleri başarı sayılmaz."""
    df = _load_runs(store, "nightly")
    df = _after_epoch(df, store)
    start, end = _pilot_window(store, asof, lookback_days)
    expected = pd.date_range(start, end, freq="D")
    if not df.empty:
        df = df.assign(run_date=df["started_at"].dt.normalize())
        recent = (
            df.loc[df["run_date"].between(start, end)]
            .sort_values("finished_at")
            .drop_duplicates("run_date", keep="last")
        )
    else:
        recent = df
    if recent.empty:
        return {
            "nightly_success_rate": None,
            "avg_duration_min": None,
            "expected_runs": len(expected),
            "observed_runs": 0,
            "missing_runs": len(expected),
            "data_available": False,
        }
    successful_dates = recent.loc[recent["status"] == "ok", "run_date"].nunique()
    duration = (recent["finished_at"] - recent["started_at"]).dt.total_seconds() / 60.0
    return {
        "nightly_success_rate": float(successful_dates / len(expected)) if len(expected) else None,
        "avg_duration_min": float(duration.mean()),
        "expected_runs": len(expected),
        "observed_runs": int(recent["run_date"].nunique()),
        "missing_runs": int(len(expected) - recent["run_date"].nunique()),
        "data_available": True,
    }


def _morning_kpi(store, cfg: dict, asof: str | pd.Timestamp | None = None, lookback_days: int = 28) -> dict:
    """Gönderim KPI'ı yalnız morning summary'deki açık onay ve sent_at'ı kullanır."""
    start, end = _pilot_window(store, asof, lookback_days)
    expected_dates = business_days(start, end, cfg)
    df = _load_runs(store, "morning")
    df = _after_epoch(df, store)
    if not df.empty:
        df = df.assign(run_date=df["started_at"].dt.normalize())
        df = (
            df.loc[df["run_date"].isin(expected_dates)]
            .sort_values("finished_at")
            .drop_duplicates("run_date", keep="last")
        )
    if df.empty:
        return {
            "morning_sent_at": None,
            "morning_on_time": None,
            "morning_sent_ok": None,
            "morning_expected_runs": len(expected_dates),
            "morning_observed_runs": 0,
            "morning_missing_runs": len(expected_dates),
            "morning_on_time_rate": None,
        }
    last = df.iloc[-1]
    summary = last["summary"]
    if isinstance(summary, str):
        summary = json.loads(summary)
    sent_ok = summary.get("sent_ok")
    sent_raw = summary.get("sent_at")
    sent = pd.Timestamp(sent_raw) if sent_raw else None
    deadline = _parse_time(_kpi_thresholds(cfg)["morning_deadline"])
    on_time = bool(sent_ok and sent is not None and sent.time() <= deadline) if sent_ok is not None else None
    if sent_ok is False:
        on_time = False
    on_time_count = 0
    for payload in df["summary"]:
        if isinstance(payload, str):
            payload = json.loads(payload)
        sent_time = pd.Timestamp(payload["sent_at"]) if payload.get("sent_at") else None
        if payload.get("sent_ok") is True and sent_time is not None and sent_time.time() <= deadline:
            on_time_count += 1
    observed = int(df["run_date"].nunique())
    return {
        "morning_sent_at": sent.strftime("%H:%M") if sent is not None else None,
        "morning_on_time": on_time,
        "morning_sent_ok": bool(sent_ok) if sent_ok is not None else None,
        "morning_expected_runs": len(expected_dates),
        "morning_observed_runs": observed,
        "morning_missing_runs": int(len(expected_dates) - observed),
        "morning_on_time_rate": float(on_time_count / len(expected_dates)) if len(expected_dates) else None,
    }


def _proposal_kpis(
    store, asof: str | pd.Timestamp | None = None, lookback_days: int = 28, cfg: dict | None = None
) -> dict:
    """Proposal kapsamı yalnız configured iş günleri için ölçülür; hafta sonu kayıtları paydada yoktur."""
    start, end = _pilot_window(store, asof, lookback_days)
    expected_dates = business_days(start, end, cfg or {})
    epoch = _epoch_started_at(store)
    proposals = store.con.execute(
        "SELECT date FROM paper_proposals WHERE date BETWEEN ? AND ? AND (? IS NULL OR created_at >= ?)",
        [
            start.date(),
            end.date(),
            epoch.to_pydatetime() if epoch is not None else None,
            epoch.to_pydatetime() if epoch is not None else None,
        ],
    ).df()
    if proposals.empty:
        observed = 0
    else:
        observed_dates = pd.to_datetime(proposals["date"]).dt.normalize().drop_duplicates()
        observed = int(observed_dates.isin(expected_dates).sum())
    return {
        "expected_proposals": len(expected_dates),
        "observed_proposals": observed,
        "missing_proposals": int(len(expected_dates) - observed),
        "data_available": observed > 0,
    }


def _source_stale_alarms(store, asof: str | pd.Timestamp | None = None, lookback_days: int = 28) -> int | None:
    df = _load_runs(store, "nightly")
    df = _after_epoch(df, store)
    if df.empty:
        return None
    start, end = _pilot_window(store, asof, lookback_days)
    recent = df[df["started_at"].dt.normalize().between(start, end)]
    if recent.empty:
        return None
    n = 0
    for summary in recent["summary"]:
        if isinstance(summary, str):
            summary = json.loads(summary)
        q = summary.get("quality", {})
        if q.get("source_stale") or q.get("data_stale"):
            n += 1
    return n


def _proposal_to_fill_days(store) -> float | None:
    proposals = store.con.execute("SELECT proposal_id, date FROM paper_proposals WHERE status = 'filled'").df()
    if proposals.empty:
        return None
    proposals["date"] = pd.to_datetime(proposals["date"])
    fills = store.con.execute(
        "SELECT proposal_id, min(fill_date) AS fill_date FROM paper_fills GROUP BY proposal_id"
    ).df()
    if fills.empty:
        return None
    fills["fill_date"] = pd.to_datetime(fills["fill_date"])
    merged = proposals.merge(fills, on="proposal_id", how="inner")
    if merged.empty:
        return None
    diff = (merged["fill_date"] - merged["date"]).dt.total_seconds() / 86400.0
    return float(diff.median())


def _reconcile_kpi(store, cfg: dict) -> dict:
    row = store.con.execute(
        "SELECT identity_ok, max_weight_diff FROM paper_reconcile ORDER BY date DESC LIMIT 1"
    ).fetchone()
    if not row:
        return {"identity_ok": None, "max_diff_pp": None}
    identity_ok = bool(row[0]) if row[0] is not None else None
    max_diff = float(row[1]) if row[1] is not None else None
    return {"identity_ok": identity_ok, "max_diff_pp": max_diff * 100 if max_diff is not None else None}


def compute_kpis(
    store,
    cfg: dict,
    asof: str | pd.Timestamp | None = None,
    decision_asof: str | pd.Timestamp | None = None,
) -> dict:
    """Tüm KPI'ları hesapla; veri yoksa None / 'veri yok'."""
    thresholds = _kpi_thresholds(cfg)
    nightly = _nightly_kpis(store, asof=asof)
    morning = _morning_kpi(store, cfg, asof=asof)
    proposals = _proposal_kpis(store, asof=decision_asof if decision_asof is not None else asof, cfg=cfg)
    stale_alarms = _source_stale_alarms(store, asof=asof)
    fill_delay = _proposal_to_fill_days(store)
    rec = _reconcile_kpi(store, cfg)

    nightly_ok = (
        None if not nightly["data_available"] else nightly["nightly_success_rate"] >= thresholds["nightly_success_min"]
    )
    morning_ok = morning["morning_on_time"]
    reconcile_ok = (
        None
        if rec["identity_ok"] is None or rec["max_diff_pp"] is None
        else rec["identity_ok"] is True and rec["max_diff_pp"] <= thresholds["reconcile_max_pp"]
    )

    checks = [nightly_ok, morning_ok, reconcile_ok]
    epoch = _epoch_started_at(store)
    date_only_asof = isinstance(asof, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", asof) is not None
    end = pd.Timestamp.now() if asof is None else pd.Timestamp(asof)
    full_pilot = epoch is not None and not date_only_asof and end >= epoch + pd.Timedelta(days=28)
    gate_met = all(checks) if full_pilot and all(c is not None for c in checks) else None

    return {
        "thresholds": thresholds,
        "nightly_success_rate": nightly["nightly_success_rate"],
        "nightly_success_ok": nightly_ok,
        "nightly_expected_runs": nightly["expected_runs"],
        "nightly_observed_runs": nightly["observed_runs"],
        "nightly_missing_runs": nightly["missing_runs"],
        "avg_nightly_duration_min": nightly["avg_duration_min"],
        "morning_sent_at": morning["morning_sent_at"],
        "morning_sent_ok": morning["morning_sent_ok"],
        "morning_on_time": morning["morning_on_time"],
        "morning_expected_runs": morning["morning_expected_runs"],
        "morning_observed_runs": morning["morning_observed_runs"],
        "morning_missing_runs": morning["morning_missing_runs"],
        "morning_on_time_rate": morning["morning_on_time_rate"],
        "proposal_expected_runs": proposals["expected_proposals"],
        "proposal_observed_runs": proposals["observed_proposals"],
        "proposal_missing_runs": proposals["missing_proposals"],
        "source_stale_alarms": stale_alarms,
        "proposal_to_fill_days": fill_delay,
        "reconcile_identity_ok": rec["identity_ok"],
        "reconcile_max_diff_pp": rec["max_diff_pp"],
        "reconcile_ok": reconcile_ok,
        "gate_met": gate_met,
    }


def paper_kpi(
    store,
    cfg: dict,
    date: str | None = None,
    out_dir: Path | None = None,
    asof: str | pd.Timestamp | None = None,
    decision_asof: str | pd.Timestamp | None = None,
) -> dict:
    d = pd.Timestamp(date).normalize() if date else pd.Timestamp.now().normalize()
    kpi_asof = asof
    if kpi_asof is None:
        if date is None:
            kpi_asof = datetime.now(ZoneInfo(cfg.get("project", {}).get("timezone", "UTC"))).replace(tzinfo=None)
        else:
            timezone = ZoneInfo(cfg.get("project", {}).get("timezone", "UTC"))
            trusted_now = datetime.now(timezone)
            if d.date() == trusted_now.date():
                # DuckDB pilot timestamps are local-naive; preserve the trusted local wall-clock time.
                kpi_asof = trusted_now.replace(tzinfo=None)
            else:
                # A date-only historical/future report date cannot certify elapsed pilot time.
                kpi_asof = date
    kpis = compute_kpis(
        store,
        cfg,
        asof=kpi_asof if kpi_asof is not None else d,
        decision_asof=decision_asof if decision_asof is not None else d,
    )
    th = kpis["thresholds"]

    def status_line(ok: bool | None, value, threshold_str: str, unit: str = ""):
        if ok is None:
            return f"veri yok | eşik: {threshold_str}"
        sym = "✓" if ok else "✗"
        val = f"{value:.4f}" if isinstance(value, float) else str(value)
        return f"{sym} {val}{unit} | eşik: {threshold_str}"

    lines = [
        f"# Operasyon KPI — {d.date()}",
        "",
        "| KPI | Durum |",
        "|-----|-------|",
        f"| Gece beklenen / gözlenen / eksik | {kpis['nightly_expected_runs']} / "
        f"{kpis['nightly_observed_runs']} / {kpis['nightly_missing_runs']} |",
        f"| Gece koşusu başarı oranı | {status_line(kpis['nightly_success_ok'], kpis['nightly_success_rate'], f'≥ {th["nightly_success_min"] * 100:.0f}%')} |",
        f"| Ort. gece süresi | {kpis['avg_nightly_duration_min']:.1f} dk"
        if kpis["avg_nightly_duration_min"] is not None
        else "| Ort. gece süresi | veri yok |",
        f"| Sabah mesajı saati | {status_line(kpis['morning_on_time'], kpis['morning_sent_at'], f'≤ {th["morning_deadline"]}')} |",
        f"| Sabah beklenen / gözlenen / eksik | {kpis['morning_expected_runs']} / "
        f"{kpis['morning_observed_runs']} / {kpis['morning_missing_runs']} |",
        "| Sabah zamanında oran | "
        + (f"{kpis['morning_on_time_rate']:.1%}" if kpis["morning_on_time_rate"] is not None else "veri yok")
        + " |",
        f"| Öneri beklenen / gözlenen / eksik (iş günü) | {kpis['proposal_expected_runs']} / "
        f"{kpis['proposal_observed_runs']} / {kpis['proposal_missing_runs']} |",
        f"| source_stale alarm (28g) | {kpis['source_stale_alarms'] if kpis['source_stale_alarms'] is not None else 'veri yok'} |",
        f"| Öneri→fill gecikmesi | {kpis['proposal_to_fill_days']:.2f} gün"
        if kpis["proposal_to_fill_days"] is not None
        else "| Öneri→fill gecikmesi | veri yok |",
        f"| Mutabakat özdeşliği | {'✓' if kpis['reconcile_identity_ok'] else ('✗' if kpis['reconcile_identity_ok'] is False else 'veri yok')} |",
        f"| Mutabakat max farkı | {status_line(kpis['reconcile_ok'], kpis['reconcile_max_diff_pp'], f'≤ {th["reconcile_max_pp"]} pp', ' pp')} |",
        "",
        f"**4 haftalık kapı:** {'sağlandı' if kpis['gate_met'] else ('sağlanmadı' if kpis['gate_met'] is False else 'veri yok')}",
    ]

    out = Path(out_dir) if out_dir else Path(cfg.get("reporting", {}).get("orders_dir", "reports"))
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"paper_kpi_{d:%Y-%m-%d}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"date": str(d.date()), "kpis": kpis, "md_path": str(md_path)}
