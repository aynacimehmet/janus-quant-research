from __future__ import annotations

import json

import pandas as pd

from janus.data.store import Store
from janus.paper.core import _ensure_paper_tables
from janus.paper.kpi import _morning_kpi, _nightly_kpis, _proposal_kpis, compute_kpis, paper_kpi


def _put_run(store: Store, run_id: str, kind: str, finished: str, status: str, summary: dict) -> None:
    finished_at = pd.Timestamp(finished)
    store.con.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
        [run_id, kind, finished_at - pd.Timedelta(minutes=2), finished_at, status, json.dumps(summary)],
    )


def _put_proposal(store: Store, proposal_id: str, date: str, status: str) -> None:
    decision_date = pd.Timestamp(date)
    store.con.execute(
        """INSERT INTO paper_proposals
        (proposal_id, date, created_at, expires_at, status, orders_json, evidence_count, evidence_codes,
         target_weights_json) VALUES (?, ?, ?, ?, ?, '[]', 0, '[]', '{}')""",
        [proposal_id, decision_date.date(), decision_date, decision_date, status],
    )


def test_nightly_kpi_uses_expected_run_denominator_and_missing_dates(tmp_path):
    store = Store(tmp_path / "kpi.duckdb")
    _ensure_paper_tables(store)
    days = pd.date_range("2026-09-01", "2026-09-28", freq="D")
    for i, day in enumerate(days[:-1]):
        _put_run(store, f"nightly-{i}", "nightly", f"{day.date()} 23:30", "ok", {})

    result = _nightly_kpis(store, asof="2026-09-28", lookback_days=28)

    assert result["expected_runs"] == 28
    assert result["observed_runs"] == 27
    assert result["missing_runs"] == 1
    assert result["nightly_success_rate"] == 27 / 28
    store.close()


def test_nightly_kpi_reports_no_data_separately(tmp_path):
    store = Store(tmp_path / "empty_kpi.duckdb")
    _ensure_paper_tables(store)

    result = _nightly_kpis(store, asof="2026-09-28", lookback_days=28)

    assert result["nightly_success_rate"] is None
    assert result["data_available"] is False
    assert result["missing_runs"] == 28
    store.close()


def test_nightly_kpi_epoch_excludes_runs_before_reset_and_starts_at_epoch(tmp_path):
    store = Store(tmp_path / "pilot_epoch.duckdb")
    _ensure_paper_tables(store)
    store.con.execute(
        "CREATE TABLE janus_pilot_epoch "
        "(epoch_id INTEGER PRIMARY KEY, started_at TIMESTAMP, initialized_at TIMESTAMP, "
        "published_at TIMESTAMP, available_from TIMESTAMP)"
    )
    store.con.execute(
        "INSERT INTO janus_pilot_epoch VALUES (1, '2026-09-25', '2026-09-25', '2026-09-25', '2026-09-25')"
    )
    _put_run(store, "old-nightly", "nightly", "2026-09-24 23:30", "ok", {})
    _put_run(store, "epoch-nightly", "nightly", "2026-09-25 23:30", "ok", {})

    result = _nightly_kpis(store, asof="2026-09-28", lookback_days=28)

    assert result["expected_runs"] == 4
    assert result["observed_runs"] == 1
    assert result["missing_runs"] == 3
    store.close()


def test_kpi_epoch_uses_exact_timestamp_and_gate_waits_full_28_calendar_days(tmp_path):
    store = Store(tmp_path / "precise_epoch.duckdb")
    _ensure_paper_tables(store)
    store.con.execute(
        "CREATE TABLE janus_pilot_epoch "
        "(epoch_id INTEGER PRIMARY KEY, started_at TIMESTAMP, initialized_at TIMESTAMP, "
        "published_at TIMESTAMP, available_from TIMESTAMP)"
    )
    store.con.execute(
        "INSERT INTO janus_pilot_epoch VALUES "
        "(1, '2026-09-01 12:00:00', '2026-09-01 12:00:00', '2026-09-01 12:00:00', '2026-09-01 12:00:00')"
    )
    # Same calendar day but before the reset timestamp must not count.
    _put_run(store, "same-day-before", "nightly", "2026-09-01 11:59", "ok", {})
    _put_run(store, "same-day-after", "nightly", "2026-09-01 12:30", "ok", {})
    for day in pd.date_range("2026-09-02", "2026-09-28", freq="D"):
        _put_run(store, f"nightly-{day:%Y%m%d}", "nightly", f"{day.date()} 23:30", "ok", {})
    _put_run(
        store,
        "morning-epoch",
        "morning",
        "2026-09-28 09:10",
        "ok",
        {"sent_ok": True, "sent_at": "2026-09-28T09:00:00"},
    )
    store.con.execute("INSERT INTO paper_reconcile VALUES ('2026-09-28', 100, 100, 0, 0, 0, TRUE, 100, 0, 0)")

    cfg = {"paper": {"kpi": {"nightly_success_min": 0.0}}}
    early = compute_kpis(store, cfg, asof="2026-09-28")
    assert early["nightly_observed_runs"] == 28
    assert early["gate_met"] is None

    before_boundary = compute_kpis(store, cfg, asof=pd.Timestamp("2026-09-29 11:59:00"))
    assert before_boundary["gate_met"] is None

    # Date-only asof has no known time of day, so it cannot prove pilot completion.
    date_only = compute_kpis(store, cfg, asof="2026-09-29")
    assert date_only["gate_met"] is None

    completed = compute_kpis(store, cfg, asof=pd.Timestamp("2026-09-29 12:00:00"))
    assert completed["gate_met"] is True
    store.close()


def test_paper_kpi_date_only_does_not_promote_past_or_future_report_date_to_clock(tmp_path):
    store = Store(tmp_path / "paper_date_gate.duckdb")
    _ensure_paper_tables(store)
    store.con.execute(
        "CREATE TABLE janus_pilot_epoch "
        "(epoch_id INTEGER PRIMARY KEY, started_at TIMESTAMP, initialized_at TIMESTAMP, "
        "published_at TIMESTAMP, available_from TIMESTAMP)"
    )
    store.con.execute(
        "INSERT INTO janus_pilot_epoch VALUES "
        "(1, '2026-09-01 12:00:00', '2026-09-01 12:00:00', '2026-09-01 12:00:00', '2026-09-01 12:00:00')"
    )

    past = paper_kpi(store, {}, date="2026-09-29", out_dir=tmp_path / "past")
    future = paper_kpi(store, {}, date="2026-09-30", out_dir=tmp_path / "future")

    assert past["kpis"]["gate_met"] is None
    assert future["kpis"]["gate_met"] is None
    store.close()


def test_paper_kpi_historical_date_keeps_date_only_asof_with_fake_local_clock(tmp_path, monkeypatch):
    import janus.paper.kpi as kpi_module

    store = Store(tmp_path / "paper_fake_clock.duckdb")
    _ensure_paper_tables(store)
    observed = {}
    original = kpi_module.compute_kpis

    def capture_asof(_store, cfg, asof=None, decision_asof=None):
        observed["asof"] = asof
        return original(_store, cfg, asof=asof, decision_asof=decision_asof)

    class FakeDateTime:
        @staticmethod
        def now(tz=None):
            return pd.Timestamp("2026-09-30 12:00:00", tz=tz).to_pydatetime()

    monkeypatch.setattr(kpi_module, "datetime", FakeDateTime)
    monkeypatch.setattr(kpi_module, "compute_kpis", capture_asof)

    paper_kpi(store, {"project": {"timezone": "Europe/Istanbul"}}, date="2026-09-29", out_dir=tmp_path)

    assert observed["asof"] == "2026-09-29"
    assert isinstance(observed["asof"], str)
    store.close()


def test_paper_kpi_explicit_datetime_asof_counts_only_post_epoch_pilot_runs(tmp_path):
    store = Store(tmp_path / "paper_datetime_gate.duckdb")
    _ensure_paper_tables(store)
    store.con.execute(
        "CREATE TABLE janus_pilot_epoch "
        "(epoch_id INTEGER PRIMARY KEY, started_at TIMESTAMP, initialized_at TIMESTAMP, "
        "published_at TIMESTAMP, available_from TIMESTAMP)"
    )
    store.con.execute(
        "INSERT INTO janus_pilot_epoch VALUES "
        "(1, '2026-09-01 12:00:00', '2026-09-01 12:00:00', '2026-09-01 12:00:00', '2026-09-01 12:00:00')"
    )
    _put_run(store, "pre-epoch", "nightly", "2026-09-01 11:00", "ok", {})
    _put_run(store, "post-epoch", "nightly", "2026-09-28 13:00", "ok", {})

    result = paper_kpi(
        store,
        {},
        date="2026-09-29",
        asof=pd.Timestamp("2026-09-29 12:00:00"),
        out_dir=tmp_path / "explicit",
    )

    assert result["kpis"]["gate_met"] is None  # Required KPI observations are absent.
    assert result["kpis"]["nightly_observed_runs"] == 1
    store.close()


def test_kpi_gate_is_none_without_pilot_epoch(tmp_path):
    store = Store(tmp_path / "no_epoch.duckdb")
    _ensure_paper_tables(store)
    result = compute_kpis(store, {}, asof="2026-09-28")
    assert result["gate_met"] is None
    store.close()


def test_four_week_gate_is_no_data_when_required_observations_are_missing(tmp_path):
    store = Store(tmp_path / "no_data_gate.duckdb")
    _ensure_paper_tables(store)

    result = compute_kpis(store, {}, asof="2026-09-28")

    assert result["nightly_success_ok"] is None
    assert result["morning_on_time"] is None
    assert result["reconcile_ok"] is None
    assert result["source_stale_alarms"] is None
    assert result["gate_met"] is None
    store.close()


def test_morning_kpi_requires_successful_send_ack_and_uses_sent_at(tmp_path):
    store = Store(tmp_path / "morning_kpi.duckdb")
    _ensure_paper_tables(store)
    _put_run(
        store,
        "morning-failed-send",
        "morning",
        "2026-09-25 09:10",
        "failed",
        {"sent_ok": False, "sent_at": "2026-09-25T08:55:00"},
    )

    result = _morning_kpi(store, {"paper": {"kpi": {"morning_deadline": "09:30"}}}, asof="2026-09-25")

    assert result["morning_sent_at"] == "08:55"
    assert result["morning_on_time"] is False
    store.close()


def test_morning_kpi_does_not_use_run_finish_time_as_send_time(tmp_path):
    store = Store(tmp_path / "late_send.duckdb")
    _ensure_paper_tables(store)
    _put_run(
        store,
        "morning-late-send",
        "morning",
        "2026-09-25 09:10",
        "ok",
        {"sent_ok": True, "sent_at": "2026-09-25T09:31:00"},
    )

    result = _morning_kpi(store, {"paper": {"kpi": {"morning_deadline": "09:30"}}}, asof="2026-09-25")

    assert result["morning_sent_at"] == "09:31"
    assert result["morning_on_time"] is False
    store.close()


def test_weekend_morning_no_new_data_is_not_a_missing_business_run(tmp_path):
    store = Store(tmp_path / "weekend_morning.duckdb")
    _ensure_paper_tables(store)
    _put_run(
        store,
        "morning-friday",
        "morning",
        "2026-09-25 09:10",
        "ok",
        {"sent_ok": True, "sent_at": "2026-09-25T09:05:00"},
    )
    _put_run(
        store,
        "morning-saturday-no-new-data",
        "morning",
        "2026-09-26 09:10",
        "failed",
        {"business_status": "no_new_data"},
    )

    result = _morning_kpi(
        store,
        {"paper": {"kpi": {"morning_deadline": "09:30"}}},
        asof="2026-09-27",
        lookback_days=3,
    )

    assert result["morning_expected_runs"] == 1
    assert result["morning_observed_runs"] == 1
    assert result["morning_missing_runs"] == 0
    assert result["morning_sent_at"] == "09:05"
    assert result["morning_on_time"] is True
    assert result["morning_on_time_rate"] == 1.0
    store.close()


def test_weekend_proposal_is_not_in_weekday_denominator(tmp_path):
    store = Store(tmp_path / "weekend_proposals.duckdb")
    _ensure_paper_tables(store)
    _put_proposal(store, "20260925-01", "2026-09-25", "proposed")
    _put_proposal(store, "20260926-01", "2026-09-26", "rejected")

    result = _proposal_kpis(store, asof="2026-09-27", lookback_days=3, cfg={})

    assert result["expected_proposals"] == 1
    assert result["observed_proposals"] == 1
    assert result["missing_proposals"] == 0
    store.close()
