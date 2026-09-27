from __future__ import annotations

import json
from datetime import datetime as RealDateTime

import pandas as pd
from typer.testing import CliRunner

import janus.cli as cli
from janus.data.store import Store


class FakeStore:
    snapshot_asof = None

    def __init__(self):
        self.logged = None

    def nav_wide(self):
        return pd.DataFrame({"Y1": [1.0]}, index=pd.DatetimeIndex(["2026-09-24"]))

    def latest_fund_master(self):
        return pd.DataFrame({"fund_code": ["Y1"], "fund_class": ["YAT"]})

    def log_run(self, *args):
        self.logged = args


def _patch_nightly(monkeypatch, tmp_path, fail_features=False, partial_ingest=False):
    import janus.data.ingest_evds as macro
    import janus.data.ingest_tefas as tefas
    import janus.data.quality as quality
    import janus.paper.core as paper
    import janus.paper.fill as fill
    import janus.paper.kpi as kpi
    import janus.paper.reconcile as reconcile
    import janus.paper.shadows as shadows

    st = FakeStore()
    monkeypatch.setattr(cli, "_store", lambda dry_run=False: st)
    monkeypatch.setattr(cli, "load_config", lambda: {"reporting": {"orders_dir": "reports"}})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "_client", lambda dry_run: object())
    monkeypatch.setattr(cli, "_macro_fetcher", lambda dry_run: object())
    monkeypatch.setattr(cli, "_export_snapshot", lambda *args: None)
    monkeypatch.setattr(
        tefas,
        "ingest_tefas",
        lambda *args, **kwargs: {"status": "partial" if partial_ingest else "ok", "n_nav_rows": 1},
    )
    monkeypatch.setattr(macro, "ingest_macro", lambda *args, **kwargs: {"status": "ok", "n_rows": 1})
    monkeypatch.setattr(quality, "quality_summary", lambda *args, **kwargs: {"fresh_ratio": 1.0})

    def features(*args, **kwargs):
        if fail_features:
            raise RuntimeError("synthetic feature failure")
        path = args[4]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return {"status": "ok", "n_rows": 1, "elapsed_seconds": 0.01}

    def predictions(*args, **kwargs):
        path = args[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return {"status": "ok", "n_rows": 1, "elapsed_seconds": 0.01}

    def calibrate(*args, **kwargs):
        path = args[3]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return {"status": "ok", "n_rows": 1, "elapsed_seconds": 0.01}

    monkeypatch.setattr(cli, "_features_build_impl", features)
    monkeypatch.setattr(cli, "_predictions_build_impl", predictions)
    monkeypatch.setattr(cli, "_predictions_calibrate_impl", calibrate)
    monkeypatch.setattr(cli, "_model_kwargs_from_cpcv", lambda: {})
    monkeypatch.setattr(
        cli, "_select_impl", lambda *args, **kwargs: {"status": "ok", "n_rows": 1, "elapsed_seconds": 0.01}
    )
    monkeypatch.setattr(
        cli.pd, "read_parquet", lambda path: pd.DataFrame({"decision_at": pd.to_datetime(["2026-09-24"])})
    )
    monkeypatch.setattr(paper, "paper_propose", lambda *args, **kwargs: {"status": "ok", "n_rows": 1})
    monkeypatch.setattr(fill, "paper_fill_auto", lambda *args, **kwargs: {"status": "ok", "n_rows": 0})
    monkeypatch.setattr(reconcile, "paper_reconcile", lambda *args, **kwargs: {"status": "ok", "n_rows": 1})
    monkeypatch.setattr(shadows, "run_shadows", lambda *args, **kwargs: {})
    monkeypatch.setattr(shadows, "snapshot_shadow_equity", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(kpi, "paper_kpi", lambda *args, **kwargs: {"kpis": {"gate_met": True}, "n_rows": 1})
    return st


def test_nightly_all_steps_ok(monkeypatch, tmp_path):
    st = _patch_nightly(monkeypatch, tmp_path)
    result = CliRunner().invoke(cli.app, ["nightly"])
    assert result.exit_code == 0, result.output
    assert "NIGHTLY ok" in result.output
    assert "steps={" in result.output
    assert st.logged[3] == "ok"
    assert "features" in st.logged[4]


def test_nightly_normalizes_expected_paper_business_statuses(monkeypatch, tmp_path):
    import janus.paper.core as paper
    import janus.paper.fill as fill
    import janus.paper.shadows as shadows

    st = _patch_nightly(monkeypatch, tmp_path)
    expected = cli._expected_decision_date(RealDateTime.now(), {})
    frame = pd.DataFrame({"decision_at": [expected]})
    monkeypatch.setattr(cli.pd, "read_parquet", lambda *_args, **_kwargs: frame.copy())
    monkeypatch.setattr(paper, "paper_propose", lambda *_args, **_kwargs: {"status": "hold", "n_orders": 0})
    monkeypatch.setattr(fill, "paper_fill_auto", lambda *_args, **_kwargs: {"status": "no_proposal"})
    monkeypatch.setattr(shadows, "snapshot_shadow_equity", lambda *_args: {"status": "ok", "n_rows": 0})

    result = CliRunner().invoke(cli.app, ["nightly"])

    assert result.exit_code == 0, result.output
    steps = st.logged[4]
    assert steps["paper_propose"]["status"] == "ok"
    assert steps["paper_propose"]["business_status"] == "hold"
    assert steps["paper_fill_auto"]["status"] == "ok"
    assert steps["paper_fill_auto"]["business_status"] == "no_proposal"
    assert steps["shadow_mtm"]["status"] == "ok"


def test_nightly_failed_step_skips_dependents_but_runs_independent(monkeypatch, tmp_path):
    st = _patch_nightly(monkeypatch, tmp_path, fail_features=True)
    result = CliRunner().invoke(cli.app, ["nightly"])
    assert result.exit_code == 0, result.output
    assert "NIGHTLY failed" in result.output
    steps = st.logged[4]
    assert steps["features"]["status"] == "failed"
    assert steps["predictions"]["status"] == "skipped"
    assert steps["calibrate_020"]["status"] == "skipped"
    assert steps["paper_propose"]["status"] == "skipped"
    assert steps["paper_fill_auto"]["status"] == "ok", repr(steps)
    assert steps["paper_kpi"]["status"] == "ok"


def test_tefas_features_exclude_emk(monkeypatch, tmp_path):
    import janus.backtest.data as backtest_data
    import janus.features.fund_features as fund_features
    import janus.features.macro as macro_features
    import janus.features.market as market_features

    nav = pd.DataFrame({"Y1": [1.0, 1.1], "E1": [2.0, 2.1]}, index=pd.to_datetime(["2026-09-23", "2026-09-24"]))
    fm = pd.DataFrame(
        {
            "fund_code": ["Y1", "E1"],
            "fund_class": ["YAT", "EMK"],
            "umbrella_type": ["Hisse", "Hisse"],
            "withholding_rate": [0.1, 0.0],
        }
    )
    seen = {}
    monkeypatch.setattr(backtest_data, "cash_proxy_codes", lambda *args, **kwargs: [])
    monkeypatch.setattr(backtest_data, "cash_proxy_returns", lambda *args, **kwargs: pd.Series(dtype=float))
    monkeypatch.setattr(backtest_data, "equity_index", lambda *args, **kwargs: pd.Series(dtype=float))
    monkeypatch.setattr(macro_features, "macro_features", lambda *args: pd.DataFrame(index=nav.index))
    monkeypatch.setattr(market_features, "market_features", lambda *args: pd.DataFrame(index=nav.index))

    def build(panel, fund_master, *args, **kwargs):
        seen["funds"] = set(fund_master["fund_code"])
        return pd.DataFrame(
            {
                "feature_asof": [pd.Timestamp("2026-09-24")],
                "fund_code": ["Y1"],
                "label_ready": [False],
                "eligible_at_decision": [True],
            }
        )

    monkeypatch.setattr(fund_features, "build_features", build)
    summary = cli._features_build_impl(nav, fm, {}, FakeStore(), tmp_path / "features.parquet")
    assert seen["funds"] == {"Y1"}
    assert summary["status"] == "ok"
    assert summary["n_rows"] == 1


def test_nightly_partial_ingest_continues_and_sets_partial(monkeypatch, tmp_path):
    st = _patch_nightly(monkeypatch, tmp_path, partial_ingest=True)
    result = CliRunner().invoke(cli.app, ["nightly"])
    assert result.exit_code == 0, result.output
    assert "NIGHTLY partial" in result.output
    assert st.logged[3] == "partial"
    assert st.logged[4]["features"]["status"] == "ok"


def test_nightly_real_cli_chain_uses_synthetic_dry_run_data(tmp_path, monkeypatch):
    """Sabit saatli CLI zinciri üretim adımlarını kullanıp terminal karar günü seçer."""
    import janus.data.ingest_evds as macro_ingest
    import janus.data.ingest_tefas as tefas_ingest
    import janus.data.store as store_module
    import janus.paper.core as paper_core
    from janus.data.tefas_client import FakeTefasClient

    class FrozenDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 22, 23, 30, tzinfo=tz)

    class FrozenProfileDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            # Synthetic profile fetch occurred on the snapshot date, before the next-day decision.
            return cls(2026, 9, 22, 8, 0, tzinfo=tz)

    class FullHistorySyntheticClient(FakeTefasClient):
        def history(self, code, period="5y"):
            return super().history(code, period="5y")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    cfg = cli.load_config()
    cfg["store"]["path"] = str(tmp_path / "janus.duckdb")
    cfg["store"]["parquet_dir"] = str(tmp_path / "curated")
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "_client", lambda _dry_run: FullHistorySyntheticClient(today="2026-09-22"))
    monkeypatch.setattr(cli, "datetime", FrozenDateTime)
    monkeypatch.setattr(tefas_ingest, "datetime", FrozenProfileDateTime)
    # Synthetic execution profiles explicitly carry the PIT fields required by strict B0 selection.
    monkeypatch.setattr(
        tefas_ingest,
        "_tax_fields",
        lambda category, _name, _asof: (
            pd.Series("diger", index=category.index),
            pd.Series(0.175, index=category.index, dtype=float),
        ),
    )

    build_fund_master = tefas_ingest.build_fund_master

    def build_synthetic_profile(*args, **kwargs):
        fm = build_fund_master(*args, **kwargs)
        cash = fm["umbrella_type"].fillna("").str.contains("Para Piyasası", case=False)
        cash &= fm["fund_class"].astype(str).str.upper().eq("YAT")
        fm.loc[cash, "buy_valor"] = 1
        fm.loc[cash, "sell_valor"] = 2
        fm.loc[cash, "can_buy"] = True
        fm.loc[cash, "can_sell"] = True
        fm.loc[cash, "tefas_status"] = "İşlem Görüyor"
        fm.loc[cash, "tax_category"] = "diger"
        fm.loc[cash, "last_success_at"] = pd.Timestamp("2026-09-22 08:00:00")
        fm.loc[cash, "info_ok"] = True
        return fm

    monkeypatch.setattr(tefas_ingest, "build_fund_master", build_synthetic_profile)
    monkeypatch.setattr(macro_ingest, "datetime", FrozenDateTime)
    monkeypatch.setattr(store_module, "datetime", FrozenDateTime)
    monkeypatch.setattr(paper_core, "datetime", FrozenDateTime)

    original_store = cli._store

    def synthetic_store(dry_run=False, read_only=False):
        store = original_store(dry_run=dry_run, read_only=read_only)
        store.snapshot_asof = pd.Timestamp("2026-09-22 08:00:00")
        return store

    monkeypatch.setattr(cli, "_store", synthetic_store)
    from janus.paper.core import _ensure_paper_tables

    store = Store(tmp_path / "janus_dryrun.duckdb")
    _ensure_paper_tables(store)
    store.close()

    result = CliRunner().invoke(cli.app, ["nightly", "--dry-run"])

    assert result.exit_code == 0, result.output
    steps_json = result.output.split("steps=", maxsplit=1)[1]
    steps = json.loads(steps_json)
    assert steps["ingest"]["status"] in {"ok", "partial"}
    assert steps["features"]["status"] == "ok"
    assert steps["predictions"]["status"] == "ok"
    assert steps["predictions"]["n_rows"] > 0
    assert steps["calibrate_020"]["status"] == "ok"
    expected = pd.Timestamp("2026-09-23")
    features = pd.read_parquet(tmp_path / "data" / "features" / "fund_features.parquet")
    terminal = features.loc[pd.to_datetime(features["feature_asof"]) == pd.Timestamp("2026-09-22")]
    assert terminal["decision_at"].eq(expected).all()
    predictions = pd.read_parquet(tmp_path / "data" / "predictions" / "predictions.parquet")
    assert pd.to_datetime(predictions["decision_at"]).eq(expected).any()
    calibrated = pd.read_parquet(tmp_path / "data" / "predictions" / "calibrated_target_020.parquet")
    terminal_cal = calibrated.loc[pd.to_datetime(calibrated["decision_at"]) == expected]
    assert not terminal_cal.empty
    assert terminal_cal["n_calib"].gt(0).any()
    assert terminal_cal["quality_flag"].ne("no_calendar").all()
    assert terminal_cal["lower"].notna().any()
    assert steps["select"]["status"] == "ok", steps["select"]
    assert steps["select"]["date"] == str(expected.date())
    assert steps["paper_propose"]["status"] == "ok", (
        f"status={steps['paper_propose'].get('status')}; warnings={steps['paper_propose'].get('warnings')!r}"
    )
    assert steps["paper_propose"]["business_status"] == "proposed", steps["paper_propose"]
    assert steps["paper_propose"]["date"] == str(expected.date())
    assert steps["paper_propose"]["proposal_id"]
    selected = pd.read_parquet(tmp_path / "data" / "predictions" / "selection_2026-09-23.parquet")
    assert pd.to_datetime(selected["decision_at"]).eq(expected).all()

    store = Store(tmp_path / "janus_dryrun.duckdb", read_only=True)
    assert store.con.execute("SELECT count(*) FROM fund_nav").fetchone()[0] > 0
    assert store.con.execute("SELECT count(*) FROM runs WHERE kind='nightly'").fetchone()[0] == 1
    assert store.con.execute("SELECT count(*) FROM paper_proposals WHERE date='2026-09-23'").fetchone()[0] == 1
    store.close()


def test_expected_decision_date_uses_quality_nav_reference_and_configured_holiday():
    cfg = {"calendar": {"nav_publish_time": "10:00", "holidays": ["2026-09-28"]}}

    decision_date = cli._expected_decision_date(pd.Timestamp("2026-09-25 23:30"), cfg)

    assert decision_date == pd.Timestamp("2026-09-29")


def test_morning_never_serves_previous_decision_proposal(tmp_path, monkeypatch):
    class FrozenDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 25, 8, 45, tzinfo=tz)

    db = tmp_path / "morning.duckdb"
    store = Store(db)
    store.log_run(
        "nightly-old-selection",
        "nightly",
        RealDateTime(2026, 9, 24, 23, 30),
        "ok",
        {
            "quality": {},
            "macro": {"status": "ok"},
            "paper_propose": {"status": "ok", "date": "2026-09-24", "telegram": "STALE ORDER"},
        },
    )
    store.close()
    monkeypatch.setattr(cli, "_store", lambda: Store(db))
    monkeypatch.setattr(cli, "datetime", FrozenDateTime)
    monkeypatch.setattr(cli, "load_config", lambda: {"calendar": {}, "reporting": {}})
    monkeypatch.setattr(cli, "_has_b2c_evidence", lambda _date: True)
    sent = []
    monkeypatch.setattr("janus.report.telegram.send_message", lambda text: sent.append(text) or True)

    result = CliRunner().invoke(cli.app, ["morning"])

    assert result.exit_code == 0, result.output
    assert "STALE ORDER" not in sent[0]
    assert "güncel öneri yok" in sent[0]


def test_weekend_morning_is_successful_no_new_data_without_sending(tmp_path, monkeypatch):
    class FrozenDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 26, 9, 15, tzinfo=tz)

    db = tmp_path / "weekend_morning_cli.duckdb"
    Store(db).close()
    monkeypatch.setattr(cli, "_store", lambda: Store(db))
    monkeypatch.setattr(cli, "datetime", FrozenDateTime)
    monkeypatch.setattr(cli, "load_config", lambda: {"calendar": {"holidays": []}})
    sent = []
    monkeypatch.setattr("janus.report.telegram.send_message", lambda text: sent.append(text) or True)

    result = CliRunner().invoke(cli.app, ["morning"])

    assert result.exit_code == 0, result.output
    assert sent == []
    store = Store(db, read_only=True)
    row = store.con.execute("SELECT status, summary FROM runs WHERE kind='morning'").fetchone()
    assert row is not None and row[0] == "ok"
    assert json.loads(row[1])["business_status"] == "no_new_data"
    store.close()
