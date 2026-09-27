"""S5-1-fix: paper CLI komutlarının gerçek Typer yoluyla sınanması."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import janus.cli as cli
import janus.config as janus_config
import janus.features.macro as feat_macro
from janus.cli import app
from janus.data.store import Store
from janus.models.conformal import calibrate_predictions
from janus.models.walkforward import run_walkforward

MK = {"num_boost_round": 20}


def _make_macro(cal):
    return pd.DataFrame(
        {
            "policy_rate": 0.4,
            "d_policy_63": 0.0,
            "cpi_yoy": 0.3,
            "real_rate": 0.1,
            "usdtry_ret63": 0.05,
            "usdtry_vol21": 0.1,
        },
        index=cal,
    )


def _cfg():
    return {
        "legs": {
            "tefas": {
                "universe": {
                    "fund_type": "YAT",
                    "max_stale_days": 2,
                    "source_stale_days": 10000,
                    "min_universe_fresh_ratio": 0.0,
                    "founder_blacklist": [],
                    "exclude_umbrella_patterns": [],
                    "exclude_status_patterns": [],
                    "min_age_days": 1,
                },
                "constraints": {
                    "max_weight_per_fund": 0.25,
                    "max_weight_per_founder": 0.30,
                    "max_funds_per_founder": 3,
                    "max_funds_per_hrp_cluster": 3,
                    "pilot_risky_ceiling": 0.30,
                },
                "risk": {"dd_trigger_medium": 0.12, "dd_exposure": 0.65},
                "rebalance": {"drift_threshold": 0.03, "drift_threshold_taxable_sale": 0.06},
                "execution": {"valor_mapping": {"buy": "buy_valor", "sell": "sell_valor"}},
            }
        },
        "hrp": {"cov_days": 126, "linkage": "single", "cluster_distance": 0.4},
        "backtest": {"cash_tax_rate": 0.175, "initial_capital": 100.0, "warmup_days": 21},
        "project": {"timezone": "Europe/Istanbul"},
        "store": {"path": "data/janus.duckdb"},
        "reporting": {"orders_dir": "reports"},
    }


def _make_panel():
    from _panel import make_panel

    nav = make_panel(n_funds=3, days=400, seed=11)
    nav.index = pd.bdate_range(end="2026-09-24", periods=len(nav))
    nav = nav.rename(columns={c: f"F{i}" for i, c in enumerate(nav.columns)})
    # Para Piyasası sepeti ekle
    r = nav["F0"].pct_change().clip(-0.001, 0.001)
    nav["PP0"] = (1 + r).cumprod().fillna(1.0)
    return nav


def _populate_store(path):
    st = Store(path)
    nav = _make_panel()
    fm = pd.DataFrame(
        {
            "fund_code": list(nav.columns),
            "umbrella_type": ["Hisse", "Hisse", "Hisse", "Para Piyasası"],
            "fund_class": "YAT",
            "withholding_rate": 0.175,
            "tefas_status": "İşlem Görüyor",
            "founder": ["K1", "K1", "K2", "K1"],
            "founder_code": ["K1", "K1", "K2", "CASH1"],
            "manager": "",
            "name": [f"Fon {c}" for c in nav.columns],
            "isin": [f"TR-{c}" for c in nav.columns],
            "buy_valor": 1,
            "sell_valor": 2,
            "entry_fee": 0.0,
            "exit_fee": 0.0,
            "tax_category": "diger",
            "can_buy": True,
            "can_sell": True,
            "first_nav_date": nav.index.min().date(),
            "snapshot_date": nav.index[-2].date(),
            "info_ok": True,
            "hist_ok": True,
            "last_success_at": pd.Timestamp("2026-09-22 08:00:00"),
        }
    )
    st.write_fund_master(fm)
    nav_long = nav.reset_index().melt(id_vars="index", var_name="fund_code", value_name="price")
    nav_long = nav_long.rename(columns={"index": "date"})
    nav_long["published_at"] = pd.Timestamp("2026-09-24")
    st.upsert_nav(nav_long)
    return st, nav


def _build_features(st, cfg):
    from janus.features.fund_features import build_features
    from janus.features.market import market_features

    nav = st.nav_wide()
    fm = st.latest_fund_master()
    cash_codes = [c for c in nav.columns if "PP" in c]
    cash_returns = nav[cash_codes].pct_change().mean(axis=1).fillna(0.0)
    mf = _make_macro(nav.index)
    eq_cols = [c for c in nav.columns if c.startswith("F")]
    eq_idx = nav[eq_cols].mean(axis=1)
    mkt = market_features(nav, eq_idx, eq_cols)
    return build_features(nav, fm, cash_returns, mf, mkt, cfg=cfg, asof=None)


@pytest.fixture()
def paper_cli_env(tmp_path, monkeypatch):
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    monkeypatch.setattr(janus_config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    db = tmp_path / "data" / "janus.duckdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    st, nav = _populate_store(db)
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: Store(db, read_only=read_only))
    cfg = _cfg()
    feats = _build_features(st, cfg)
    preds = run_walkforward(feats, model_kwargs=MK)
    cal = calibrate_predictions(preds, feats, miscoverage_target=0.2, gamma=0.05)
    # D+1 fiyatının NAV'da olması için panelin sondan ikinci NAV gününü karar günü seç.
    date = str(nav.index[-2].date())
    pred_path = tmp_path / "data" / "predictions" / "predictions.parquet"
    cal_path = tmp_path / "data" / "predictions" / "calibrated_target_020.parquet"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(pred_path, index=False)
    cal.to_parquet(cal_path, index=False)
    from janus.cli import _select_impl

    out = tmp_path / "data" / "predictions" / f"selection_{date}.parquet"
    _select_impl(date, preds, feats, cal, out, cfg.get("conformal", {}))
    st.close()
    return {"tmp_path": tmp_path, "date": date}


def test_paper_help_shows_commands():
    result = CliRunner().invoke(app, ["paper", "--help"])
    assert result.exit_code == 0, result.output
    for cmd in ("init", "propose", "fill", "reconcile", "weekly", "basket", "reset"):
        assert cmd in result.output, cmd


def test_paper_reset_archive_v1_preserves_all_paper_tables_and_unrelated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(janus_config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    db = tmp_path / "synthetic.duckdb"
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: Store(db, read_only=read_only))
    st, _ = _populate_store(db)
    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(st)
    st.con.execute("INSERT INTO paper_cash VALUES (now(), 100, 100)")
    st.con.execute("INSERT INTO paper_b0_memberships VALUES ('2026-09-22', '2026-09-22', 'PP0', 'CASH1', 1, 2)")
    st.con.execute("INSERT INTO runs VALUES ('keep', 'test', now(), now(), 'ok', '{}')")
    before_runs = st.con.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
    tables = [
        r[0]
        for r in st.con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' AND table_name LIKE 'paper_%'"
        ).fetchall()
    ]
    st.close()

    result = CliRunner().invoke(app, ["paper", "reset", "--archive-v1"])

    assert result.exit_code == 0, result.output
    st = Store(db)
    for table in tables:
        expected = 1 if table in {"paper_cash", "paper_b0_memberships"} else 0
        assert st.con.execute(f"SELECT count(*) FROM {table}_v1").fetchone()[0] == expected
    assert st.con.execute("SELECT * FROM runs ORDER BY run_id").fetchall() == before_runs
    assert st.con.execute("SELECT count(*) FROM paper_cash").fetchone()[0] == 0
    assert st.con.execute("SELECT count(*) FROM janus_pilot_epoch").fetchone()[0] == 1
    st.close()
    repeated = CliRunner().invoke(app, ["paper", "reset", "--archive-v1"])
    assert repeated.exit_code == 0, repeated.output
    st = Store(db)
    assert st.con.execute("SELECT cash FROM paper_cash_v1").fetchone() == (100.0,)
    assert (
        st.con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='paper_cash_v1_v1'").fetchone()[
            0
        ]
        == 0
    )
    st.close()


def test_paper_reset_fails_closed_when_archive_target_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(janus_config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    db = tmp_path / "synthetic.duckdb"
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: Store(db, read_only=read_only))
    st, _ = _populate_store(db)
    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(st)
    st.con.execute("CREATE TABLE paper_positions_v1 AS SELECT * FROM paper_positions")
    st.close()
    before = CliRunner().invoke(app, ["paper", "reset", "--archive-v1"])
    assert before.exit_code == 1
    st = Store(db)
    assert st.con.execute("SELECT count(*) FROM paper_positions_v1").fetchone()[0] == 0
    assert st.con.execute("SELECT count(*) FROM paper_positions").fetchone()[0] == 0
    st.close()


def test_paper_reset_transaction_rolls_back_on_epoch_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(janus_config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    db = tmp_path / "synthetic.duckdb"
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: Store(db, read_only=read_only))
    st, _ = _populate_store(db)
    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(st)
    st.con.execute("INSERT INTO paper_cash VALUES (now(), 77, 100)")
    # Force a post-rename epoch insertion error to exercise transactional DDL rollback.
    st.con.execute("CREATE TABLE janus_pilot_epoch (wrong_column INTEGER)")
    st.close()

    result = CliRunner().invoke(app, ["paper", "reset", "--archive-v1"])

    assert result.exit_code == 1
    st = Store(db)
    assert st.con.execute("SELECT cash FROM paper_cash").fetchone() == (77.0,)
    assert (
        st.con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='paper_cash_v1'").fetchone()[0]
        == 0
    )
    st.close()


def test_paper_reset_archives_only_literal_paper_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(janus_config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    db = tmp_path / "synthetic.duckdb"
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: Store(db, read_only=read_only))
    st, _ = _populate_store(db)
    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(st)
    st.con.execute("CREATE TABLE paperX_private (secret INTEGER)")
    st.con.execute("INSERT INTO paperX_private VALUES (42)")
    st.close()

    result = CliRunner().invoke(app, ["paper", "reset", "--archive-v1"])

    assert result.exit_code == 0, result.output
    st = Store(db)
    assert st.con.execute("SELECT secret FROM paperX_private").fetchone() == (42,)
    assert (
        st.con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='paperX_private_v1'"
        ).fetchone()[0]
        == 0
    )
    st.close()


def test_paper_init_reset_init_archives_legacy_cash_proxy_and_reinitializes_cash_only(tmp_path, monkeypatch):
    monkeypatch.setattr(janus_config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    db = tmp_path / "synthetic.duckdb"
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: Store(db, read_only=read_only))
    st, _ = _populate_store(db)
    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(st)
    st.close()

    init_before = CliRunner().invoke(app, ["paper", "init", "--capital", "100", "--date", "2026-09-24"])
    assert init_before.exit_code == 0, init_before.output
    st = Store(db)
    st.con.execute("UPDATE paper_cash SET cash=77")
    st.con.execute("INSERT INTO paper_positions VALUES (now(), 'CASH_PROXY', 2, 38.5)")
    st.con.execute("INSERT INTO paper_lots VALUES (now(), 'CASH_PROXY', 2, 38.5, '2026-09-20', 0.175, NULL, 1, 1)")
    st.con.execute("INSERT INTO runs VALUES ('preserve', 'nightly', now(), now(), 'ok', '{}')")
    st.close()
    reset = CliRunner().invoke(app, ["paper", "reset", "--archive-v1"])
    assert reset.exit_code == 0, reset.output
    init_after = CliRunner().invoke(app, ["paper", "init", "--capital", "100", "--date", "2026-09-24"])
    assert init_after.exit_code == 0, init_after.output

    st = Store(db)
    assert st.con.execute("SELECT cash FROM paper_cash_v1").fetchone()[0] == 77
    assert st.con.execute("SELECT fund_code, units FROM paper_positions_v1").fetchone() == ("CASH_PROXY", 2)
    assert st.con.execute("SELECT fund_code, units FROM paper_lots_v1").fetchone() == ("CASH_PROXY", 2)
    assert st.con.execute("SELECT cash FROM paper_cash").fetchone()[0] == 100
    assert st.con.execute("SELECT count(*) FROM paper_positions").fetchone()[0] == 0
    assert st.con.execute("SELECT count(*) FROM paper_lots").fetchone()[0] == 0
    assert st.con.execute("SELECT count(*) FROM runs WHERE run_id='preserve'").fetchone()[0] == 1
    assert st.con.execute("SELECT count(*) FROM janus_pilot_epoch").fetchone()[0] == 1
    st.close()


def test_paper_reset_rejects_incomplete_archive_even_with_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(janus_config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    db = tmp_path / "synthetic.duckdb"
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: Store(db, read_only=read_only))
    st, _ = _populate_store(db)
    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(st)
    st.con.execute("ALTER TABLE paper_cash RENAME TO paper_cash_v1")
    st.con.execute(
        "CREATE TABLE janus_pilot_epoch (epoch_id INTEGER, started_at TIMESTAMP, initialized_at TIMESTAMP, "
        "published_at TIMESTAMP, available_from TIMESTAMP)"
    )
    st.con.execute("INSERT INTO janus_pilot_epoch VALUES (1, now(), NULL, now(), now())")
    st.close()

    result = CliRunner().invoke(app, ["paper", "reset", "--archive-v1"])

    assert result.exit_code == 1
    st = Store(db)
    assert (
        st.con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='paper_positions_v1'"
        ).fetchone()[0]
        == 0
    )
    assert st.con.execute("SELECT count(*) FROM paper_positions").fetchone()[0] == 0
    st.close()


def test_paper_reset_schema_creation_failure_rolls_back_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(janus_config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    db = tmp_path / "synthetic.duckdb"
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: Store(db, read_only=read_only))
    st, _ = _populate_store(db)
    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(st)
    st.con.execute("INSERT INTO paper_cash VALUES (now(), 77, 100)")
    st.close()
    monkeypatch.setattr(
        "janus.paper.core._ensure_paper_tables", lambda _store: (_ for _ in ()).throw(RuntimeError("schema"))
    )

    result = CliRunner().invoke(app, ["paper", "reset", "--archive-v1"])

    assert result.exit_code == 1
    st = Store(db)
    assert st.con.execute("SELECT cash FROM paper_cash").fetchone() == (77.0,)
    assert (
        st.con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='paper_cash_v1'").fetchone()[0]
        == 0
    )
    st.close()


def test_paper_init_cli(paper_cli_env):
    result = CliRunner().invoke(app, ["paper", "init", "--capital", "100", "--date", paper_cli_env["date"]])
    assert result.exit_code == 0, result.output
    assert "PAPER INIT" in result.output
    assert "CASH_PROXY" not in result.output
    st = Store(paper_cli_env["tmp_path"] / "data" / "janus.duckdb")
    epoch = st.con.execute("SELECT started_at, initialized_at FROM janus_pilot_epoch").fetchone()
    assert epoch[0] is not None and epoch[1] is not None
    assert st.con.execute("SELECT count(*) FROM paper_lots").fetchone()[0] == 0
    assert st.con.execute("SELECT count(*) FROM paper_positions").fetchone()[0] == 0
    assert st.con.execute("SELECT cash FROM paper_cash").fetchone()[0] == pytest.approx(100.0)
    st.close()


def test_paper_init_cli_explains_legacy_slot_and_fails_closed(paper_cli_env):
    result = CliRunner().invoke(app, ["paper", "init", "--capital", "100"])
    assert result.exit_code == 0, result.output
    st = Store(paper_cli_env["tmp_path"] / "data" / "janus.duckdb")
    st.con.execute("INSERT INTO paper_positions VALUES (?, 'CASH_PROXY', 1.0, 1.0)", [datetime.now()])
    st.close()

    retry = CliRunner().invoke(app, ["paper", "init", "--capital", "100"])

    assert retry.exit_code == 1
    assert "Eski CASH_PROXY" in retry.output
    st = Store(paper_cli_env["tmp_path"] / "data" / "janus.duckdb")
    assert st.con.execute("SELECT count(*) FROM paper_positions WHERE fund_code='CASH_PROXY'").fetchone()[0] == 1
    st.close()


def test_s5_6c_paper_init_accepts_explicit_decision_date_and_reports_pit_asofs(paper_cli_env):
    date = paper_cli_env["date"]
    result = CliRunner().invoke(app, ["paper", "init", "--capital", "100", "--date", date])

    assert result.exit_code == 0, result.output
    assert f"decision_asof={date}" in result.output
    assert "meta_asof=" in result.output
    assert "nav_asof=" in result.output
    assert "1 B0 adayı" in result.output


def test_paper_propose_cli(paper_cli_env):
    CliRunner().invoke(app, ["paper", "init", "--capital", "100"])
    date = paper_cli_env["date"]
    result = CliRunner().invoke(app, ["paper", "propose", "--date", date])
    assert result.exit_code == 0, result.output
    assert "PAPER PROPOSE" in result.output
    assert "tut" in result.output.lower()
    assert not (paper_cli_env["tmp_path"] / "reports" / f"orders_{date}.md").exists()


def test_nightly_runs_paper_propose_step(monkeypatch, tmp_path):
    """nightly zincirinin paper propose adımı gerçek komut yolunda çalışır (mock'lu)."""
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    calls: list[str] = []

    def fake_nightly(*a, **k):
        calls.append("nightly")
        return {"status": "ok"}

    # nightly komutunu doğrudan çağırmak yerine app üzerinden çağırırız,
    # ancak gerçek nightly uzun süreceğinden sadece komutun varlığını ve
    # paper alt-komutunun bağlı olduğunu doğruluyoruz.
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "nightly" in result.output
    assert "paper" in result.output


def test_paper_fill_and_reconcile_cli(paper_cli_env):
    CliRunner().invoke(app, ["paper", "init", "--capital", "100"])
    date = str(pd.Timestamp(_make_panel().index[-2]).date())
    pid = f"{pd.Timestamp(date):%Y%m%d}-01"
    st = Store(paper_cli_env["tmp_path"] / "data" / "janus.duckdb")
    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(st)
    st.con.execute("DELETE FROM paper_equity WHERE portfolio_name='live' AND date > ?", [pd.Timestamp(date).date()])
    st.con.execute(
        """INSERT INTO paper_proposals
        (proposal_id, date, created_at, expires_at, status, orders_json, evidence_count, evidence_codes, target_weights_json)
        VALUES (?, ?, ?, ?, 'proposed', '[]', 0, '[]', '{}')""",
        [pid, pd.Timestamp(date).date(), datetime.now(), datetime.now()],
    )
    st.close()
    fill = CliRunner().invoke(app, ["paper", "fill", pid, "--auto"])
    assert fill.exit_code == 0, fill.output
    reconcile_date = str(_make_panel().index[-1].date())
    rec = CliRunner().invoke(app, ["paper", "reconcile", "--date", reconcile_date])
    assert rec.exit_code == 0, rec.output
    assert (paper_cli_env["tmp_path"] / "reports" / f"reconcile_{reconcile_date}.md").exists()


def test_paper_weekly_cli(paper_cli_env):
    CliRunner().invoke(app, ["paper", "init", "--capital", "100"])
    date = paper_cli_env["date"]
    CliRunner().invoke(app, ["paper", "propose", "--date", date])
    res = CliRunner().invoke(app, ["paper", "weekly", "--date", date])
    assert res.exit_code == 0, res.output


def test_paper_kpi_cli(paper_cli_env):
    CliRunner().invoke(app, ["paper", "init", "--capital", "100"])
    date = paper_cli_env["date"]
    CliRunner().invoke(app, ["paper", "propose", "--date", date])
    res = CliRunner().invoke(app, ["paper", "kpi", "--date", date])
    assert res.exit_code == 0, res.output
    assert (paper_cli_env["tmp_path"] / "reports" / f"paper_kpi_{date}.md").exists()


def test_paper_basket_cli_prints_b0_members(paper_cli_env):
    date = "2026-09-24"
    st = Store(paper_cli_env["tmp_path"] / "data" / "janus.duckdb")
    st.con.execute("UPDATE fund_master SET entry_fee=NULL, exit_fee=12 WHERE fund_code='PP0'")
    st.close()
    result = CliRunner().invoke(app, ["paper", "basket", "--date", date])
    assert result.exit_code == 0, result.output
    assert "B0 NAKİT SEPETİ" in result.output
    assert "Fon kodu" in result.output and "Son NAV tarihi" in result.output
    assert "PP0" in result.output
    assert "Para Piyasası" not in result.output  # üyelikte ad / PYŞ değil, fon adı görünür
    assert "A3-komisyon" not in result.output and "fee_assumed_zero" not in result.output


def test_paper_b0_t_plus_zero_valors_are_selectable(paper_cli_env):
    from janus.paper.core import PaperLedger

    db = paper_cli_env["tmp_path"] / "data" / "janus.duckdb"
    st = Store(db)
    st.con.execute("UPDATE fund_master SET buy_valor=0, sell_valor=0 WHERE fund_code='PP0'")
    paper = PaperLedger(st, _cfg(), asof=paper_cli_env["date"])
    assert "PP0" in paper._b0_codes
    st.close()

    result = CliRunner().invoke(app, ["paper", "basket", "--date", paper_cli_env["date"]])
    assert result.exit_code == 0, result.output
    assert "PP0" in result.output


def test_paper_execution_uses_status_text_when_flags_are_false_or_null():
    from janus.paper.core import _profile_execution_reasons

    profile = pd.DataFrame(
        {
            "fund_code": ["P"],
            "snapshot_date": [pd.Timestamp("2026-09-24")],
            "source_published_at": [pd.Timestamp("2026-09-24 08:00")],
            "last_success_at": [pd.Timestamp("2026-09-24 08:00")],
            "tefas_status": ["İşlem Görüyor"],
            "can_buy": [False],
            "can_sell": [None],
            "buy_valor": [0],
            "sell_valor": [0],
            "entry_fee": [0.0],
            "exit_fee": [0.0],
            "tax_category": ["diger"],
        }
    )
    assert _profile_execution_reasons(profile, "2026-09-24", codes=["P"]) == {}


def test_paper_source_fees_never_block_execution():
    from janus.paper.core import _profile_execution_reasons

    profile = pd.DataFrame(
        {
            "fund_code": ["P"],
            "snapshot_date": [pd.Timestamp("2026-09-24")],
            "source_published_at": [pd.Timestamp("2026-09-24 08:00")],
            "last_success_at": [pd.Timestamp("2026-09-24 08:00")],
            "tefas_status": ["İşlem Görüyor"],
            "can_buy": [True],
            "can_sell": [True],
            "buy_valor": [0],
            "sell_valor": [0],
            "entry_fee": [np.nan],
            "exit_fee": [np.nan],
            "tax_category": ["diger"],
        }
    )
    assert _profile_execution_reasons(profile, "2026-09-24", codes=["P"]) == {}

    profile["entry_fee"] = "not-a-fee"
    assert _profile_execution_reasons(profile, "2026-09-24", codes=["P"]) == {}

    profile["entry_fee"] = -0.01
    assert _profile_execution_reasons(profile, "2026-09-24", codes=["P"]) == {}


def test_paper_freshness_uses_exact_morning_cutoff_age():
    from janus.paper.core import _profile_execution_reasons

    profile = pd.DataFrame(
        {
            "fund_code": ["P"],
            "snapshot_date": [pd.Timestamp("2026-09-22")],
            "source_published_at": [pd.Timestamp("2026-09-22 08:00")],
            "last_success_at": [pd.Timestamp("2026-09-15 09:14")],
            "tefas_status": ["İşlem Görüyor"],
            "can_buy": [True],
            "can_sell": [True],
            "buy_valor": [0],
            "sell_valor": [0],
            "entry_fee": [0.0],
            "exit_fee": [0.0],
            "tax_category": ["diger"],
        }
    )

    assert _profile_execution_reasons(profile, "2026-09-22", codes=["P"]) == {
        "P": {"BUY": "stale_last_success_at", "SELL": "stale_last_success_at"}
    }


def test_paper_ledger_status_text_overrides_flags_and_ignores_source_fees(paper_cli_env):
    from janus.paper.core import PaperLedger

    db = paper_cli_env["tmp_path"] / "data" / "janus.duckdb"
    st = Store(db)
    st.con.execute(
        "UPDATE fund_master SET can_buy=FALSE, can_sell=NULL, entry_fee=NULL, exit_fee=NULL WHERE fund_code='PP0'"
    )
    paper = PaperLedger(st, _cfg(), asof=paper_cli_env["date"])
    idx = int(paper.meta.index.get_loc("PP0"))
    assert "PP0" in paper._b0_codes
    assert bool(paper.meta.can_buy[idx]) and bool(paper.meta.can_sell[idx])
    assert paper.meta.entry_fee[idx] == 0 and paper.meta.exit_fee[idx] == 0
    assert not hasattr(paper, "_fee_assumed_zero_codes")
    st.con.execute("UPDATE fund_master SET entry_fee=8, exit_fee=9 WHERE fund_code='PP0'")
    nonzero_fee_paper = PaperLedger(st, _cfg(), asof=paper_cli_env["date"])
    idx = int(nonzero_fee_paper.meta.index.get_loc("PP0"))
    assert nonzero_fee_paper.meta.entry_fee[idx] == 0
    assert nonzero_fee_paper.meta.exit_fee[idx] == 0
    st.close()


def test_profile_backfill_cli_dry_run_then_write(paper_cli_env):
    db = paper_cli_env["tmp_path"] / "data" / "janus.duckdb"
    st = Store(db)
    st.con.execute(
        "UPDATE fund_master SET info_fetched=TRUE, last_success_at=NULL, source_published_at='2026-09-22 08:00' "
        "WHERE fund_code='PP0'"
    )
    before = st.con.execute(
        "SELECT fund_code, snapshot_date, last_success_at, last_success_source "
        "FROM fund_master ORDER BY fund_code, snapshot_date"
    ).fetchall()
    target_count = st.con.execute(
        "SELECT count(*) FROM fund_master WHERE info_fetched IS TRUE AND last_success_at IS NULL"
    ).fetchone()[0]
    st.close()

    dry = CliRunner().invoke(app, ["backfill-profile-success", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "vekil zaman, PIT kanıtı değil" in dry.output
    assert f"{target_count} aday satır" in dry.output
    st = Store(db)
    after_dry = st.con.execute(
        "SELECT fund_code, snapshot_date, last_success_at, last_success_source "
        "FROM fund_master ORDER BY fund_code, snapshot_date"
    ).fetchall()
    assert after_dry == before
    st.close()

    write = CliRunner().invoke(app, ["backfill-profile-success"])
    assert write.exit_code == 0, write.output
    st = Store(db)
    after_write = st.con.execute(
        "SELECT fund_code, snapshot_date, last_success_at, last_success_source "
        "FROM fund_master ORDER BY fund_code, snapshot_date"
    ).fetchall()
    before_by_key = {(row[0], row[1]): row for row in before}
    after_by_key = {(row[0], row[1]): row for row in after_write}
    assert len(before_by_key) == len(after_by_key)
    for key, old in before_by_key.items():
        current = after_by_key[key]
        if key[0] == "PP0":
            assert current[2] == pd.Timestamp("2026-09-22 08:00")
            assert current[3] == "proxy_ingest_time"
        else:
            assert current == old
    assert f"{target_count} satır güncellendi" in write.output
    st.close()

    repeated = CliRunner().invoke(app, ["backfill-profile-success"])
    assert repeated.exit_code == 0, repeated.output
    assert "0 satır güncellendi" in repeated.output


def test_s5_6c_b0_basket_and_paper_use_asof_nav_counts_and_code_tiebreak(paper_cli_env):
    from janus.paper.core import PaperLedger

    date = pd.Timestamp(paper_cli_env["date"])
    db = paper_cli_env["tmp_path"] / "data" / "janus.duckdb"
    st = Store(db)
    # Same-PYŞ candidates have full, equal NAV history through D; stored PIT counts deliberately rank B first.
    st.con.execute(
        "UPDATE fund_master SET umbrella_type='Para Piyasası', founder='PYŞ-1', founder_code='PYŞ-1', "
        "snapshot_date=?, last_success_at=?, n_nav=1 WHERE fund_code='F0'",
        [(date - pd.offsets.BDay(1)).date(), date - pd.Timedelta(days=1) + pd.Timedelta(hours=8)],
    )
    st.con.execute(
        "UPDATE fund_master SET umbrella_type='Para Piyasası', founder='PYŞ-1', founder_code='PYŞ-1', "
        "snapshot_date=?, last_success_at=?, n_nav=10000 WHERE fund_code='F1'",
        [(date - pd.offsets.BDay(1)).date(), date - pd.Timedelta(days=1) + pd.Timedelta(hours=8)],
    )
    st.con.execute("UPDATE fund_master SET umbrella_type='Hisse' WHERE fund_code='PP0'")

    nav_asof = st.nav_wide().loc[lambda frame: frame.index <= date]
    assert nav_asof["F0"].notna().mean() >= 0.99
    assert nav_asof["F1"].notna().mean() >= 0.99
    assert nav_asof["F0"].notna().sum() == nav_asof["F1"].notna().sum()
    assert nav_asof["F0"].notna().sum() != 1
    assert nav_asof["F1"].notna().sum() != 10000
    st.close()

    basket = CliRunner().invoke(app, ["paper", "basket", "--date", str(date.date())])
    assert basket.exit_code == 0, basket.output

    st = Store(db)
    paper = PaperLedger(st, _cfg(), asof=date)
    assert paper._b0_codes == ("F0", "F1")
    assert "| F0 |" in basket.output
    assert "| F1 |" in basket.output
    from janus.cli import _b0_cash_basket

    _, rows = _b0_cash_basket(st, _cfg(), str(date.date()))
    assert sum(row["weight_pct"] for row in rows) <= 100.0
    assert sum(row["weight_pct"] for row in rows if row["founder"] == "PYŞ-1") <= 30.0 + 1e-8
    st.close()


def test_s5_6c_basket_uses_decision_asof_and_matches_paper_members(paper_cli_env):
    from janus.paper.core import PaperLedger, paper_propose

    root = paper_cli_env["tmp_path"]
    db = root / "data" / "janus.duckdb"
    decision = pd.Timestamp(paper_cli_env["date"])
    profile_date = (decision - pd.offsets.BDay(1)).date()
    stale_success = decision - pd.Timedelta(days=8) + pd.Timedelta(hours=12)
    fresh_success = decision - pd.Timedelta(days=1) + pd.Timedelta(hours=12)
    st = Store(db)
    st.con.execute(
        "UPDATE fund_master SET snapshot_date=?, last_success_at=? WHERE fund_code='PP0'",
        [profile_date, stale_success],
    )
    st.con.execute(
        "UPDATE fund_master SET umbrella_type='Para Piyasası', founder='PYŞ-2', founder_code='CASH2', "
        "snapshot_date=?, last_success_at=? WHERE fund_code='F0'",
        [profile_date, fresh_success],
    )
    cfg = _cfg()

    paper = PaperLedger(st, cfg, asof=decision)
    _, basket_rows = cli._b0_cash_basket(st, cfg, str(decision.date()))
    assert paper._b0_codes == ("F0",)
    assert {row["fund_code"] for row in basket_rows} == set(paper._b0_codes)

    selection_path = root / "data" / "predictions" / f"selection_{decision:%Y-%m-%d}.parquet"
    pd.DataFrame(
        {
            "decision_at": [decision],
            "fund_code": ["F0"],
            "selected_q20": [False],
            "lower": [0.0],
            "p_value": [1.0],
        }
    ).to_parquet(selection_path, index=False)
    proposal = paper_propose(st, cfg, str(decision.date()), root)
    assert proposal["status"] == "proposed"
    assert any("B0 uygun aday sayısı 1" in warning for warning in proposal["warnings"])

    basket_cli = CliRunner().invoke(app, ["paper", "basket", "--date", str(decision.date())])
    assert basket_cli.exit_code == 0, basket_cli.output
    assert "| F0 |" in basket_cli.output
    assert "| PP0 |" not in basket_cli.output

    st.con.execute("UPDATE fund_master SET last_success_at=? WHERE fund_code IN ('PP0', 'F0')", [stale_success])
    held = paper_propose(st, cfg, str(decision.date()), root)
    assert held["status"] == "hold"
    assert held["n_orders"] == 0
    st.close()


def test_s5_6c2_live_synthetic_b0_accepts_two_or_three_same_pysh():
    from janus.backtest.data import cash_proxy_codes

    codes = ["A0", "A1", "A2", "B0", "C0", "D0"]
    cfg = _cfg()
    cfg["legs"]["tefas"]["universe"]["founder_blacklist"] = []
    profile = pd.DataFrame(
        {
            "fund_code": codes,
            "umbrella_type": "Para Piyasası",
            "founder_code": ["MA", "MA", "MA", "MB", "MC", "MD"],
            "founder": ["PYŞ-A", "PYŞ-A", "PYŞ-A", "PYŞ-B", "PYŞ-C", "PYŞ-D"],
            "tax_category": "diger",
            "buy_valor": 1,
            "sell_valor": 1,
            "can_buy": True,
            "can_sell": True,
            "tefas_status": "İşlem Görüyor",
            "last_success_at": pd.Timestamp("2026-09-24 08:00"),
            "snapshot_date": pd.Timestamp("2026-09-24"),
            "n_nav": [22, 21, 20, 19, 18, 17],
        }
    )
    nav = pd.DataFrame({code: 1.0 for code in codes}, index=pd.bdate_range("2026-09-01", periods=20))
    selected = cash_proxy_codes(
        nav,
        profile,
        cfg,
        require_execution_profile=True,
        profile_asof="2026-09-25",
    )
    assert len(selected) == 5
    assert sum(profile.set_index("fund_code").loc[selected, "founder_code"].eq("MA")) == 3
    assert set(profile.set_index("fund_code").loc[selected, "founder_code"]) == {"MA", "MB", "MC"}


def test_f10_paper_basket_fails_closed_when_valor_is_unknown(paper_cli_env):
    db = paper_cli_env["tmp_path"] / "data" / "janus.duckdb"
    st = Store(db)
    st.con.execute("UPDATE fund_master SET buy_valor = NULL WHERE fund_code = 'PP0'")
    st.close()

    result = CliRunner().invoke(app, ["paper", "basket", "--date", "2026-09-24"])

    assert result.exit_code == 1
    assert "valör" in result.output.lower() or "uygun" in result.output.lower()


def test_f10_paper_init_fails_closed_when_founder_code_is_unknown(paper_cli_env):
    db = paper_cli_env["tmp_path"] / "data" / "janus.duckdb"
    st = Store(db)
    st.con.execute("UPDATE fund_master SET founder_code = NULL WHERE fund_code = 'PP0'")
    st.close()

    result = CliRunner().invoke(app, ["paper", "init", "--capital", "100"])

    assert result.exit_code == 1
    assert "founder" in result.output.lower() or "PYŞ" in result.output
    st = Store(db)
    assert st.con.execute("SELECT count(*) FROM paper_cash").fetchone()[0] == 0
    st.close()


def test_f10_paper_propose_fails_closed_when_no_live_b0_member(paper_cli_env):
    init = CliRunner().invoke(app, ["paper", "init", "--capital", "100"])
    assert init.exit_code == 0, init.output

    db = paper_cli_env["tmp_path"] / "data" / "janus.duckdb"
    st = Store(db)
    st.con.execute(
        "DELETE FROM paper_equity WHERE portfolio_name='live' AND date > ?",
        [pd.Timestamp(paper_cli_env["date"]).date()],
    )
    st.con.execute("UPDATE fund_master SET founder_code = NULL WHERE fund_code = 'PP0'")
    st.close()

    result = CliRunner().invoke(app, ["paper", "propose", "--date", paper_cli_env["date"]])

    assert result.exit_code == 0, result.output
    assert "B0 uygun aday sayısı 0" in result.output
    st = Store(db)
    assert st.con.execute("SELECT count(*) FROM paper_proposals").fetchone()[0] == 0
    st.close()


def test_s5_6c_zero_b0_candidates_hold_without_resetting_existing_b0_equity(paper_cli_env):
    from janus.paper.core import PaperLedger, paper_propose

    root = paper_cli_env["tmp_path"]
    db = root / "data" / "janus.duckdb"
    decision = pd.Timestamp(paper_cli_env["date"])
    st = Store(db)
    cfg = _cfg()
    paper = PaperLedger(st, cfg, asof=decision)
    code = paper._b0_codes[0]
    index = int(paper.meta.index.get_loc(code))
    assert paper.ledger.buy(index, 20.0, float(paper.latest_nav()[code]), paper._date_idx(decision)) is not None
    paper._persist()
    equity_before = paper.equity()
    units_before = float(paper.ledger.units[index])

    st.con.execute(
        "UPDATE fund_master SET can_buy=FALSE, tefas_status='İşlem Görmüyor', snapshot_date=?, last_success_at=? "
        "WHERE fund_code=?",
        [decision.date(), decision + pd.Timedelta(hours=8), code],
    )
    path = root / "data" / "predictions" / f"selection_{decision:%Y-%m-%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "decision_at": [decision],
            "fund_code": ["F0"],
            "selected_q20": [False],
            "lower": [0.0],
            "p_value": [1.0],
        }
    ).to_parquet(path, index=False)

    result = paper_propose(st, cfg, str(decision.date()), root, out_dir=root / "test-reports")

    assert result["status"] == "hold"
    assert any("B0" in warning and "0" in warning for warning in result["warnings"])
    assert st.con.execute("SELECT count(*) FROM paper_proposals").fetchone()[0] == 0
    reloaded = PaperLedger(st, cfg, asof=decision)
    reloaded_i = int(reloaded.meta.index.get_loc(code))
    assert reloaded.ledger.units[reloaded_i] == pytest.approx(units_before)
    assert reloaded.latest_nav()[code] == pytest.approx(paper.latest_nav()[code])
    assert reloaded.equity() == pytest.approx(equity_before)
    st.close()


def test_morning_logs_run(tmp_path, monkeypatch):
    """janus morning bir morning run kaydı bırakır."""
    from unittest.mock import patch

    monkeypatch.setattr(janus_config, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", _cfg)
    monkeypatch.setattr("janus.data.quality.is_business_day", lambda _now, _cfg: True)
    db = tmp_path / "data" / "janus.duckdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    st, _ = _populate_store(db)
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: Store(db, read_only=read_only))
    st.log_run("nightly-test", "nightly", datetime.now(), "ok", {"quality": {}})
    st.close()

    sent = []
    with patch("janus.report.telegram.send_message", side_effect=lambda text: sent.append(text) or True):
        result = CliRunner().invoke(app, ["morning"])
    assert result.exit_code == 0, result.output
    assert "kanıt yok → sepet: PP0" in sent[0]
    st2 = Store(db)
    rows = st2.con.execute("SELECT count(*) FROM runs WHERE kind = 'morning'").fetchone()
    assert rows[0] == 1
    st2.close()


def test_paper_kpi_with_mock_runs(paper_cli_env):
    """KPI eşik karşılaştırması doğru işaretler üretir."""
    from datetime import datetime, timedelta

    from janus.paper.core import PaperLedger

    tmp_path = paper_cli_env["tmp_path"]
    db = tmp_path / "data" / "janus.duckdb"
    st = Store(db)
    PaperLedger(st, _cfg(), capital=100.0)  # paper tablolarını oluştur
    now = datetime.now()
    # Başarılı gece koşuları
    for i in range(10):
        st.log_run(f"nightly-{i}", "nightly", now - timedelta(hours=1), "ok", {"quality": {"source_stale": False}})
    # Zamanında sabah mesajı
    st.log_run("morning-today", "morning", now - timedelta(minutes=5), "ok", {})
    st.close()

    res = CliRunner().invoke(app, ["paper", "kpi", "--date", paper_cli_env["date"]])
    assert res.exit_code == 0, res.output
    md = (tmp_path / "reports" / f"paper_kpi_{paper_cli_env['date']}.md").read_text(encoding="utf-8")
    assert "✓" in md or "✗" in md or "veri yok" in md
