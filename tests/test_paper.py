"""S5-1: kağıt defter ve öneri listesi testleri."""

from __future__ import annotations

import pandas as pd
import pytest

import janus.cli as cli
import janus.features.macro as feat_macro
from _panel import make_panel
from janus.backtest.data import prepare
from janus.data.store import Store
from janus.models.conformal import calibrate_predictions
from janus.models.walkforward import run_walkforward
from janus.paper.core import PaperLedger, paper_propose, target_weights_from_selection

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
                "rebalance": {
                    "drift_threshold": 0.03,
                    "drift_threshold_taxable_sale": 0.06,
                    "schedule": "monthly_first_business_day",
                },
                "risk": {
                    "dd_trigger_medium": 0.12,
                    "dd_exposure": 0.65,
                    "dd_release": 0.06,
                    "exposure_levels": {"low": 0.0, "medium": 0.65, "high": 1.0},
                    "suspension_haircut": 0.20,
                },
                "execution": {"valor_mapping": {"buy": "buy_valor", "sell": "sell_valor"}},
            }
        },
        "hrp": {"cov_days": 126, "linkage": "single", "cluster_distance": 0.4},
        "backtest": {"cash_tax_rate": 0.175, "initial_capital": 100.0, "warmup_days": 21},
        "project": {"timezone": "Europe/Istanbul"},
        "store": {"path": "data/janus.duckdb"},
        "reporting": {"orders_dir": "reports"},
        "paper": {"min_hold_days": 21},
    }


def _populate_store(path, monkeypatch):
    cfg = _cfg()
    st = Store(path)
    nav = make_panel(n_funds=6, days=400, seed=11)
    nav.index = pd.bdate_range(end="2026-09-24", periods=len(nav))
    # Para Piyasası fonlarını ekle (F4, F5) sepet slotu için
    nav["PP0"] = 1.0 * (1 + nav["F0"].pct_change().clip(-0.001, 0.001)).cumprod().fillna(1.0)
    nav["PP1"] = 1.0 * (1 + nav["F1"].pct_change().clip(-0.001, 0.001)).cumprod().fillna(1.0)
    fm = pd.DataFrame(
        {
            "fund_code": list(nav.columns),
            "umbrella_type": ["Hisse"] * 6 + ["Para Piyasası"] * 2,
            "withholding_rate": 0.175,
            "tefas_status": "İşlem Görüyor",
            "founder": ["K1", "K1", "K2", "K2", "K3", "K3", "K1", "K2"],
            "founder_code": ["K1", "K1", "K2", "K2", "K3", "K3", "CASH1", "CASH2"],
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
            "fund_class": "YAT",
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
    return st, cfg, nav, fm


@pytest.fixture()
def paper_env(tmp_path, monkeypatch):
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    st, cfg, nav, fm = _populate_store(tmp_path / "janus.duckdb", monkeypatch)
    prepared = prepare(st, cfg)
    feats = build_features_for_test(prepared, cfg)
    preds = run_walkforward(feats, model_kwargs=MK)
    cal = calibrate_predictions(preds, feats, miscoverage_target=0.2, gamma=0.05)
    date = str(pd.to_datetime(cal["decision_at"]).max().date())
    return {
        "st": st,
        "cfg": cfg,
        "nav": nav,
        "feats": feats,
        "preds": preds,
        "cal": cal,
        "date": date,
        "tmp_path": tmp_path,
    }


def build_features_for_test(prepared, cfg):
    from janus.features.fund_features import build_features
    from janus.features.market import market_features

    nav = prepared["nav"]
    fm = prepared["fund_master"]
    cash_returns = prepared["cash_returns"]
    mf = _make_macro(nav.index)
    eq_cols = fm[fm["umbrella_type"].fillna("").str.contains("Hisse", case=False)]["fund_code"].tolist()
    mkt = market_features(nav, prepared["equity_index"], eq_cols)
    return build_features(nav, fm, cash_returns, mf, mkt, cfg=cfg, asof=None)


class FakeTelegram:
    def __init__(self):
        self.sent: list[str] = []

    def __call__(self, text):
        self.sent.append(text)
        return True


def test_target_weights_no_proof_all_cash(paper_env):
    selection = pd.DataFrame(
        {
            "fund_code": ["F0", "F1"],
            "p_value": [0.5, 0.6],
            "selected_q20": [False, False],
            "lower": [0.01, 0.02],
            "q50": [0.0, 0.0],
        }
    )
    w = target_weights_from_selection(
        selection, paper_env["nav"], paper_env["st"].latest_fund_master(), paper_env["cfg"]
    )
    assert w.shape == (1,)
    assert "CASH_PROXY" in w.index
    assert abs(w["CASH_PROXY"] - 1.0) < 1e-9


def test_target_weights_proof_respects_ceiling(paper_env):
    selection = pd.DataFrame(
        {
            "fund_code": ["F0", "F1", "F2", "F3"],
            "p_value": [0.01, 0.02, 0.03, 0.04],
            "selected_q20": [True, True, True, True],
            "lower": [0.05, 0.04, 0.03, 0.02],
            "q50": [0.0, 0.0, 0.0, 0.0],
        }
    )
    w = target_weights_from_selection(
        selection, paper_env["nav"], paper_env["st"].latest_fund_master(), paper_env["cfg"]
    )
    risky = w.drop("CASH_PROXY", errors="ignore")
    assert risky.sum() <= 0.30 + 1e-6
    assert (risky <= 0.25 + 1e-6).all()
    founder = paper_env["st"].latest_fund_master().set_index("fund_code").get("founder")
    grp = risky.groupby(founder).sum()
    assert (grp <= 0.30 + 1e-6).all()


def test_paper_ledger_identity(paper_env):
    st = paper_env["st"]
    cfg = paper_env["cfg"]
    paper = PaperLedger(st, cfg, capital=100.0)
    eq = paper.equity()
    assert abs(eq - 100.0) < 1e-3
    nav = paper.latest_nav()
    arr = nav.reindex(paper.meta.index).to_numpy(float).copy()
    arr[paper._slot_i] = paper._slot_price() or 1.0
    cash = float(paper.ledger.cash)
    rec = float(paper.ledger.receivable_total())
    risky = (
        float(paper.ledger.position_value(arr, None, 0.0).sum())
        - arr[paper._slot_i] * paper.ledger.units[paper._slot_i]
    )
    slot = arr[paper._slot_i] * paper.ledger.units[paper._slot_i]
    assert abs(cash + rec + risky + slot - eq) < 1e-6


def test_s5_6c_paper_init_starts_with_cash_and_no_b0_lots(paper_env):
    paper = PaperLedger(paper_env["st"], paper_env["cfg"], capital=100.0)

    b0_codes = paper._b0_codes
    assert len(b0_codes) == 2
    assert paper.ledger.units[paper._slot_i] == 0.0
    assert paper.ledger.lots[paper._slot_i] == []
    assert set(b0_codes) == {"PP0", "PP1"}
    for code in b0_codes:
        i = int(paper.meta.index.get_loc(code))
        assert paper.ledger.units[i] == 0.0
        assert paper.ledger.lots[i] == []
    assert paper.ledger.cash == pytest.approx(100.0)
    assert (
        paper_env["st"].con.execute("SELECT count(*) FROM paper_positions WHERE fund_code='CASH_PROXY'").fetchone()[0]
        == 0
    )
    assert (
        paper_env["st"].con.execute("SELECT count(*) FROM paper_lots WHERE fund_code='CASH_PROXY'").fetchone()[0] == 0
    )


def test_s5_6c_empty_bh_target_creates_equal_code_specific_b0_buys(paper_env):
    from janus.paper.core import orders_from_targets

    paper = PaperLedger(paper_env["st"], paper_env["cfg"], capital=100.0)
    orders = orders_from_targets(
        pd.Series({"CASH_PROXY": 1.0}), paper, paper_env["cfg"], "empty-bh", decision_idx=paper._latest_idx()
    )

    assert set(orders["fund_code"]) == set(paper._b0_codes)
    assert set(orders["action"]) == {"BUY"}
    assert orders["delta_w"].to_numpy() == pytest.approx([0.3, 0.3])
    assert 1.0 - orders["delta_w"].sum() == pytest.approx(0.4)


def test_s5_6c_propose_empty_bh_persists_b0_targets_and_code_orders(paper_env, monkeypatch):
    import json

    st, cfg, nav, root = paper_env["st"], paper_env["cfg"], paper_env["nav"], paper_env["tmp_path"]
    decision = pd.Timestamp(nav.index[-2])
    monkeypatch.setattr(cli, "ROOT", root)
    path = root / "data" / "predictions" / f"selection_{decision:%Y-%m-%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "decision_at": [decision, decision],
            "fund_code": ["F0", "F1"],
            "selected_q20": [False, False],
            "lower": [0.0, 0.0],
            "p_value": [0.5, 0.6],
        }
    ).to_parquet(path, index=False)

    result = paper_propose(st, cfg, str(decision.date()), root)

    assert result["status"] == "proposed"
    assert result["proof_count"] == 0
    assert "PP0" in result["telegram"] and "BUY" in result["telegram"]
    proposal = st.con.execute(
        "SELECT orders_json, target_weights_json FROM paper_proposals WHERE date=?", [decision.date()]
    ).fetchone()
    order_rows = pd.DataFrame(json.loads(proposal[0]))
    assert set(order_rows["fund_code"]) == {"PP0", "PP1"}
    assert set(order_rows["action"]) == {"BUY"}
    assert json.loads(proposal[1]).get("CASH_PROXY") == pytest.approx(1.0)
    assert st.con.execute("SELECT cash FROM paper_cash").fetchone()[0] == pytest.approx(100.0)
    assert st.con.execute("SELECT count(*) FROM paper_positions").fetchone()[0] == 0
    retry = paper_propose(st, cfg, str(decision.date()), root)
    assert retry["proposal_id"] == result["proposal_id"]
    assert st.con.execute("SELECT count(*) FROM paper_proposals WHERE date=?", [decision.date()]).fetchone()[0] == 1


def test_s5_6c_orders_do_not_spend_immature_receivables(paper_env):
    from janus.paper.core import orders_from_targets

    paper = PaperLedger(paper_env["st"], paper_env["cfg"], capital=100.0)
    paper.ledger.cash = 10.0
    paper.ledger.receivables[paper._latest_idx() + 1] = 90.0
    target = pd.Series({"F0": 0.30, "CASH_PROXY": 0.70})

    orders = orders_from_targets(target, paper, paper_env["cfg"], "cash-only", decision_idx=paper._latest_idx())

    risky = orders.loc[(orders["fund_code"] == "F0") & (orders["action"] == "BUY")]
    assert risky["delta_w"].sum() * paper.equity() <= 10.0 + 1e-8
    b0_buys = orders.loc[orders["fund_code"].isin(paper._b0_codes) & (orders["action"] == "BUY")]
    assert b0_buys.empty


def test_s5_6c_b0_lots_are_exempt_from_risky_min_hold(paper_env):
    from janus.paper.core import orders_from_targets

    paper = PaperLedger(paper_env["st"], paper_env["cfg"], capital=100.0)
    idx = paper._latest_idx()
    for code in paper._b0_codes:
        i = int(paper.meta.index.get_loc(code))
        assert paper.ledger.buy(i, 50.0, float(paper.latest_nav()[code]), idx) is not None
        paper.ledger.lots[i][0].available_idx = idx

    target = pd.Series({"PP0": 0.0, "PP1": 1.0})
    orders = orders_from_targets(target, paper, paper_env["cfg"], "b0-no-min-hold", decision_idx=idx)

    assert orders.loc[orders["fund_code"] == "PP0", "action"].item() == "SELL"


def test_s5_6c_init_keeps_decision_date_and_separate_pit_asofs(paper_env):
    nav = paper_env["nav"]
    decision_date = pd.Timestamp(nav.index[-1]) + pd.offsets.BDay(1)
    profile_date = decision_date
    master = paper_env["st"].latest_fund_master().copy()
    master["snapshot_date"] = profile_date.date()
    master["last_success_at"] = pd.Timestamp("2026-09-24 08:00:00")
    paper_env["st"].write_fund_master(master)
    paper = PaperLedger(paper_env["st"], paper_env["cfg"], capital=100.0, asof=decision_date)

    assert paper.nav_asof == pd.Timestamp(nav.index[-1])
    assert paper.meta_asof == profile_date
    assert paper.nav_asof < paper.meta_asof  # D-1 NAV + D profile is valid; independent PIT axes.


def test_f20_b0_reload_retains_old_member_and_targets_its_sale(paper_env):
    from janus.paper.core import orders_from_targets

    st, cfg = paper_env["st"], paper_env["cfg"]
    initial = PaperLedger(st, cfg, capital=100.0, asof=pd.Timestamp("2026-09-24"))
    assert set(initial._b0_codes) == {"PP0", "PP1"}
    for code in initial._b0_codes:
        i = int(initial.meta.index.get_loc(code))
        assert initial.ledger.buy(i, 50.0, float(initial.latest_nav()[code]), initial._latest_idx()) is not None
    initial._persist()

    master = st.latest_fund_master().copy()
    master.loc[master["fund_code"] == "PP0", "umbrella_type"] = "Hisse"
    master["snapshot_date"] = pd.Timestamp("2026-09-25").date()
    st.write_fund_master(master)

    reloaded = PaperLedger(st, cfg, asof=pd.Timestamp("2026-09-25"))
    assert reloaded._b0_codes == ("PP1",)
    assert "PP0" in reloaded.meta.index
    assert reloaded.position_components()[2] > 0

    old_i = int(reloaded.meta.index.get_loc("PP0"))
    reloaded.ledger.lots[old_i][0].bought_idx = 0
    reloaded.ledger.lots[old_i][0].available_idx = reloaded._latest_idx()
    reloaded._persist()
    orders = orders_from_targets(
        pd.Series({"CASH_PROXY": 1.0}), reloaded, cfg, "membership-change", decision_idx=reloaded._latest_idx()
    )

    assert (orders["fund_code"] == "PP0").any()
    assert orders.loc[orders["fund_code"] == "PP0", "action"].item() == "SELL"
    member_rows = st.con.execute(
        "SELECT membership_date, fund_code, founder_code, buy_valor, sell_valor "
        "FROM paper_b0_memberships ORDER BY membership_date, fund_code"
    ).fetchall()
    assert {(r[1], r[2], r[3], r[4]) for r in member_rows} == {
        ("PP0", "CASH1", 1, 2),
        ("PP1", "CASH2", 1, 2),
    }
    assert len(member_rows) == 3  # PP1 kimliği üyelik değişiminin iki tarihli kaydında korunur.


def test_f20_missing_recorded_b0_master_member_fails_without_db_state_change(paper_env):
    st, cfg = paper_env["st"], paper_env["cfg"]
    PaperLedger(st, cfg, capital=100.0)
    tables = ("paper_cash", "paper_positions", "paper_lots", "paper_receivables", "paper_b0_memberships")
    before = {table: st.con.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall() for table in tables}
    st.con.execute("DELETE FROM fund_master WHERE fund_code='PP0'")

    with pytest.raises(ValueError, match="PP0.*master|master.*PP0"):
        PaperLedger(st, cfg)

    after = {table: st.con.execute(f"SELECT * FROM {table} ORDER BY ALL").fetchall() for table in tables}
    assert after == before


def test_f20_recorded_b0_member_without_any_nav_fails_closed(paper_env):
    st, cfg = paper_env["st"], paper_env["cfg"]
    PaperLedger(st, cfg, capital=100.0)
    st.con.execute("DELETE FROM fund_nav WHERE fund_code='PP0'")

    with pytest.raises(ValueError, match="PP0.*NAV|NAV.*PP0"):
        PaperLedger(st, cfg)


def test_b0_membership_asof_does_not_write_future_master_snapshot(paper_env):
    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision_date = pd.Timestamp(nav.index[-2])
    future_master = st.latest_fund_master().copy()
    future_master["snapshot_date"] = (decision_date + pd.offsets.BDay(2)).date()
    st.write_fund_master(future_master)

    paper = PaperLedger(st, cfg, asof=decision_date)
    paper._persist()

    rows = st.con.execute(
        "SELECT membership_date, snapshot_date FROM paper_b0_memberships WHERE membership_date=?",
        [decision_date.date()],
    ).fetchall()
    assert rows
    assert all(snapshot <= membership <= decision_date.date() for membership, snapshot in rows)


def test_paper_default_decision_date_does_not_read_future_master_snapshot(paper_env):
    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    future_master = st.latest_fund_master().copy()
    future_master["snapshot_date"] = (pd.Timestamp(nav.index[-1]) + pd.offsets.BDay(2)).date()
    st.write_fund_master(future_master)

    decision_date = pd.Timestamp(nav.index[-1])
    paper = PaperLedger(st, cfg, asof=decision_date)

    assert paper.meta_asof <= decision_date
    assert st.con.execute("SELECT count(*) FROM paper_b0_memberships").fetchone()[0] == 2
    assert st.con.execute("SELECT count(*) FROM paper_cash").fetchone()[0] == 1


def test_b0_membership_change_is_recorded_for_each_asof_date(paper_env):
    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    first_date, second_date = pd.Timestamp(nav.index[-2]), pd.Timestamp(nav.index[-1])
    first = PaperLedger(st, cfg, asof=first_date)
    assert set(first._b0_codes) == {"PP0", "PP1"}

    second_master = st.latest_fund_master().copy()
    second_master.loc[second_master["fund_code"] == "PP0", "umbrella_type"] = "Hisse"
    second_master["snapshot_date"] = second_date.date()
    st.write_fund_master(second_master)
    second = PaperLedger(st, cfg, asof=second_date)
    assert second._b0_codes == ("PP1",)
    second._persist()

    rows = st.con.execute(
        "SELECT membership_date, snapshot_date, fund_code FROM paper_b0_memberships ORDER BY membership_date, fund_code"
    ).fetchall()
    first_members = {row[2] for row in rows if row[0] == first_date.date()}
    second_members = {row[2] for row in rows if row[0] == second_date.date()}
    assert first_members == {"PP0", "PP1"}
    assert second_members == {"PP1"}
    assert all(snapshot <= membership for membership, snapshot, _ in rows)


def test_f20_b0_missing_current_nav_marks_last_known_value(paper_env):
    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    initial = PaperLedger(st, cfg, capital=100.0)
    fill_idx = initial._date_idx(nav.index[-1])
    assert fill_idx is not None
    for code in initial._b0_codes:
        i = int(initial.meta.index.get_loc(code))
        assert initial.ledger.buy(i, 50.0, float(initial.latest_nav()[code]), fill_idx) is not None
        initial.ledger.lots[i][0].available_idx = fill_idx
        st.con.execute("UPDATE fund_master SET sell_valor=0 WHERE fund_code=?", [code])
    initial._persist()
    fill_d = nav.index[-1]
    st.con.execute("UPDATE paper_equity SET date=? WHERE portfolio_name='live'", [fill_d.date()])
    expected_marks = {code: float(nav.loc[nav.index[-2], code]) for code in ("PP0", "PP1")}
    for code in ("PP0", "PP1"):
        st.con.execute("UPDATE fund_nav SET price=NULL WHERE fund_code=? AND date=?", [code, fill_d.date()])
    marked = PaperLedger(st, cfg)
    assert {code: float(marked.latest_nav()[code]) for code in expected_marks} == pytest.approx(expected_marks)
    assert marked.equity() > 0

    assert marked.position_components()[2] > 0.0


def test_f20_paper_snapshot_values_real_b0_members_as_slot(paper_env):
    paper = PaperLedger(paper_env["st"], paper_env["cfg"], capital=100.0)
    risky_i = int(paper.meta.index.get_loc("F0"))
    paper.ledger.cash = 10.0
    assert paper.ledger.buy(risky_i, 10.0, float(paper.latest_nav()["F0"]), paper._latest_idx()) is not None
    date = pd.Timestamp(paper.ledger.dates[-1])

    paper.snapshot_equity(date)

    row = (
        paper_env["st"]
        .con.execute(
            "SELECT equity, cash, risky_value, slot_value FROM paper_equity WHERE portfolio_name='live' AND date=?",
            [date.date()],
        )
        .fetchone()
    )
    nav = paper.latest_nav()
    expected_slot = sum(
        float(nav[code]) * paper.ledger.units[paper.meta.index.get_loc(code)] for code in paper._b0_codes
    )
    expected_risky = float(nav["F0"]) * paper.ledger.units[risky_i]
    assert row[1] == pytest.approx(0.0)
    assert row[2] == pytest.approx(expected_risky)
    assert row[3] == pytest.approx(expected_slot)
    assert row[0] == pytest.approx(row[1] + expected_risky + expected_slot)


def test_f20_slot_target_is_distributed_to_real_b0_members(paper_env):
    from janus.paper.core import orders_from_targets

    paper = PaperLedger(paper_env["st"], paper_env["cfg"], capital=100.0)
    for code in paper._b0_codes:
        i = int(paper.meta.index.get_loc(code))
        paper.ledger.units[i] = 0.0
        paper.ledger.lots[i].clear()
    paper.ledger.cash = 100.0

    orders = orders_from_targets(
        pd.Series({"CASH_PROXY": 1.0}), paper, paper_env["cfg"], "f20", decision_idx=paper._latest_idx()
    )

    assert set(orders["fund_code"]) == set(paper._b0_codes)
    assert set(orders["action"]) == {"BUY"}
    assert orders["delta_w"].to_numpy() == pytest.approx([0.3, 0.3])
    assert 1.0 - orders["delta_w"].sum() == pytest.approx(0.4)


def test_s5_6c2_combined_b0_and_risky_founder_target_is_capped(paper_env):
    from janus.paper.core import orders_from_targets

    paper = PaperLedger(paper_env["st"], paper_env["cfg"], capital=100.0)
    codes = ["F0", *paper._b0_codes]
    paper._fund_master.loc[paper._fund_master["fund_code"].isin(codes), "founder_code"] = "SHARED"
    target = pd.Series({code: 0.20 for code in codes})

    orders = orders_from_targets(target, paper, paper_env["cfg"], "combined-founder", decision_idx=paper._latest_idx())

    assert set(orders["fund_code"]) == set(codes)
    assert orders["delta_w"].sum() <= 0.30 + 1e-9
    assert (orders["delta_w"] > 0).sum() == 3
    assert 1.0 - orders["delta_w"].sum() >= 0.70 - 1e-9


@pytest.mark.parametrize("locked_by", ["can_sell_false", "min_hold"])
def test_s5_6c2_locked_risky_lot_uses_founder_capacity_in_ledger_orders(paper_env, locked_by):
    from janus.paper.core import orders_from_targets

    paper = PaperLedger(paper_env["st"], paper_env["cfg"], capital=100.0)
    codes = ["F0", *paper._b0_codes]
    paper._fund_master.loc[paper._fund_master["fund_code"].isin(codes), "founder_code"] = "SHARED"
    risky_i = int(paper.meta.index.get_loc("F0"))
    if locked_by == "can_sell_false":
        paper.meta.can_sell[risky_i] = False
    price = float(paper.latest_nav()["F0"])
    assert paper.ledger.buy(risky_i, 25.0, price, paper._latest_idx()) is not None
    paper._persist()

    target = pd.Series({"F0": 0.10, **{code: 0.15 for code in paper._b0_codes}})
    orders = orders_from_targets(target, paper, paper_env["cfg"], "locked-founder", decision_idx=paper._latest_idx())

    assert "F0" not in set(orders.loc[orders["action"] == "SELL", "fund_code"])
    b0_buys = orders.loc[orders["fund_code"].isin(paper._b0_codes) & orders["action"].eq("BUY")]
    assert b0_buys["delta_w"].sum() <= 0.05 + 1e-8
    assert (
        b0_buys["delta_w"].sum()
        + float(paper.ledger.weights(paper.latest_nav().reindex(paper.meta.index).to_numpy(float), None, 0.0)[risky_i])
        <= 0.30 + 1e-8
    )


def test_f20_legacy_cash_proxy_lot_fails_closed_without_reinitializing(paper_env):
    from datetime import datetime

    from janus.paper.core import _ensure_paper_tables

    st, cfg = paper_env["st"], paper_env["cfg"]
    _ensure_paper_tables(st)
    st.con.execute("DELETE FROM paper_cash")
    st.con.execute("DELETE FROM paper_positions")
    st.con.execute("DELETE FROM paper_lots")
    st.con.execute("INSERT INTO paper_positions VALUES (?, ?, ?, ?)", [datetime.now(), "CASH_PROXY", 100.0, 1.0])
    st.con.execute(
        """INSERT INTO paper_lots
        (updated_at, fund_code, units, cost_per_unit, bought_date, tax_rate, available_date, bought_idx, available_idx)
        VALUES (?, 'CASH_PROXY', 100.0, 1.0, ?, 0.175, ?, 0, 1)""",
        [datetime.now(), paper_env["nav"].index[0].date(), paper_env["nav"].index[1].date()],
    )

    with pytest.raises(ValueError, match="Eski CASH_PROXY"):
        PaperLedger(st, cfg, capital=100.0)

    assert st.con.execute("SELECT units FROM paper_positions WHERE fund_code='CASH_PROXY'").fetchone() == (100.0,)
    assert st.con.execute("SELECT count(*) FROM paper_lots WHERE fund_code='CASH_PROXY'").fetchone() == (1,)


def test_paper_propose_no_orders_when_source_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    st, cfg, nav, fm = _populate_store(tmp_path / "janus.duckdb", monkeypatch)
    cfg["legs"]["tefas"]["universe"]["source_stale_days"] = 5
    # NAV'ı çok geriye çek
    old_nav = nav.iloc[:10]
    old_long = old_nav.reset_index().melt(id_vars="index", var_name="fund_code", value_name="price")
    old_long = old_long.rename(columns={"index": "date"})
    old_long["published_at"] = pd.Timestamp("2026-09-24")
    # nav tablosunu yeniden yaz
    st.con.execute("DELETE FROM fund_nav")
    st.upsert_nav(old_long)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    res = paper_propose(st, cfg, "2026-09-24", root=tmp_path)
    assert res["status"] == "hold"
    assert res["n_orders"] == 0


def test_paper_propose_idempotent(paper_env, monkeypatch):
    st, cfg, nav, tmp_path = paper_env["st"], paper_env["cfg"], paper_env["nav"], paper_env["tmp_path"]
    d = nav.index[-1]
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    selection_path = tmp_path / "data" / "predictions" / f"selection_{d:%Y-%m-%d}.parquet"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"decision_at": [d], "fund_code": ["F0"], "selected_q20": [False], "lower": [0.0], "p_value": [1.0]}
    ).to_parquet(selection_path, index=False)
    paper = PaperLedger(st, cfg, capital=100.0, asof=d)
    st.con.execute("DELETE FROM paper_equity WHERE portfolio_name='live' AND date > ?", [d.date()])
    paper.snapshot_equity(d, portfolio_name="live")

    r1 = paper_propose(st, cfg, str(d.date()), root=tmp_path)
    r2 = paper_propose(st, cfg, str(d.date()), root=tmp_path)
    assert r1["status"] == "proposed", r1
    assert r2["status"] == "proposed"
    assert r1["proposal_id"] == r2["proposal_id"]
    assert r1["n_orders"] == r2["n_orders"]
    assert st.con.execute("SELECT count(*) FROM paper_proposals WHERE date = ?", [d.date()]).fetchone()[0] == 1


def test_paper_produce_orders_with_proof(paper_env, monkeypatch):
    st, cfg, nav, tmp_path = paper_env["st"], paper_env["cfg"], paper_env["nav"], paper_env["tmp_path"]
    d = nav.index[-2]
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    selection_path = tmp_path / "data" / "predictions" / f"selection_{d:%Y-%m-%d}.parquet"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"decision_at": [d], "fund_code": ["F0"], "selected_q20": [True], "lower": [0.05], "p_value": [0.01]}
    ).to_parquet(selection_path, index=False)
    paper = PaperLedger(st, cfg, capital=100.0, asof=d)
    st.con.execute("DELETE FROM paper_equity WHERE portfolio_name='live' AND date > ?", [d.date()])
    paper.snapshot_equity(d, portfolio_name="live")

    result = paper_propose(st, cfg, str(d.date()), root=tmp_path)
    assert result["status"] == "proposed"
    assert result["proof_count"] == 1
    assert result["n_orders"] >= 0
    assert (tmp_path / "reports" / f"orders_{d:%Y-%m-%d}.md").exists()
    assert (tmp_path / "reports" / f"orders_{d:%Y-%m-%d}.csv").exists()
    if result["n_orders"] > 0:
        assert result["risky_weight"] <= 0.30 + 1e-6


# S5-2/3 testleri


def _select_and_propose(paper_env, monkeypatch, date=None):
    st, cfg, tmp_path = paper_env["st"], paper_env["cfg"], paper_env["tmp_path"]
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    d = pd.Timestamp(date or _pick_fill_date(paper_env))
    selection_path = tmp_path / "data" / "predictions" / f"selection_{d:%Y-%m-%d}.parquet"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"decision_at": [d], "fund_code": ["F0"], "selected_q20": [True], "lower": [0.05], "p_value": [0.01]}
    ).to_parquet(selection_path, index=False)
    paper = PaperLedger(st, cfg, capital=100.0, asof=d)
    st.con.execute("DELETE FROM paper_equity WHERE portfolio_name='live' AND date > ?", [d.date()])
    paper.snapshot_equity(d, portfolio_name="live")
    return paper_propose(st, cfg, str(d.date()), root=tmp_path)


def _pick_fill_date(paper_env):
    from janus.paper.fill import _next_business_day

    nav = paper_env["nav"]
    d = pd.Timestamp(paper_env["date"])
    for _ in range(10):
        if _next_business_day(d, paper_env["cfg"]) in nav.index:
            return str(d.date())
        d -= pd.Timedelta(days=1)
    return paper_env["date"]


def test_paper_fill_idempotent_and_expires(paper_env, monkeypatch):
    from janus.paper.fill import paper_fill

    st = paper_env["st"]
    cfg = paper_env["cfg"]
    fill_date = _pick_fill_date(paper_env)
    st.con.execute("UPDATE fund_master SET sell_valor=0 WHERE umbrella_type='Para Piyasası'")
    res = _select_and_propose(paper_env, monkeypatch, date=fill_date)
    assert res["status"] == "proposed"
    pid = res["proposal_id"]
    # auto fill ile D+1 fiyatından işle
    r1 = paper_fill(st, cfg, pid, auto=True)
    assert r1["status"] == "filled"
    r2 = paper_fill(st, cfg, pid, auto=True)
    assert r2["status"] == "filled"
    assert r1.get("n_fills", 0) == r2.get("n_fills", 0)
    # expires: manuel modda 12:00 sonrası varsayımını test için status='proposed' ve eski expires_at
    st.con.execute(
        "UPDATE paper_proposals SET status='proposed', expires_at=? WHERE proposal_id=?",
        [pd.Timestamp("2000-01-01"), pid],
    )
    r3 = paper_fill(st, cfg, pid)
    assert r3["status"] == "expired"


def test_paper_fill_auto_follows_backtest_rules(paper_env, monkeypatch):
    from janus.paper.fill import paper_fill_auto

    st = paper_env["st"]
    cfg = paper_env["cfg"]
    fill_date = _pick_fill_date(paper_env)
    st.con.execute("UPDATE fund_master SET sell_valor=0 WHERE umbrella_type='Para Piyasası'")
    res = _select_and_propose(paper_env, monkeypatch, date=fill_date)
    assert res["status"] == "proposed"
    auto = paper_fill_auto(st, cfg, decision_date=fill_date)
    assert auto["status"] == "filled"
    fills = st.con.execute("SELECT * FROM paper_fills WHERE proposal_id=?", [res["proposal_id"]]).df()
    for _, row in fills.iterrows():
        assert row["side"] in ("BUY", "SELL")
        if row["side"] == "BUY":
            assert row["gross"] > 0
            # fee + tax >= 0 (vergi satışta gerçekleşir; fee her işlemde 0 olabilir)
        if row["side"] == "SELL":
            assert row["units"] > 0


def test_paper_fill_auto_uses_previous_configured_business_day_after_holiday(paper_env, monkeypatch):
    from datetime import datetime

    import janus.paper.fill as fill_module
    from janus.paper.fill import paper_fill_auto

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision = pd.Timestamp("2026-09-25")  # Friday
    fill_date = pd.Timestamp("2026-09-29")  # Tuesday; Monday is a configured holiday
    cfg["calendar"] = {"holidays": ["2026-09-28"]}
    master = st.latest_fund_master().copy()
    master["snapshot_date"] = decision.date()
    master["last_success_at"] = decision + pd.Timedelta(hours=8)
    st.write_fund_master(master)

    # Add only synthetic Friday and Tuesday labels; Monday is absent from the NAV session axis.
    last_prices = nav.iloc[-1]
    rows = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": date,
                    "fund_code": last_prices.index.astype(str),
                    "price": last_prices.to_numpy(float) * multiplier,
                    "published_at": date + pd.Timedelta(hours=9),
                }
            )
            for date, multiplier in ((decision, 1.01), (fill_date, 1.02))
        ],
        ignore_index=True,
    )
    st.upsert_nav(rows)

    paper_on_decision = PaperLedger(st, cfg, capital=100.0, asof=decision)
    assert paper_on_decision.meta_asof <= decision
    proposal_id = f"{decision:%Y%m%d}-01"
    orders = [
        {"fund_code": code, "action": "BUY", "delta_w": 0.5, "target_w": 0.5} for code in paper_on_decision._b0_codes
    ]
    assert orders and {order["action"] for order in orders} == {"BUY"}
    assert {order["fund_code"] for order in orders} == set(paper_on_decision._b0_codes)
    _insert_proposal(st, proposal_id, decision, orders)

    observed_timezones = []

    class TuesdayDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            now = datetime(2026, 9, 29, 12, 0)
            if tz is not None:
                observed_timezones.append(tz.key)
                return now.replace(tzinfo=tz)
            return now

    monkeypatch.setattr(fill_module, "datetime", TuesdayDateTime)
    result = paper_fill_auto(st, cfg)

    assert observed_timezones == ["Europe/Istanbul", "Europe/Istanbul"]
    assert result["status"] == "partially_filled", result
    assert result["proposal_id"] == proposal_id
    assert result["fill_date"] == str(fill_date.date())
    fills = st.con.execute(
        "SELECT fund_code, side, fill_date FROM paper_fills WHERE proposal_id=? ORDER BY fund_code",
        [proposal_id],
    ).fetchall()
    assert fills == [(code, "BUY", fill_date.date()) for code in sorted(paper_on_decision._b0_codes)]
    filled_ledger = PaperLedger(st, cfg, asof=fill_date)
    fill_idx = filled_ledger._date_idx(fill_date)
    assert fill_idx is not None
    for code in paper_on_decision._b0_codes:
        i = int(filled_ledger.meta.index.get_loc(code))
        lot = filled_ledger.ledger.lots[i][0]
        assert lot.bought_idx == fill_idx
        assert lot.available_idx == fill_idx + int(filled_ledger.meta.buy_valor[i])
    assert filled_ledger.ledger.receivable_total() == pytest.approx(0.0)

    retry = paper_fill_auto(st, cfg)
    assert retry["status"] == "partially_filled"
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [proposal_id]).fetchone()[0] == len(
        orders
    )


def test_paper_reconcile_identity(paper_env, monkeypatch):
    from janus.paper.fill import paper_fill
    from janus.paper.reconcile import paper_reconcile

    st = paper_env["st"]
    cfg = paper_env["cfg"]
    fill_date = _pick_fill_date(paper_env)
    res = _select_and_propose(paper_env, monkeypatch, date=fill_date)
    paper_fill(st, cfg, res["proposal_id"], auto=True)
    rec = paper_reconcile(st, cfg, str(paper_env["nav"].index[-1].date()), out_dir=paper_env["tmp_path"] / "reports")
    assert rec["identity_ok"] is True
    assert rec["liquidation_value"] <= rec["equity"] + 1e-6
    assert (paper_env["tmp_path"] / "reports" / f"reconcile_{paper_env['nav'].index[-1].date()}.md").exists()


def test_paper_shadow_b0_equals_cash_basket_return(paper_env, monkeypatch):
    from janus.paper.shadows import run_shadows, snapshot_shadow_equity

    st = paper_env["st"]
    cfg = paper_env["cfg"]
    monkeypatch.setattr(cli, "ROOT", paper_env["tmp_path"])
    # selection dosyaları olmadan da B0 dönüyor
    equities = run_shadows(st, cfg, names=["B0_cash"])
    assert "B0_cash" in equities
    assert equities["B0_cash"]["status"] == "ok"
    assert not equities["B0_cash"]["equity"].empty
    # B0 son değer, para piyasası sepetinin bileşik getirisine eşit olmalı
    nav = st.nav_wide()
    cash_codes = [c for c in nav.columns if c in ("PP0", "PP1")]
    basket = nav[cash_codes].pct_change().mean(axis=1).add(1).cumprod()
    basket = basket.reindex(equities["B0_cash"]["equity"].index)
    expected = 100.0 * basket.iloc[-1]
    assert abs(equities["B0_cash"]["equity"].iloc[-1] - expected) < 1e-3
    summary = snapshot_shadow_equity(st, equities, pd.Timestamp(nav.index.max()))
    assert summary["status"] == "ok"
    rows = st.con.execute("SELECT count(*) FROM paper_equity WHERE portfolio_name='B0_cash'").fetchone()
    assert rows[0] >= 1
    assert st.con.execute(
        "SELECT cash, receivables, risky_value, slot_value FROM paper_equity WHERE portfolio_name='B0_cash' "
        "ORDER BY date DESC LIMIT 1"
    ).fetchone() == (None, None, None, None)


# S5-6a regression tests: assert persisted ledger state, not only return values.
def _insert_proposal(store, proposal_id, decision_date, orders):
    from datetime import datetime, timedelta

    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(store)
    d = pd.Timestamp(decision_date).date()
    store.con.execute(
        """INSERT INTO paper_proposals
        (proposal_id, date, created_at, expires_at, status, orders_json, evidence_count, evidence_codes, target_weights_json)
        VALUES (?, ?, ?, ?, ?, ?, 0, '[]', '{}')""",
        [
            proposal_id,
            d,
            datetime.now(),
            datetime.now() + timedelta(days=1),
            "proposed",
            pd.DataFrame(orders).to_json(orient="records"),
        ],
    )


def test_s5_6c2_fill_rechecks_founder_capacity_at_d_plus_one(paper_env):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision_d, fill_d = pd.Timestamp(nav.index[-2]), pd.Timestamp(nav.index[-1])
    st.con.execute("UPDATE fund_master SET founder_code='SHARED' WHERE fund_code IN ('F0', 'F1')")
    paper = PaperLedger(st, cfg, capital=100.0, asof=decision_d)
    f0_i = int(paper.meta.index.get_loc("F0"))
    assert paper.ledger.buy(f0_i, 25.0, float(paper._nav.loc[decision_d, "F0"]), paper._date_idx(decision_d))
    paper._persist()
    st.upsert_nav(
        pd.DataFrame(
            {
                "fund_code": ["F0"],
                "date": [fill_d],
                "price": [float(nav.loc[fill_d, "F0"]) * 1.5],
                "published_at": [fill_d],
            }
        )
    )
    proposal_id = f"{decision_d:%Y%m%d}-d1-cap"
    _insert_proposal(
        st,
        proposal_id,
        decision_d,
        [{"fund_code": "F1", "action": "BUY", "delta_w": 0.10, "target_w": 0.10}],
    )

    paper_fill(st, cfg, proposal_id, auto=True, price_date=str(fill_d.date()))

    after = PaperLedger(st, cfg, asof=fill_d)
    f1_i = int(after.meta.index.get_loc("F1"))
    assert after.ledger.units[f1_i] == pytest.approx(0.0)
    assert (
        st.con.execute("SELECT reason FROM paper_order_results WHERE proposal_id=?", [proposal_id]).fetchone()[0]
        == "D+1 gerçekleşebilir ortak PYŞ kapasitesi yok"
    )


@pytest.mark.parametrize("sell_valor", [0, 1])
def test_fill_does_not_implicitly_sell_b0_to_fund_risky_buy(paper_env, sell_valor):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    fill_d = nav.index[-1]
    decision_d = nav.index[-2]
    pid = f"{decision_d:%Y%m%d}-01"
    _insert_proposal(
        st,
        pid,
        decision_d,
        [
            {
                "fund_code": "F0",
                "action": "BUY",
                "delta_w": 0.30,
                "target_w": 0.30,
            }
        ],
    )
    before = PaperLedger(st, cfg, asof=fill_d)
    fill_idx = before._date_idx(fill_d)
    assert fill_idx is not None
    for code in before._b0_codes:
        i = int(before.meta.index.get_loc(code))
        assert before.ledger.buy(i, 50.0, float(before._nav.loc[fill_d, code]), fill_idx) is not None
        before.ledger.lots[i][0].available_idx = fill_idx
        before.meta.sell_valor[i] = sell_valor
        st.con.execute("UPDATE fund_master SET sell_valor=? WHERE fund_code=?", [sell_valor, code])
    before._persist()
    b0_units_before = sum(before.ledger.units[before.meta.index.get_loc(code)] for code in before._b0_codes)

    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_d.date()))

    after = PaperLedger(st, cfg)
    assert result["status"] == "rejected"
    assert after.ledger.units[after.meta.index.get_loc("F0")] == 0.0
    assert after.ledger.receivable_total() == pytest.approx(0.0)
    assert sum(after.ledger.units[after.meta.index.get_loc(code)] for code in after._b0_codes) == pytest.approx(
        b0_units_before
    )
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == 0
    assert st.con.execute("SELECT status FROM paper_proposals WHERE proposal_id=?", [pid]).fetchone()[0] == "rejected"


def test_f20_t1_b0_sale_is_pending_until_receivable_settles(paper_env):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    fill_d = nav.index[-1]
    decision_d = nav.index[-2]
    pid = f"{decision_d:%Y%m%d}-t1"
    paper = PaperLedger(st, cfg, asof=fill_d)
    fill_idx = paper._date_idx(fill_d)
    assert fill_idx is not None
    for code in paper._b0_codes:
        i = int(paper.meta.index.get_loc(code))
        assert paper.ledger.buy(i, 50.0, float(paper.latest_nav()[code]), fill_idx) is not None
        paper.ledger.lots[i][0].available_idx = fill_idx
        paper.meta.sell_valor[i] = 1
        st.con.execute("UPDATE fund_master SET sell_valor=1 WHERE fund_code=?", [code])
    paper._persist()
    _insert_proposal(
        st,
        pid,
        decision_d,
        [{"fund_code": "F0", "action": "BUY", "delta_w": 0.30, "target_w": 0.30}],
    )

    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_d.date()))

    assert result["status"] == "rejected"
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == 0
    assert st.con.execute("SELECT count(*) FROM paper_receivables WHERE amount > 0").fetchone()[0] == 0
    assert st.con.execute("SELECT status FROM paper_proposals WHERE proposal_id=?", [pid]).fetchone()[0] == "rejected"


def test_fill_rejects_b0_fund_without_d_plus_one_profile_and_snapshots_unchanged_ledger(paper_env):
    from janus.paper.fill import paper_fill
    from janus.paper.reconcile import paper_reconcile

    st, cfg, nav, root = paper_env["st"], paper_env["cfg"], paper_env["nav"], paper_env["tmp_path"]
    decision_d, fill_d = map(pd.Timestamp, nav.index[-2:])
    pid = f"{decision_d:%Y%m%d}-unknown-valor"
    paper = PaperLedger(st, cfg, capital=100.0, asof=decision_d)
    assert paper._b0_codes
    pp0_i = int(paper.meta.index.get_loc("PP0"))
    prior_idx = len(paper.ledger.dates) - 30
    prior_date = pd.Timestamp(paper.ledger.dates[prior_idx])
    prior_price = float(paper._nav.loc[prior_date, "PP0"])
    assert paper.ledger.buy(pp0_i, 25.0, prior_price, prior_idx) is not None
    paper._persist()
    before_cash = paper.ledger.cash
    before_positions = st.con.execute("SELECT fund_code, units FROM paper_positions ORDER BY fund_code").fetchall()
    before_lots = st.con.execute("SELECT fund_code, units FROM paper_lots ORDER BY fund_code").fetchall()
    before_receivables = st.con.execute("SELECT settle_idx, amount FROM paper_receivables").fetchall()
    blocked_profiles = st.con.execute("SELECT * FROM fund_master WHERE fund_code IN ('PP0', 'PP1')").df()
    blocked_profiles["snapshot_date"] = fill_d.date()
    blocked_profiles["source_published_at"] = fill_d + pd.Timedelta(hours=8)
    blocked_profiles["last_success_at"] = fill_d - pd.Timedelta(days=8)
    blocked_profiles["sell_valor"] = None
    st.write_fund_master_rows(blocked_profiles)
    _insert_proposal(
        st,
        pid,
        decision_d,
        [{"fund_code": "PP0", "action": "BUY", "delta_w": 0.30, "target_w": 0.30}],
    )

    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_d.date()))

    assert result["status"] == "rejected"
    outcomes = st.con.execute(
        "SELECT fund_code, outcome, reason FROM paper_order_results WHERE proposal_id=?", [pid]
    ).fetchall()
    assert outcomes[0][0:2] == ("PP0", "rejected")
    assert "D+1 PIT profili" in outcomes[0][2]
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == 0
    assert st.con.execute("SELECT cash FROM paper_cash").fetchone()[0] == pytest.approx(before_cash)
    assert (
        st.con.execute("SELECT fund_code, units FROM paper_positions ORDER BY fund_code").fetchall() == before_positions
    )
    assert st.con.execute("SELECT fund_code, units FROM paper_lots ORDER BY fund_code").fetchall() == before_lots
    assert st.con.execute("SELECT settle_idx, amount FROM paper_receivables").fetchall() == before_receivables
    snapshot = st.con.execute(
        "SELECT equity, cash, receivables, risky_value, slot_value FROM paper_equity WHERE date=?",
        [fill_d.date()],
    ).fetchone()
    assert snapshot is not None
    assert snapshot[0] == pytest.approx(sum(snapshot[1:]))
    held_pp0 = st.con.execute("SELECT units FROM paper_positions WHERE fund_code='PP0'").fetchone()[0]
    assert snapshot[4] == pytest.approx(held_pp0 * float(nav.loc[fill_d, "PP0"]))
    assert st.con.execute("SELECT status FROM paper_proposals WHERE proposal_id=?", [pid]).fetchone()[0] == "rejected"
    restarted = PaperLedger(st, cfg, asof=fill_d)
    assert restarted.ledger.cash == pytest.approx(before_cash)
    assert restarted.ledger.units[restarted.meta.index.get_loc("PP0")] == pytest.approx(held_pp0)
    rec = paper_reconcile(st, cfg, str(fill_d.date()), out_dir=root / "test-reports")
    assert rec["identity_ok"] is True


def test_fill_rejects_ineligible_b0_order_but_fills_healthy_risky_order(paper_env):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision_d, fill_d = map(pd.Timestamp, nav.index[-2:])
    pid = f"{decision_d:%Y%m%d}-mixed-profile"
    PaperLedger(st, cfg, capital=100.0, asof=decision_d)
    blocked_profiles = st.con.execute("SELECT * FROM fund_master WHERE fund_code IN ('PP0', 'PP1')").df()
    blocked_profiles["snapshot_date"] = fill_d.date()
    blocked_profiles["source_published_at"] = fill_d + pd.Timedelta(hours=8)
    blocked_profiles["last_success_at"] = fill_d - pd.Timedelta(days=8)
    st.write_fund_master_rows(blocked_profiles)
    _insert_proposal(
        st,
        pid,
        decision_d,
        [
            {"fund_code": "PP0", "action": "BUY", "delta_w": 0.20, "target_w": 0.20},
            {"fund_code": "F0", "action": "BUY", "delta_w": 0.20, "target_w": 0.20},
        ],
    )

    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_d.date()))

    assert result["status"] == "partially_filled"
    outcomes = st.con.execute(
        "SELECT fund_code, outcome, reason FROM paper_order_results WHERE proposal_id=? ORDER BY order_index", [pid]
    ).fetchall()
    assert outcomes[0][0:2] == ("PP0", "rejected")
    assert "D+1 PIT profili" in outcomes[0][2]
    assert outcomes[1][0:2] == ("F0", "filled")
    assert st.con.execute("SELECT fund_code, side FROM paper_fills WHERE proposal_id=?", [pid]).fetchall() == [
        ("F0", "BUY")
    ]


def test_held_risky_fund_uses_last_asof_nav_and_stale_price_never_trades(paper_env):
    from janus.paper.core import orders_from_targets
    from janus.paper.fill import paper_fill
    from janus.paper.reconcile import paper_reconcile

    st, cfg, nav, root = paper_env["st"], paper_env["cfg"], paper_env["nav"], paper_env["tmp_path"]
    decision_d, fill_d = map(pd.Timestamp, nav.index[-2:])
    cfg["legs"]["tefas"]["universe"]["max_stale_days"] = 0
    paper = PaperLedger(st, cfg, capital=100.0, asof=fill_d)
    fund_i = int(paper.meta.index.get_loc("F0"))
    purchase_idx = len(paper.ledger.dates) - 30
    purchase_date = pd.Timestamp(paper.ledger.dates[purchase_idx])
    purchase_price = float(paper._nav.loc[purchase_date, "F0"])
    fill = paper.ledger.buy(fund_i, 25.0, purchase_price, purchase_idx)
    assert fill is not None
    paper._persist()
    last_valid_price = float(nav.loc[decision_d, "F0"])
    st.upsert_nav(
        pd.DataFrame(
            {
                "date": [fill_d],
                "fund_code": ["F0"],
                "price": [float("nan")],
                "published_at": [fill_d + pd.Timedelta(hours=8)],
            }
        )
    )
    st.con.execute("UPDATE fund_nav SET price=NULL WHERE fund_code='F0' AND date=?", [fill_d.date()])

    restarted = PaperLedger(st, cfg, asof=fill_d)
    expected_equity = restarted.ledger.cash + restarted.ledger.receivable_total() + fill.units * last_valid_price
    assert restarted.latest_nav()["F0"] == pytest.approx(last_valid_price)
    assert restarted.equity() == pytest.approx(expected_equity)
    before_units = restarted.ledger.units[fund_i]
    before_cash = restarted.ledger.cash
    for target in (pd.Series({"F0": 1.0}), pd.Series({"F0": 0.0})):
        orders = orders_from_targets(target, restarted, cfg, "stale-risk", decision_idx=restarted._latest_idx())
        assert "F0" not in set(orders.get("fund_code", pd.Series(dtype=str)))

    pid = f"{decision_d:%Y%m%d}-stale-risk-fill"
    _insert_proposal(
        st,
        pid,
        decision_d,
        [{"fund_code": "F0", "action": "BUY", "delta_w": 0.20, "target_w": 0.20}],
    )
    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_d.date()))
    assert result["status"] == "rejected"
    assert "data_stale" in result["order_results"].iloc[0]["reason"]
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == 0
    assert st.con.execute("SELECT cash FROM paper_cash").fetchone()[0] == pytest.approx(before_cash)
    assert st.con.execute("SELECT units FROM paper_positions WHERE fund_code='F0'").fetchone()[0] == pytest.approx(
        before_units
    )
    snapshot = st.con.execute(
        "SELECT equity, cash, receivables, risky_value, slot_value FROM paper_equity WHERE date=?",
        [fill_d.date()],
    ).fetchone()
    assert snapshot[0] == pytest.approx(sum(snapshot[1:]))
    assert snapshot[3] == pytest.approx(before_units * last_valid_price)
    rec = paper_reconcile(st, cfg, str(fill_d.date()), out_dir=root / "test-reports")
    assert rec["identity_ok"] is True
    st.con.execute("UPDATE fund_nav SET price=NULL WHERE fund_code='F0'")
    unpriced = PaperLedger(st, cfg, asof=fill_d)
    with pytest.raises(ValueError, match="as-of öncesi erişilebilir NAV yok"):
        unpriced.equity()


def test_s5_6c_cash_only_init_first_empty_bh_proposal_fills_b0_on_d_plus_one(paper_env, monkeypatch):
    import json

    from janus.paper.fill import paper_fill

    st, cfg, nav, root = paper_env["st"], paper_env["cfg"], paper_env["nav"], paper_env["tmp_path"]
    decision = pd.Timestamp(nav.index[-2])
    fill_date = pd.Timestamp(nav.index[-1])
    monkeypatch.setattr(cli, "ROOT", root)
    selection_path = root / "data" / "predictions" / f"selection_{decision:%Y-%m-%d}.parquet"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "decision_at": [decision],
            "fund_code": ["F0"],
            "selected_q20": [False],
            "lower": [0.0],
            "p_value": [0.5],
        }
    ).to_parquet(selection_path, index=False)

    initialized = PaperLedger(st, cfg, capital=100.0, asof=decision)
    assert initialized.ledger.cash == pytest.approx(100.0)
    assert not initialized.ledger.lots[initialized._slot_i]
    assert all(not initialized.ledger.lots[initialized.meta.index.get_loc(code)] for code in initialized._b0_codes)
    proposed = paper_propose(st, cfg, str(decision.date()), root)
    assert proposed["status"] == "proposed"
    proposal = st.con.execute(
        "SELECT orders_json FROM paper_proposals WHERE proposal_id=?", [proposed["proposal_id"]]
    ).fetchone()
    orders = pd.DataFrame(json.loads(proposal[0]))
    assert set(orders["action"]) == {"BUY"}
    assert set(orders["fund_code"]) == set(initialized._b0_codes)

    result = paper_fill(st, cfg, proposed["proposal_id"], auto=True, price_date=str(fill_date.date()))

    assert result["status"] == "filled"
    after = PaperLedger(st, cfg, asof=fill_date)
    assert after.ledger.cash == pytest.approx(40.0, abs=1e-8)
    assert all(after.ledger.units[after.meta.index.get_loc(code)] > 0 for code in after._b0_codes)
    for code in after._b0_codes:
        i = int(after.meta.index.get_loc(code))
        lot = after.ledger.lots[i][0]
        assert lot.bought_idx == after._date_idx(fill_date)
        assert lot.available_idx == lot.bought_idx + int(after.meta.buy_valor[i])
        assert lot.tax_rate == pytest.approx(after.ledger._lot_tax_rate(i, lot.bought_idx))
    fills = st.con.execute(
        "SELECT fund_code, side FROM paper_fills WHERE proposal_id=?", [proposed["proposal_id"]]
    ).fetchall()
    assert set(fills) == {(code, "BUY") for code in after._b0_codes}


def test_s5_6c_sale_proceeds_only_fund_a_new_next_day_proposal(paper_env):
    from janus.paper.core import orders_from_targets
    from janus.paper.fill import paper_fill, paper_fill_auto

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision, fill_date, next_decision = map(pd.Timestamp, nav.index[-3:])
    st.con.execute(
        "UPDATE fund_master SET sell_valor=1, snapshot_date=? WHERE umbrella_type='Para Piyasası'",
        [decision.date()],
    )
    paper = PaperLedger(st, cfg, capital=100.0, asof=decision)
    decision_idx = paper._date_idx(decision)
    assert decision_idx is not None
    purchase_idx = decision_idx - 22
    for code in paper._b0_codes:
        i = int(paper.meta.index.get_loc(code))
        purchase_date = pd.Timestamp(paper.ledger.dates[purchase_idx])
        assert paper.ledger.buy(i, 50.0, float(paper._nav.loc[purchase_date, code]), purchase_idx) is not None
    paper._persist()
    target = pd.Series({"F0": 0.30, "CASH_PROXY": 0.70})
    old_orders = orders_from_targets(target, paper, cfg, f"{decision:%Y%m%d}-01", decision_idx=decision_idx)
    assert set(old_orders["action"]) == {"SELL"}
    assert set(old_orders["fund_code"]).issubset(set(paper._b0_codes))
    old_id = f"{decision:%Y%m%d}-01"
    _insert_proposal(st, old_id, decision, old_orders.to_dict(orient="records"))

    first_fill = paper_fill(st, cfg, old_id, auto=True, price_date=str(fill_date.date()))

    assert first_fill["status"] == "filled"
    assert st.con.execute("SELECT sum(amount) FROM paper_receivables").fetchone()[0] > 0
    assert paper_fill_auto(st, cfg, decision_date=str(decision.date()))["status"] == "filled"
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [old_id]).fetchone()[0] == len(
        old_orders
    )

    next_paper = PaperLedger(st, cfg, asof=next_decision)
    next_idx = next_paper._date_idx(next_decision)
    assert next_idx is not None
    available = next_paper.ledger.cash + sum(
        amount for settle_idx, amount in next_paper.ledger.receivables.items() if settle_idx <= next_idx
    )
    next_orders = orders_from_targets(target, next_paper, cfg, f"{next_decision:%Y%m%d}-01", decision_idx=next_idx)
    risky = next_orders.loc[(next_orders["fund_code"] == "F0") & (next_orders["action"] == "BUY")]
    assert not risky.empty
    assert risky["delta_w"].sum() * next_paper.equity() <= available + 1e-8
    assert risky["delta_w"].sum() * next_paper.equity() <= 0.30 * next_paper.equity() + 1e-8


def test_s5_6c_fill_executes_only_proposed_orders_without_sweep(paper_env):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision, fill_date = pd.Timestamp(nav.index[-2]), pd.Timestamp(nav.index[-1])
    pid = f"{decision:%Y%m%d}-only-order"
    _insert_proposal(
        st,
        pid,
        decision,
        [{"fund_code": "F0", "action": "BUY", "delta_w": 0.10, "target_w": 0.10}],
    )

    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_date.date()))

    assert result["status"] == "filled"
    assert st.con.execute("SELECT fund_code, side FROM paper_fills WHERE proposal_id=?", [pid]).fetchall() == [
        ("F0", "BUY")
    ]
    after = PaperLedger(st, cfg, asof=fill_date)
    assert after.ledger.cash > 0
    assert all(after.ledger.units[after.meta.index.get_loc(code)] == 0 for code in after._b0_codes)
    assert after.equity() == pytest.approx(100.0, rel=1e-8)


def test_s5_6c_rejected_order_does_not_change_positions_or_count_as_filled(paper_env):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision, fill_date = pd.Timestamp(nav.index[-2]), pd.Timestamp(nav.index[-1])
    pid = f"{decision:%Y%m%d}-rejected-order"
    _insert_proposal(
        st,
        pid,
        decision,
        [{"fund_code": "F0", "action": "BUY", "delta_w": 0.10, "target_w": 0.10}],
    )
    st.con.execute("UPDATE fund_master SET tefas_status='Fon Alımına Kapalı' WHERE fund_code='F0'")
    before = PaperLedger(st, cfg, asof=fill_date)
    equity_before = before.equity()
    cash_before = before.ledger.cash

    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_date.date()))

    after = PaperLedger(st, cfg, asof=fill_date)
    assert result["status"] == "rejected"
    assert st.con.execute("SELECT status FROM paper_proposals WHERE proposal_id=?", [pid]).fetchone()[0] == "rejected"
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == 0
    assert after.ledger.cash == pytest.approx(cash_before)
    assert after.equity() == pytest.approx(equity_before)
    assert all(after.ledger.units[after.meta.index.get_loc(code)] == 0 for code in after._b0_codes)


def test_s5_6c_strict_b0_profile_freshness_uses_decision_day_not_snapshot_day(paper_env):
    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision = pd.Timestamp(nav.index[-1]) + pd.offsets.BDay(1)
    snapshot_date = decision - pd.offsets.BDay(1)
    master = st.latest_fund_master().copy()
    master["snapshot_date"] = snapshot_date.date()
    master["last_success_at"] = (
        pd.Timestamp(snapshot_date.date()) - pd.Timedelta(days=7) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    )
    st.write_fund_master(master)

    with pytest.raises(ValueError, match="B0 canlı/PIT"):
        PaperLedger(st, cfg, capital=100.0, asof=decision)

    assert st.con.execute("SELECT count(*) FROM paper_cash").fetchone()[0] == 0


def test_s5_6c_stale_b0_is_marked_but_neither_bought_nor_sold(paper_env):
    from janus.data.quality import expected_last_nav_date
    from janus.paper.core import orders_from_targets

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision = pd.Timestamp("2026-09-25")
    cfg["calendar"] = {"nav_publish_time": "10:00", "holidays": ["2026-09-24"]}
    last_row = nav.iloc[-1].copy()
    last_row["PP0"] = float("nan")
    extra = pd.DataFrame(
        {
            "date": decision,
            "fund_code": last_row.index.astype(str),
            "price": last_row.to_numpy(),
            "published_at": decision + pd.Timedelta(hours=10),
        }
    )
    st.upsert_nav(extra)
    st.con.execute("UPDATE fund_nav SET price=NULL WHERE fund_code='PP0' AND date >= '2026-09-22'")

    paper = PaperLedger(st, cfg, capital=100.0, asof=decision)
    expected_nav_date = expected_last_nav_date(pd.Timestamp("2026-09-25 10:00"), cfg)
    assert expected_nav_date == decision
    assert extra["published_at"].max() <= decision + pd.Timedelta(hours=10)
    assert paper._nav.index.max() <= decision
    assert paper.meta_asof <= decision
    assert set(paper._b0_codes) == {"PP0", "PP1"}  # uzun geçmişte PP0 kapsaması hâlâ ≥ %99.

    buy_orders = orders_from_targets(
        pd.Series({"CASH_PROXY": 1.0}), paper, cfg, "stale-buy", decision_idx=paper._date_idx(decision)
    )
    assert "PP0" not in set(buy_orders["fund_code"])
    assert buy_orders.loc[buy_orders["fund_code"] == "PP1", "action"].item() == "BUY"

    stale_i = int(paper.meta.index.get_loc("PP0"))
    stale_price = float(paper.latest_nav()["PP0"])
    prior_idx = paper._date_idx(pd.Timestamp("2026-09-21"))
    assert prior_idx is not None
    assert paper.ledger.buy(stale_i, 40.0, stale_price, prior_idx) is not None
    sell_orders = orders_from_targets(
        pd.Series({"PP0": 0.0, "PP1": 1.0}), paper, cfg, "stale-sell", decision_idx=paper._date_idx(decision)
    )

    assert "PP0" not in set(sell_orders["fund_code"])
    assert sell_orders.loc[sell_orders["fund_code"] == "PP1", "action"].item() == "BUY"
    assert paper.latest_nav()["PP0"] == pytest.approx(stale_price)  # official value uses last known NAV.
    assert paper.equity() > 0.0


@pytest.mark.parametrize(
    ("invalid_field", "invalid_value", "expected_reason"),
    [
        ("last_success_at", "stale", "stale_last_success_at"),
        ("buy_valor", None, "missing_execution_field"),
        ("source_published_at", "late", "no_pit_snapshot"),
        ("tefas_status", "bilinmeyen durum", "unknown_tefas_status"),
    ],
)
def test_pit_execution_blocks_are_fund_scoped_and_persisted(
    paper_env, monkeypatch, invalid_field, invalid_value, expected_reason
):
    import json

    st, cfg, nav, root = paper_env["st"], paper_env["cfg"], paper_env["nav"], paper_env["tmp_path"]
    decision = pd.Timestamp(nav.index[-2])
    profile = st.con.execute("SELECT * FROM fund_master WHERE fund_code='F0' ORDER BY snapshot_date DESC LIMIT 1").df()
    profile["snapshot_date"] = decision.date()
    profile["last_success_at"] = decision + pd.Timedelta(hours=8)
    profile["source_published_at"] = decision + pd.Timedelta(hours=8)
    profile["entry_fee"] = 0.0
    profile["exit_fee"] = 0.0
    profile["buy_valor"] = 1
    profile["sell_valor"] = 2
    if invalid_field == "last_success_at":
        profile[invalid_field] = decision - pd.Timedelta(days=8)
    elif invalid_field == "source_published_at":
        profile[invalid_field] = decision + pd.Timedelta(hours=9, minutes=16)
    else:
        profile[invalid_field] = invalid_value
    st.write_fund_master_rows(profile)

    selection_path = root / "data" / "predictions" / f"selection_{decision:%Y-%m-%d}.parquet"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "decision_at": [decision, decision],
            "fund_code": ["F0", "F1"],
            "selected_q20": [True, True],
            "lower": [0.01, 0.02],
            "p_value": [0.01, 0.02],
        }
    ).to_parquet(selection_path, index=False)

    result = paper_propose(st, cfg, str(decision.date()), root, out_dir=root / "test-reports")

    assert result["status"] == "proposed"
    assert any("F0" in warning and expected_reason in warning for warning in result["warnings"])
    rows = st.con.execute(
        "SELECT orders_json, execution_blocks_json FROM paper_proposals WHERE proposal_id=?",
        [result["proposal_id"]],
    ).fetchone()
    orders = pd.DataFrame(json.loads(rows[0]))
    assert "F0" not in set(orders["fund_code"])
    assert "F1" in set(orders["fund_code"])
    assert set(orders.loc[orders["fund_code"].isin(["PP0", "PP1"]), "action"]) == {"BUY"}
    blocks = json.loads(rows[1])
    blocked = [block for block in blocks if block["record_type"] == "block"]
    assert {block["fund_code"] for block in blocked} == {"F0"}
    assert expected_reason in blocked[0]["reason"]
    if expected_reason == "unknown_tefas_status":
        assert {block["side"] for block in blocked} == {"BUY", "SELL"}
        assert all(block["reason"] == expected_reason for block in blocked)
    assert any(
        block["record_type"] == "publication_assumption"
        and block["fund_code"] == "F1"
        and block["reason"] == "date_only_pit_assumption"
        for block in blocks
    )


def test_latest_pit_profile_published_after_cutoff_uses_prior_available_profile_for_buy(paper_env):
    from janus.paper.core import _profile_execution_reasons, _publication_assumptions, orders_from_targets

    st, cfg = paper_env["st"], paper_env["cfg"]
    decision = pd.Timestamp("2026-09-25")
    profile = st.con.execute("SELECT * FROM fund_master WHERE fund_code='F0' LIMIT 1").df()
    prior = profile.copy()
    prior["snapshot_date"] = decision - pd.Timedelta(days=1)
    prior["last_success_at"] = decision - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    prior["source_published_at"] = decision - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    same_day = profile.copy()
    same_day["snapshot_date"] = decision
    same_day["last_success_at"] = decision + pd.Timedelta(hours=8)
    same_day["source_published_at"] = decision + pd.Timedelta(hours=11)
    st.write_fund_master_rows(pd.concat([prior, same_day], ignore_index=True))
    profiles = st.con.execute("SELECT * FROM fund_master WHERE fund_code='F0'").df()

    assert _profile_execution_reasons(profiles, decision, codes=["F0"]) == {}
    assert _publication_assumptions(profiles, decision) == {"F0": "source_timestamp"}

    paper = PaperLedger(st, cfg, asof=decision)
    orders = orders_from_targets(
        pd.Series({"F0": 0.3, "CASH_PROXY": 0.7}),
        paper,
        cfg,
        "late-new-snapshot",
        decision_idx=paper._latest_idx(),
    )
    assert orders.loc[orders["fund_code"] == "F0", "action"].item() == "BUY"


@pytest.mark.parametrize("status", [None, "", "   ", "TEFAS status not recognized"])
def test_unknown_status_blocks_both_sides_even_when_trade_flags_are_true(status):
    from janus.paper.core import _profile_execution_reasons

    profile = pd.DataFrame(
        {
            "fund_code": ["SYN"],
            "snapshot_date": [pd.Timestamp("2026-09-25")],
            "last_success_at": [pd.Timestamp("2026-09-25 08:00")],
            "source_published_at": [pd.Timestamp("2026-09-25 08:00")],
            "buy_valor": [1],
            "sell_valor": [2],
            "entry_fee": [0.0],
            "exit_fee": [0.0],
            "tax_category": ["diger"],
            "can_buy": [True],
            "can_sell": [True],
            "tefas_status": [status],
        }
    )

    assert _profile_execution_reasons(profile, "2026-09-25", codes=["SYN"]) == {
        "SYN": {"BUY": "unknown_tefas_status", "SELL": "unknown_tefas_status"}
    }


def test_latest_unknown_status_does_not_fall_back_to_older_known_profile():
    from janus.paper.core import _profile_execution_reasons

    profiles = pd.DataFrame(
        {
            "fund_code": ["SYN", "SYN"],
            "snapshot_date": [pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-25")],
            "last_success_at": [pd.Timestamp("2026-09-24 08:00"), pd.Timestamp("2026-09-25 08:00")],
            "source_published_at": [pd.Timestamp("2026-09-24 08:00"), pd.Timestamp("2026-09-25 08:00")],
            "buy_valor": [1, 1],
            "sell_valor": [2, 2],
            "entry_fee": [0.0, 0.0],
            "exit_fee": [0.0, 0.0],
            "tax_category": ["diger", "diger"],
            "can_buy": [True, True],
            "can_sell": [True, True],
            "tefas_status": ["İşlem Görüyor", "  "],
        }
    )

    assert _profile_execution_reasons(profiles, "2026-09-25", codes=["SYN"]) == {
        "SYN": {"BUY": "unknown_tefas_status", "SELL": "unknown_tefas_status"}
    }


def test_known_alim_kapali_status_blocks_buy_but_permits_sell():
    from janus.paper.core import _profile_execution_reasons

    profile = pd.DataFrame(
        {
            "fund_code": ["SYN"],
            "snapshot_date": [pd.Timestamp("2026-09-25")],
            "last_success_at": [pd.Timestamp("2026-09-25 08:00")],
            "source_published_at": [pd.Timestamp("2026-09-25 08:00")],
            "buy_valor": [1],
            "sell_valor": [2],
            "entry_fee": [0.0],
            "exit_fee": [0.0],
            "tax_category": ["diger"],
            "can_buy": [True],
            "can_sell": [True],
            "tefas_status": ["Fon Alımına Kapalı, Fon Bozumuna Açık"],
        }
    )

    assert _profile_execution_reasons(profile, "2026-09-25", codes=["SYN"]) == {"SYN": {"BUY": "can_buy=False"}}


def test_latest_available_stale_pit_profile_blocks_without_prior_fallback(paper_env):
    from janus.paper.core import _profile_execution_reasons, _publication_assumptions, orders_from_targets

    st, cfg = paper_env["st"], paper_env["cfg"]
    decision = pd.Timestamp("2026-09-25")
    profile = st.con.execute("SELECT * FROM fund_master WHERE fund_code='F0' LIMIT 1").df()
    prior = profile.copy()
    prior["snapshot_date"] = decision - pd.Timedelta(days=1)
    prior["last_success_at"] = decision - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    prior["source_published_at"] = decision - pd.Timedelta(days=1) + pd.Timedelta(hours=8)
    same_day = profile.copy()
    same_day["snapshot_date"] = decision
    same_day["last_success_at"] = decision - pd.Timedelta(days=8) + pd.Timedelta(hours=8)
    same_day["source_published_at"] = decision + pd.Timedelta(hours=8)
    st.write_fund_master_rows(pd.concat([prior, same_day], ignore_index=True))
    profiles = st.con.execute("SELECT * FROM fund_master WHERE fund_code='F0'").df()

    assert _profile_execution_reasons(profiles, decision, codes=["F0"]) == {
        "F0": {"BUY": "stale_last_success_at", "SELL": "stale_last_success_at"}
    }
    assert _publication_assumptions(profiles, decision) == {"F0": "source_timestamp"}
    paper = PaperLedger(st, cfg, asof=decision)
    orders = orders_from_targets(
        pd.Series({"F0": 0.3, "CASH_PROXY": 0.7}),
        paper,
        cfg,
        "stale-new-snapshot",
        decision_idx=paper._latest_idx(),
    )
    assert "F0" not in set(orders["fund_code"])

    next_day = decision + pd.Timedelta(days=1)
    next_profile = profile.copy()
    next_profile["snapshot_date"] = next_day
    next_profile["last_success_at"] = next_day + pd.Timedelta(hours=8)
    next_profile["source_published_at"] = next_day + pd.Timedelta(hours=8)
    all_profiles = pd.concat([profiles, next_profile], ignore_index=True)
    assert _profile_execution_reasons(all_profiles, next_day, codes=["F0"]) == {}
    assert _publication_assumptions(all_profiles, next_day) == {"F0": "source_timestamp"}


@pytest.mark.parametrize("invalid_day", ["decision", "fill"])
def test_fill_rechecks_decision_and_fill_execution_profiles(paper_env, invalid_day):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision, fill_date = pd.Timestamp(nav.index[-2]), pd.Timestamp(nav.index[-1])
    for day, fee in ((decision, 0.0), (fill_date, 0.0)):
        profile = st.con.execute(
            "SELECT * FROM fund_master WHERE fund_code='F0' ORDER BY snapshot_date DESC LIMIT 1"
        ).df()
        profile["snapshot_date"] = day.date()
        profile["last_success_at"] = day + pd.Timedelta(hours=8)
        profile["source_published_at"] = day + pd.Timedelta(hours=8)
        profile["entry_fee"] = fee
        profile["exit_fee"] = 0.0
        profile["buy_valor"] = 1
        profile["sell_valor"] = 2
        if invalid_day == "decision" and day == decision:
            profile["tefas_status"] = "Tanımsız durum"
        elif invalid_day == "fill" and day == fill_date:
            profile["tefas_status"] = "Tanımsız durum"
        st.write_fund_master_rows(profile)
    pid = f"{decision:%Y%m%d}-forced-pit-{invalid_day}"
    _insert_proposal(
        st,
        pid,
        decision,
        [{"fund_code": "F0", "action": "BUY", "delta_w": 0.1, "target_w": 0.1}],
    )

    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_date.date()))

    row = st.con.execute("SELECT outcome, reason FROM paper_order_results WHERE proposal_id=?", [pid]).fetchone()
    assert result["status"] == "rejected"
    assert row[0] == "rejected"
    assert "unknown_tefas_status" in row[1]
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == 0


@pytest.mark.parametrize("invalid_day", ["decision", "fill"])
def test_fill_rejects_unknown_status_without_changing_cash_or_lots(paper_env, invalid_day):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision, fill_date = pd.Timestamp(nav.index[-2]), pd.Timestamp(nav.index[-1])
    PaperLedger(st, cfg, asof=decision)
    for day in (decision, fill_date):
        profile = st.con.execute("SELECT * FROM fund_master WHERE fund_code='F0' LIMIT 1").df()
        profile["snapshot_date"] = day.date()
        profile["last_success_at"] = day + pd.Timedelta(hours=8)
        profile["source_published_at"] = day + pd.Timedelta(hours=8)
        profile["tefas_status"] = "İşlem Görüyor"
        if day == (decision if invalid_day == "decision" else fill_date):
            profile["tefas_status"] = "Tanımsız durum"
        st.write_fund_master_rows(profile)

    pid = f"{decision:%Y%m%d}-unknown-status-{invalid_day}"
    _insert_proposal(
        st,
        pid,
        decision,
        [{"fund_code": "F0", "action": "BUY", "delta_w": 0.1, "target_w": 0.1}],
    )
    cash_before = st.con.execute("SELECT cash FROM paper_cash ORDER BY updated_at DESC LIMIT 1").fetchone()[0]
    units_before = st.con.execute("SELECT coalesce(sum(units), 0) FROM paper_lots WHERE fund_code='F0'").fetchone()[0]

    paper_fill(st, cfg, pid, auto=True, price_date=str(fill_date.date()))

    result = st.con.execute("SELECT outcome, reason FROM paper_order_results WHERE proposal_id=?", [pid]).fetchone()
    assert result[0] == "rejected"
    assert "unknown_tefas_status" in result[1]
    assert st.con.execute("SELECT cash FROM paper_cash ORDER BY updated_at DESC LIMIT 1").fetchone()[0] == cash_before
    assert (
        st.con.execute("SELECT coalesce(sum(units), 0) FROM paper_lots WHERE fund_code='F0'").fetchone()[0]
        == units_before
    )
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == 0


@pytest.mark.parametrize("invalid_day", ["decision", "fill"])
def test_unknown_status_keeps_held_lot_marked_but_blocks_forced_sell(paper_env, invalid_day):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision, fill_date = pd.Timestamp(nav.index[-2]), pd.Timestamp(nav.index[-1])
    paper = PaperLedger(st, cfg, asof=decision)
    code_i = int(paper.meta.index.get_loc("F0"))
    decision_idx = paper._date_idx(decision)
    assert decision_idx is not None
    purchase_idx = decision_idx - 5
    purchase_date = pd.Timestamp(paper.ledger.dates[purchase_idx])
    lot = paper.ledger.buy(code_i, 20.0, float(paper._nav.loc[purchase_date, "F0"]), purchase_idx)
    assert lot is not None
    paper.ledger.lots[code_i][-1].available_idx = decision_idx
    paper._persist()

    if invalid_day == "decision":
        st.con.execute("UPDATE fund_master SET tefas_status='Tanımsız durum' WHERE fund_code='F0'")
    else:
        profile = st.con.execute("SELECT * FROM fund_master WHERE fund_code='F0' LIMIT 1").df()
        profile["snapshot_date"] = fill_date.date()
        profile["last_success_at"] = fill_date + pd.Timedelta(hours=8)
        profile["source_published_at"] = fill_date + pd.Timedelta(hours=8)
        profile["tefas_status"] = "Tanımsız durum"
        st.write_fund_master_rows(profile)

    pid = f"{decision:%Y%m%d}-unknown-status-sell-{invalid_day}"
    _insert_proposal(
        st,
        pid,
        decision,
        [{"fund_code": "F0", "action": "SELL", "delta_w": -0.10, "target_w": 0.0}],
    )
    cash_before = paper.ledger.cash
    units_before = paper.ledger.units[code_i]

    paper_fill(st, cfg, pid, auto=True, price_date=str(fill_date.date()))

    outcome = st.con.execute("SELECT outcome, reason FROM paper_order_results WHERE proposal_id=?", [pid]).fetchone()
    reloaded = PaperLedger(st, cfg, asof=fill_date)
    reloaded_i = int(reloaded.meta.index.get_loc("F0"))
    marked_price = float(reloaded.latest_nav()["F0"])
    snapshot = st.con.execute(
        "SELECT equity, cash, receivables, risky_value, slot_value FROM paper_equity "
        "WHERE portfolio_name='live' AND date=?",
        [fill_date.date()],
    ).fetchone()

    assert outcome[0] == "rejected"
    assert "unknown_tefas_status" in outcome[1]
    assert reloaded.ledger.cash == pytest.approx(cash_before)
    assert reloaded.ledger.units[reloaded_i] == pytest.approx(units_before)
    assert marked_price == pytest.approx(float(nav.loc[fill_date, "F0"]))
    assert snapshot[0] == pytest.approx(sum(snapshot[1:]))
    assert snapshot[3] == pytest.approx(units_before * marked_price)
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == 0


def test_cash_proxy_evidence_count_is_bh_count_not_order_count():
    from janus.paper.core import format_orders_md

    orders = pd.DataFrame(
        [
            {
                "fund_code": "F0",
                "action": "SELL",
                "target_w": 0.0,
                "current_w": 0.1,
                "delta_w": -0.1,
                "buy_valor": 1,
                "sell_valor": 2,
            }
        ]
    )
    report = format_orders_md(orders, "id", "2026-09-25", [], proof_count=0)
    assert "kanıt:** yok (0 fon)" in report


def test_receivable_survives_restart_and_settles(paper_env):
    st, cfg = paper_env["st"], paper_env["cfg"]
    paper = PaperLedger(st, cfg)
    i = int(paper.meta.index.get_loc("F0"))
    slot_i = paper._slot_i
    paper.ledger.units[slot_i] = 0.0
    paper.ledger.lots[slot_i].clear()
    paper.ledger.cash = 50.0
    buy_idx = len(paper.ledger.dates) - 5
    buy_price = float(paper._nav.iloc[buy_idx]["F0"])
    assert paper.ledger.buy(i, 20.0, buy_price, buy_idx) is not None
    sell_idx = buy_idx + 1
    sale = paper.ledger.sell(i, paper.ledger.units[i], float(paper._nav.iloc[sell_idx]["F0"]), sell_idx)
    assert sale is not None
    settle_idx = sell_idx + int(paper.meta.sell_valor[i])
    expected_receivable = paper.ledger.receivables[settle_idx]
    paper._persist()

    restarted = PaperLedger(st, cfg)
    assert restarted.ledger.receivables == {settle_idx: pytest.approx(expected_receivable)}
    assert restarted.ledger.settle(settle_idx) == pytest.approx(expected_receivable)
    assert restarted.ledger.receivable_total() == 0
    assert restarted.ledger.cash >= expected_receivable


def test_future_available_lot_stays_locked_after_restart(paper_env):
    st, cfg = paper_env["st"], paper_env["cfg"]
    paper = PaperLedger(st, cfg)
    i = int(paper.meta.index.get_loc("F0"))
    idx = len(paper.ledger.dates) - 3
    paper.ledger.cash = 20.0
    lot = paper.ledger.buy(i, 20.0, float(paper.latest_nav()["F0"]), idx)
    assert lot is not None
    paper.ledger.lots[i][-1].available_idx = len(paper.ledger.dates) + 5
    paper._persist()

    restarted = PaperLedger(st, cfg)
    ri = int(restarted.meta.index.get_loc("F0"))
    assert restarted.ledger.sellable_units(ri, len(restarted.ledger.dates) - 1) == 0


def test_fill_respects_can_buy_and_does_not_mark_unfilled_proposal_filled(paper_env):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision_d, fill_d = nav.index[-2], nav.index[-1]
    pid = f"{decision_d:%Y%m%d}-01"
    _insert_proposal(
        st,
        pid,
        decision_d,
        [
            {
                "fund_code": "F0",
                "action": "BUY",
                "delta_w": 0.10,
                "target_w": 0.10,
            }
        ],
    )
    paper = PaperLedger(st, cfg, asof=fill_d)
    st.con.execute("UPDATE fund_master SET tefas_status='Fon Alımına Kapalı' WHERE fund_code='F0'")
    paper._persist()
    before = paper.ledger.cash

    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_d.date()))

    assert result["status"] != "filled"
    assert st.con.execute("SELECT status FROM paper_proposals WHERE proposal_id=?", [pid]).fetchone()[0] != "filled"
    assert (
        st.con.execute(
            "SELECT count(*) FROM paper_order_results WHERE proposal_id=? AND outcome='rejected'", [pid]
        ).fetchone()[0]
        == 1
    )
    assert PaperLedger(st, cfg).ledger.units[PaperLedger(st, cfg).meta.index.get_loc("F0")] == 0
    assert PaperLedger(st, cfg).ledger.cash >= before - 1e-8


def test_fill_rejects_price_date_other_than_decision_plus_one(paper_env):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision_d = nav.index[-3]
    pid = f"{decision_d:%Y%m%d}-01"
    _insert_proposal(
        st,
        pid,
        decision_d,
        [
            {
                "fund_code": "F0",
                "action": "BUY",
                "delta_w": 0.10,
                "target_w": 0.10,
            }
        ],
    )
    result = paper_fill(st, cfg, pid, auto=True, price_date=str(nav.index[-1].date()))
    assert result["status"] == "error"
    assert "D+1" in result["message"]


def test_historical_ledger_marks_only_asof_nav(paper_env):
    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    asof = nav.index[-2]  # PIT valör profili yalnızca 2026-09-22 sonrası bilinir.
    fm = st.latest_fund_master().copy()
    fm["snapshot_date"] = asof.date()
    st.write_fund_master(fm)
    paper = PaperLedger(st, cfg, asof=asof)
    baseline_nav = paper.latest_nav().copy()
    assert baseline_nav.name == asof
    assert pd.Timestamp(paper._fund_master["snapshot_date"].max()) <= asof
    future = st.nav_long().loc[lambda x: pd.to_datetime(x["date"]) > asof].copy()
    future["price"] = future["price"] * 1000.0
    future["published_at"] = pd.Timestamp("2026-09-25")
    st.upsert_nav(future)
    perturbed = PaperLedger(st, cfg, asof=asof)
    pd.testing.assert_series_equal(baseline_nav, perturbed.latest_nav())


def test_reconcile_persists_requested_date_and_detects_snapshot_mismatch(paper_env):
    from janus.paper.reconcile import paper_reconcile

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    d = nav.index[-1]
    paper = PaperLedger(st, cfg, asof=d)
    paper.snapshot_equity(d, portfolio_name="live")
    st.con.execute("UPDATE paper_equity SET equity=equity+5 WHERE portfolio_name='live' AND date=?", [d.date()])
    result = paper_reconcile(st, cfg, str(d.date()), out_dir=paper_env["tmp_path"] / "reports")
    row = st.con.execute("SELECT identity_ok FROM paper_reconcile WHERE date=?", [d.date()]).fetchone()
    assert result["identity_ok"] is False
    assert row == (False,)


def test_adr20_counts_matured_selected_fund_episodes(paper_env):
    from janus.paper.weekly import adr20_status

    st = paper_env["st"]
    dates = pd.bdate_range("2026-01-01", periods=4)
    for j, d in enumerate(dates[:3]):
        _insert_proposal(st, f"p{j}", d, [])
        st.con.execute(
            "UPDATE paper_proposals SET evidence_count=1, evidence_codes='[\"A\"]' WHERE proposal_id=?", [f"p{j}"]
        )
    features = pd.DataFrame(
        {
            "decision_at": [dates[0], dates[1], dates[2], dates[0], dates[1], dates[2]],
            "feature_asof": [dates[0] - pd.Timedelta(days=1)] * 3 + [dates[0] - pd.Timedelta(days=1)] * 3,
            "fund_code": ["A", "A", "A", "B", "B", "B"],
            "label_available_at": [dates[3]] * 3 + [dates[0]] * 3,
            "y": [1.0, 1.0, 1.0, -10.0, -10.0, -10.0],
        }
    )
    result = adr20_status(st, features, asof=dates[3])
    assert result["n_episodes"] == 1
    assert result["n_matured_episodes"] == 1
    assert result["matured_positive_rate"] == 1.0
    assert not result["adr21_ready"]


def test_proof_loss_before_21_business_days_holds_risky_lot(paper_env):
    from janus.paper.core import orders_from_targets

    paper = PaperLedger(paper_env["st"], paper_env["cfg"])
    idx = len(paper.ledger.dates) - 10
    i = int(paper.meta.index.get_loc("F0"))
    paper.ledger.cash = 100.0
    paper.ledger.buy(i, 30.0, float(paper.latest_nav()["F0"]), idx)
    weights = paper.ledger.weights(paper.latest_nav().reindex(paper.meta.index).to_numpy(float), None, 0.0)
    b0_weight = sum(weights[paper.meta.index.get_loc(code)] for code in paper._b0_codes)
    target = pd.Series({"CASH_PROXY": b0_weight})
    orders = orders_from_targets(target, paper, paper_env["cfg"], "p", decision_idx=len(paper.ledger.dates) - 1)
    assert not (
        (orders.get("fund_code", pd.Series(dtype=str)) == "F0") & (orders.get("action", pd.Series(dtype=str)) == "SELL")
    ).any()
    from janus.paper.core import format_orders_md

    report = format_orders_md(orders, "p", str(pd.Timestamp(paper.ledger.dates[-1]).date()), [], 0, ["F0"])
    assert "kanıt kaybı" in report and "nakit sepeti" not in report


def test_proof_loss_after_21_business_days_sells_to_cash(paper_env):
    from janus.paper.core import orders_from_targets

    paper = PaperLedger(paper_env["st"], paper_env["cfg"])
    idx = len(paper.ledger.dates) - 22
    i = int(paper.meta.index.get_loc("F0"))
    paper.ledger.cash = 100.0
    paper.ledger.buy(i, 30.0, float(paper.latest_nav()["F0"]), idx)
    target = pd.Series({"CASH_PROXY": 1.0})
    orders = orders_from_targets(target, paper, paper_env["cfg"], "p", decision_idx=len(paper.ledger.dates) - 1)
    assert ((orders["fund_code"] == "F0") & (orders["action"] == "SELL")).any()


@pytest.mark.parametrize(
    ("status", "remove_fill_nav", "side", "expected_reason"),
    [
        ("Fon Bozumuna Kapalı", False, "SELL", "can_sell=False"),
        ("Fon Alımına Kapalı, Fon Bozumuna Kapalı", False, "BUY", "askıda"),
        ("", True, "BUY", "data_stale"),
    ],
)
def test_fill_records_rejected_tradeability_outcomes(paper_env, status, remove_fill_nav, side, expected_reason):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision_d, fill_d = nav.index[-2], nav.index[-1]
    pid = f"{decision_d:%Y%m%d}-01"
    paper = PaperLedger(st, cfg, asof=fill_d)
    if side == "SELL":
        i = int(paper.meta.index.get_loc("F0"))
        paper.ledger.cash = 20.0
        paper.ledger.buy(i, 20.0, float(nav.loc[decision_d - pd.offsets.BDay(1), "F0"]), len(paper.ledger.dates) - 2)
        paper._persist()
    st.con.execute("UPDATE fund_master SET tefas_status=? WHERE fund_code='F0'", [status])
    if remove_fill_nav:
        st.con.execute("DELETE FROM fund_nav WHERE fund_code='F0' AND date=?", [fill_d.date()])
    _insert_proposal(
        st,
        pid,
        decision_d,
        [
            {
                "fund_code": "F0",
                "action": side,
                "delta_w": 0.10 if side == "BUY" else -0.10,
                "target_w": 0.10 if side == "BUY" else 0.0,
            }
        ],
    )
    result = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_d.date()))
    outcome = st.con.execute("SELECT outcome, reason FROM paper_order_results WHERE proposal_id=?", [pid]).fetchone()
    assert result["status"] == "rejected"
    assert outcome[0] == "rejected"
    assert expected_reason in outcome[1]
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == 0


def test_min_hold_dd_override_and_unsellable_guard(paper_env):
    from janus.paper.core import orders_from_targets

    paper = PaperLedger(paper_env["st"], paper_env["cfg"])
    idx = len(paper.ledger.dates) - 10
    i = int(paper.meta.index.get_loc("F0"))
    paper.ledger.cash = 100.0
    paper.ledger.buy(i, 30.0, float(paper.latest_nav()["F0"]), idx)
    target = pd.Series({"CASH_PROXY": 1.0})
    d_idx = len(paper.ledger.dates) - 1
    forced = orders_from_targets(target, paper, paper_env["cfg"], "p", decision_idx=d_idx, dd=0.13)
    assert ((forced["fund_code"] == "F0") & (forced["action"] == "SELL")).any()
    paper.meta.can_sell[i] = False
    blocked = orders_from_targets(target, paper, paper_env["cfg"], "p", decision_idx=d_idx, dd=0.13)
    assert not ((blocked["fund_code"] == "F0") & (blocked["action"] == "SELL")).any()


def test_partial_fill_keeps_success_and_failure_separate(paper_env):
    from janus.paper.fill import paper_fill

    st, cfg, nav = paper_env["st"], paper_env["cfg"], paper_env["nav"]
    decision_d, fill_d = nav.index[-2], nav.index[-1]
    pid = f"{decision_d:%Y%m%d}-01"
    _insert_proposal(
        st,
        pid,
        decision_d,
        [
            {"fund_code": "F0", "action": "BUY", "delta_w": 0.10, "target_w": 0.10},
            {"fund_code": "F1", "action": "BUY", "delta_w": 0.10, "target_w": 0.10},
        ],
    )
    st.con.execute("UPDATE fund_master SET tefas_status='Fon Alımına Kapalı' WHERE fund_code='F1'")
    paper = PaperLedger(st, cfg, asof=fill_d)
    fill_idx = paper._date_idx(fill_d)
    assert fill_idx is not None
    for code in paper._b0_codes:
        i = int(paper.meta.index.get_loc(code))
        assert paper.ledger.buy(i, 40.0, float(paper._nav.loc[fill_d, code]), fill_idx) is not None
        paper.ledger.lots[i][0].available_idx = fill_idx
        paper.meta.sell_valor[i] = 0
        st.con.execute("UPDATE fund_master SET sell_valor=0 WHERE fund_code=?", [code])
    paper._persist()
    first = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_d.date()))
    assert first["status"] == "partially_filled"
    assert st.con.execute(
        "SELECT outcome FROM paper_order_results WHERE proposal_id=? ORDER BY order_index", [pid]
    ).fetchall() == [("filled",), ("rejected",)]
    fills_before = st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0]
    after_first = PaperLedger(st, cfg).ledger.units[0]

    second = paper_fill(st, cfg, pid, auto=True, price_date=str(fill_d.date()))

    assert second["status"] == "partially_filled"
    assert st.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [pid]).fetchone()[0] == fills_before
    assert PaperLedger(st, cfg).ledger.units[0] == pytest.approx(after_first)
