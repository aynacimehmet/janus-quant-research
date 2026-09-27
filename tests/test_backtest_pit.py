"""S5-6c: execution metadata must be point-in-time and separate from model features."""

from dataclasses import replace

import duckdb
import numpy as np
import pandas as pd

import janus.backtest.engine as backtest_engine
from janus.backtest.costs import build_fund_meta, meta_for_synthetic
from janus.backtest.data import prepare
from janus.backtest.engine import BacktestConfig, run_backtest
from janus.backtest.ledger import Ledger
from janus.backtest.runner import run_chain


class _SyntheticStore:
    def __init__(self, profiles: pd.DataFrame, nav: pd.DataFrame):
        self.con = duckdb.connect(":memory:")
        self.con.register("profiles_df", profiles)
        self.con.execute("CREATE TABLE fund_master AS SELECT * FROM profiles_df")
        self.nav = nav
        self.snapshot_asof = None

    def latest_fund_master(self):
        return self.con.execute(
            "SELECT * EXCLUDE (rn) FROM (SELECT *, row_number() OVER (PARTITION BY fund_code ORDER BY snapshot_date DESC) rn FROM fund_master) WHERE rn=1"
        ).df()

    def fund_master_history(self):
        return self.con.execute("SELECT * FROM fund_master ORDER BY snapshot_date, fund_code").df()

    def nav_wide(self, start=None):
        return self.nav if start is None else self.nav.loc[self.nav.index >= pd.Timestamp(start)]


def _profiles() -> pd.DataFrame:
    rows = []
    for code in ["A", "B"]:
        rows.extend(
            [
                {
                    "snapshot_date": "2026-09-22",
                    "fund_code": code,
                    "umbrella_type": "Hisse",
                    "founder": "Ornek PYŞ",
                    "founder_code": "F",
                    "tax_category": "diger",
                    "withholding_rate": 0.175,
                    "buy_valor": 1,
                    "sell_valor": 2,
                    "entry_fee": 3.0,
                    "exit_fee": 2.0,
                    "tefas_status": "İşlem Görüyor",
                    "can_buy": True,
                    "can_sell": True,
                    "last_success_at": "2026-09-22 09:00",
                },
                {
                    "snapshot_date": "2026-09-24",
                    "fund_code": code,
                    "umbrella_type": "Hisse",
                    "founder": "Ornek PYŞ",
                    "founder_code": "F",
                    "tax_category": "diger",
                    "withholding_rate": 0.175,
                    "buy_valor": 3,
                    "sell_valor": 4,
                    "entry_fee": 2.0,
                    "exit_fee": 3.0,
                    "tefas_status": "İşlem Görmüyor",
                    "can_buy": False,
                    "can_sell": False,
                    "last_success_at": "2026-09-24 09:00",
                },
            ]
        )
    rows[-1]["last_success_at"] = "2026-09-15 09:00"  # B profile stale on 2026-09-24
    return pd.DataFrame(rows)


def test_prepare_builds_asof_execution_profiles_without_changing_features(cfg):
    cfg["legs"]["tefas"]["execution"]["fee_scale"] = 0.01
    dates = pd.to_datetime(["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24", "2026-09-25"])
    nav = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6]}, index=dates)
    store = _SyntheticStore(_profiles(), nav)
    result = prepare(store, cfg)
    by_date = result["execution_meta_by_date"]

    pre = by_date.xs(pd.Timestamp("2026-09-21"), level="decision_date")
    assert pre.loc["A", "buy_valor"] == 3
    assert pre.loc["A", "entry_fee"] == 0.0
    assert pre.loc["A", "exit_fee"] == 0.0
    assert bool(pre.loc["A", "can_buy"])
    assert pre.loc["A", "status"] == "A3 varsayımlı, PIT kanıtı değil"
    assert pre.loc["A", "execution_source"] == "A3_CURRENT_PROFILE"

    exact = by_date.xs(pd.Timestamp("2026-09-22"), level="decision_date")
    assert exact.loc["A", "buy_valor"] == 1
    assert exact.loc["A", "entry_fee"] == 0.0
    assert exact.loc["A", "exit_fee"] == 0.0
    assert exact.loc["A", "execution_source"] == "PIT"
    raw_snapshot = store.fund_master_history().query("fund_code == 'A' and snapshot_date == '2026-09-22'")
    assert raw_snapshot.iloc[0]["entry_fee"] == 3.0
    assert "fee_assumed_zero" not in by_date.columns

    perturbed = _profiles()
    later_rows = perturbed["snapshot_date"].eq("2026-09-24")
    perturbed.loc[later_rows, "buy_valor"] = 99
    perturbed.loc[later_rows, "can_buy"] = False
    changed = prepare(_SyntheticStore(perturbed, nav), cfg)["execution_meta_by_date"]
    pd.testing.assert_series_equal(
        by_date.loc[(pd.Timestamp("2026-09-22"), "A")],
        changed.loc[(pd.Timestamp("2026-09-22"), "A")],
    )

    # Snapshot on D+2 cannot affect D's profile; D=22 only selects its exact eligible snapshot.
    assert bool(exact.loc["A", "can_buy"]) and bool(exact.loc["A", "can_sell"])
    later = by_date.xs(pd.Timestamp("2026-09-24"), level="decision_date")
    assert not later.loc["A", "can_buy"]
    assert later.loc["B", "buy_reason"] == "stale_last_success_at"
    assert not later.loc["B", "can_buy"] and not later.loc["B", "can_sell"]

    assert "execution_meta_by_date" in result
    assert tuple(result["meta"].codes) == tuple(nav.columns)


def test_prepare_preserves_labels_and_feature_columns(cfg):
    from janus.features.fund_features import FEATURE_COLUMNS

    dates = pd.bdate_range("2026-09-01", periods=5)
    nav = pd.DataFrame({"A": 1.0}, index=dates)
    result = prepare(_SyntheticStore(_profiles(), nav), cfg)
    assert FEATURE_COLUMNS  # execution-only metadata is not added to the feature whitelist
    assert not any(column in FEATURE_COLUMNS for column in ("buy_valor", "can_buy", "entry_fee"))
    assert result["execution_meta_by_date"].index.names == ["decision_date", "fund_code"]


def test_execution_profile_seven_day_boundary_and_no_snapshot_fail_closed(cfg):
    profiles = _profiles()
    profiles.loc[profiles.fund_code == "B", "snapshot_date"] = "2026-09-25"
    profiles.loc[(profiles.fund_code == "A") & (profiles.snapshot_date == "2026-09-24"), "last_success_at"] = (
        "2026-09-17 09:15"
    )
    profiles.loc[profiles.snapshot_date == "2026-09-24", ["can_buy", "can_sell"]] = True
    profiles.loc[profiles.snapshot_date == "2026-09-24", "tefas_status"] = "İşlem Görüyor"
    dates = pd.to_datetime(["2026-09-24", "2026-09-25"])
    nav = pd.DataFrame({"A": [1, 2], "B": [2, 3]}, index=dates)
    result = prepare(_SyntheticStore(profiles, nav), cfg)["execution_meta_by_date"]

    d7 = result.xs(pd.Timestamp("2026-09-24"), level="decision_date")
    d8 = result.xs(pd.Timestamp("2026-09-25"), level="decision_date")
    assert bool(d7.loc["A", "can_buy"])
    assert d8.loc["A", "buy_reason"] == "stale_last_success_at"
    assert d7.loc["B", "buy_reason"] == "no_pit_snapshot"
    assert not d7.loc["B", "can_buy"] and not d7.loc["B", "can_sell"]


def test_intraday_execution_profiles_use_morning_cutoff_and_published_snapshot(cfg):
    profiles = pd.DataFrame(
        [
            {
                "snapshot_date": day,
                "source_published_at": published,
                "fund_code": "A",
                "umbrella_type": "Hisse",
                "founder": "Ornek PYŞ",
                "founder_code": "F",
                "tax_category": "diger",
                "buy_valor": valor,
                "sell_valor": valor,
                "entry_fee": 0.0,
                "exit_fee": 0.0,
                "tefas_status": "İşlem Görüyor",
                "can_buy": True,
                "can_sell": True,
                "last_success_at": success,
            }
            for day, published, success, valor in [
                ("2026-09-22", "2026-09-22 08:50", "2026-09-22 08:59", 1),
                ("2026-09-23", "2026-09-23 11:01", "2026-09-23 11:00", 9),
            ]
        ]
    )
    dates = pd.to_datetime(["2026-09-23", "2026-09-24"])
    nav = pd.DataFrame({"A": [1.0, 1.1]}, index=dates)

    result = prepare(_SyntheticStore(profiles, nav), cfg)["execution_meta_by_date"]
    d = result.loc[(pd.Timestamp("2026-09-23"), "A")]
    d1 = result.loc[(pd.Timestamp("2026-09-24"), "A")]

    assert d["buy_valor"] == 1
    assert d["buy_reason"] == "ok"
    assert d1["buy_valor"] == 9
    assert d1["buy_reason"] == "ok"
    assert d["available_from"] == pd.Timestamp("2026-09-22")
    assert d["published_at"] == pd.Timestamp("2026-09-22 08:50")
    assert d["publication_time_assumption"] == "source_timestamp"

    perturbed = profiles.copy()
    perturbed.loc[perturbed["snapshot_date"].eq("2026-09-23"), "buy_valor"] = 99
    changed = prepare(_SyntheticStore(perturbed, nav), cfg)["execution_meta_by_date"]
    pd.testing.assert_series_equal(
        result.loc[(pd.Timestamp("2026-09-23"), "A")],
        changed.loc[(pd.Timestamp("2026-09-23"), "A")],
    )

    no_publication_time = profiles.drop(columns="source_published_at").copy()
    no_publication_time.loc[no_publication_time["snapshot_date"].eq("2026-09-23"), "last_success_at"] = (
        "2026-09-23 11:00"
    )
    fallback = prepare(_SyntheticStore(no_publication_time, nav), cfg)["execution_meta_by_date"]
    assert fallback.loc[(pd.Timestamp("2026-09-23"), "A"), "buy_reason"] == "future_last_success_at"
    assert fallback.loc[(pd.Timestamp("2026-09-24"), "A"), "buy_reason"] == "ok"
    assert fallback.loc[(pd.Timestamp("2026-09-23"), "A"), "publication_time_assumption"] == (
        "date_only_pit_assumption"
    )


def test_prepare_defaults_missing_morning_cutoff_to_temporal_0915(cfg):
    profiles = pd.DataFrame(
        [
            {
                "snapshot_date": day,
                "source_published_at": published,
                "fund_code": "A",
                "umbrella_type": "Hisse",
                "founder": "Ornek PYŞ",
                "founder_code": "F",
                "tax_category": "diger",
                "buy_valor": valor,
                "sell_valor": valor,
                "entry_fee": 0.0,
                "exit_fee": 0.0,
                "tefas_status": "İşlem Görüyor",
                "can_buy": True,
                "can_sell": True,
                "last_success_at": success,
            }
            for day, published, success, valor in [
                ("2026-09-22", "2026-09-22 08:50", "2026-09-22 09:00", 1),
                ("2026-09-23", "2026-09-23 09:45", "2026-09-23 10:00", 9),
            ]
        ]
    )
    date = pd.Timestamp("2026-09-23")
    nav = pd.DataFrame({"A": [1.0]}, index=pd.DatetimeIndex([date]))

    cfg["project"] = {"timezone": "Europe/Istanbul"}
    after_cutoff = profiles.iloc[[1]].copy()
    after_cutoff.loc[:, "last_success_at"] = "2026-09-23 11:00"
    default = prepare(_SyntheticStore(after_cutoff, nav), cfg)["execution_meta_by_date"]
    assert default.loc[(date, "A"), "buy_reason"] == "no_pit_snapshot"
    assert not default.loc[(date, "A"), "can_buy"]

    cfg["project"]["runs"] = {"morning": "10:15"}
    configured = prepare(_SyntheticStore(profiles, nav), cfg)["execution_meta_by_date"]
    assert configured.loc[(date, "A"), "buy_valor"] == 9
    assert configured.loc[(date, "A"), "buy_reason"] == "ok"


class _SequenceStrategy:
    name = "sequence"

    def __call__(self, ctx):
        target = "A" if ctx.idx < 2 else "B"
        return pd.Series({code: float(code == target) for code in ctx.meta.codes}), 1.0


def _execution_panel(dates, codes, *, buy_valor=0, sell_valor=0, fee=0.0):
    index = pd.MultiIndex.from_product([dates, codes], names=["decision_date", "fund_code"])
    return pd.DataFrame(
        {
            "buy_valor": buy_valor,
            "sell_valor": sell_valor,
            "entry_fee": fee,
            "exit_fee": fee,
            "tax_category": "diger",
            "can_buy": True,
            "can_sell": True,
            "status": "İşlem Görüyor",
            "execution_source": "PIT",
            "source_snapshot_date": dates[0],
            "last_success_at": dates[0],
            "buy_reason": "ok",
            "sell_reason": "ok",
        },
        index=index,
    )


def test_engine_uses_decision_day_execution_meta_and_freezes_valor_tax_without_fee():
    dates = pd.bdate_range("2026-09-22", periods=7)
    nav = pd.DataFrame({"A": [1.0, 1.1, 2.0, 2.0, 2.0, 2.0, 2.0], "B": 1.0}, index=dates)
    execution = _execution_panel(dates, ["A", "B"], buy_valor=2, sell_valor=1, fee=0.02)
    execution.loc[(dates[2], "A"), "sell_valor"] = 2
    cfg = BacktestConfig(
        initial_capital=100,
        warmup_days=0,
        schedule="daily",
        drift_threshold=0.0,
        drift_threshold_taxable_sale=0.0,
    )

    result = run_backtest(
        nav,
        meta_for_synthetic(["A", "B"], tax_rate=0.20),
        _SequenceStrategy(),
        cfg,
        execution_meta_by_date=execution,
    )

    buys_a = result.fills.query("code == 'A' and side == 'BUY'")
    sells_a = result.fills.query("code == 'A' and side == 'SELL'")
    buys_b = result.fills.query("code == 'B' and side == 'BUY'")
    assert buys_a.iloc[0]["idx"] == 0
    assert sells_a.iloc[0]["idx"] >= buys_a.iloc[0]["idx"] + 2  # buy_valor is frozen into Lot.available_idx
    assert sells_a.iloc[0]["idx"] == 2
    assert buys_b.iloc[0]["idx"] == 4  # sale-day sell_valor=2 controls receivable settlement
    assert buys_a.iloc[0]["fee"] == 0.0
    assert sells_a.iloc[0]["fee"] == 0 and sells_a.iloc[0]["tax"] > 0
    assert result.weights_realized.loc[dates[3], "B"] == 0.0
    assert result.weights_realized.loc[dates[4], "B"] > 0.0

    stressed = run_backtest(
        nav,
        meta_for_synthetic(["A", "B"], tax_rate=0.20),
        _SequenceStrategy(),
        cfg,
        execution_meta_by_date=execution,
        allow_fee_stress=True,
    )
    assert stressed.fees_paid > 0.0  # Explicit hypothetical stress only, never a base TEFAS fee.


def test_engine_freezes_purchase_day_tax_category_and_settles_fifo_sale(monkeypatch):
    dates = pd.bdate_range("2026-09-22", periods=6)
    nav = pd.DataFrame({"A": [1.0, 2.0, 2.0, 2.0, 2.0, 2.0], "B": 1.0}, index=dates)
    execution = _execution_panel(dates, ["A", "B"])
    execution.loc[(dates[0], "A"), "tax_category"] = "borclanma_para_maden"
    execution.loc[(dates[1:], "A"), "tax_category"] = "pay_senedi_yogun"
    execution.loc[:, "buy_valor"] = 0
    execution.loc[:, "sell_valor"] = 0
    execution.loc[(dates[2], "A"), "sell_valor"] = 2
    meta = meta_for_synthetic(["A", "B"], buy_valor=0, sell_valor=0)
    meta = replace(
        meta,
        tax_schedule=(
            (
                pd.Timestamp("2020-01-01").date(),
                {"borclanma_para_maden": 0.10, "pay_senedi_yogun": 0.0, "default": 0.175},
            ),
        ),
    )
    captured = {}

    class RecordingLedger(Ledger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["ledger"] = self
            self.tax_rates_at_buy = []
            self.settlements = []

        def buy(self, i, amount, price, idx):
            fill = super().buy(i, amount, price, idx)
            if fill is not None:
                self.tax_rates_at_buy.append((idx, self.lots[i][-1].tax_rate))
            return fill

        def settle(self, idx):
            self.settlements.append((idx, dict(self.receivables)))
            return super().settle(idx)

    monkeypatch.setattr(backtest_engine, "Ledger", RecordingLedger)

    def strategy(ctx):
        if ctx.idx == 0:
            weights = {"A": 0.5, "B": 0.0}
        elif ctx.idx == 1:
            weights = {"A": 1.0, "B": 0.0}
        else:
            weights = {"A": 0.0, "B": 1.0}
        return pd.Series(weights), 1.0

    result = run_backtest(
        nav,
        meta,
        strategy,
        BacktestConfig(
            initial_capital=100,
            warmup_days=0,
            schedule="daily",
            drift_threshold=0.0,
            drift_threshold_taxable_sale=0.0,
        ),
        execution_meta_by_date=execution,
    )

    ledger = captured["ledger"]
    sale = result.fills.query("code == 'A' and side == 'SELL'").iloc[0]
    assert ledger.tax_rates_at_buy[:2] == [(0, 0.10), (1, 0.0)]
    assert sale["idx"] == 2 and np.isclose(sale["tax"], 5.0)
    assert (3, {4: 145.0}) in ledger.settlements
    assert (4, {4: 145.0}) in ledger.settlements
    assert ledger.receivables == {}
    assert ledger.units[0] == 0.0 and ledger.units[1] > 0.0


def test_engine_missing_daily_tax_category_blocks_only_that_fund_and_preserves_reason():
    dates = pd.bdate_range("2026-09-22", periods=4)
    nav = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0], "B": 1.0}, index=dates)
    execution = _execution_panel(dates, ["A", "B"])
    execution.loc[pd.IndexSlice[dates[1:], "A"], "tax_category"] = "  "
    execution.loc[pd.IndexSlice[dates[1:], "A"], "buy_reason"] = "missing_tax_category"
    execution.loc[pd.IndexSlice[dates[1:], "A"], "sell_reason"] = "missing_tax_category"
    cfg = BacktestConfig(initial_capital=100, warmup_days=0, schedule="daily", drift_threshold=0.0)

    result = run_backtest(
        nav,
        meta_for_synthetic(["A", "B"], buy_valor=0, sell_valor=0),
        lambda ctx: (
            pd.Series({"A": 0.25, "B": 0.25}) if ctx.idx == 0 else pd.Series({"A": 0.0, "B": 0.5}),
            1.0,
        ),
        cfg,
        execution_meta_by_date=execution,
    )

    assert ((result.fills["code"] == "A") & (result.fills["side"] == "BUY")).any()
    assert not ((result.fills["code"] == "A") & (result.fills["side"] == "SELL")).any()
    assert ((result.fills["code"] == "B") & (result.fills["side"] == "BUY")).any()
    blocked = result.events.query("event == 'execution_blocked' and code == 'A' and action == 'SELL'")
    assert not blocked.empty and blocked.iloc[0]["reason"] == "missing_tax_category"


def test_execution_meta_future_perturbation_does_not_change_prior_engine_fills():
    dates = pd.bdate_range("2026-09-22", periods=8)
    nav = pd.DataFrame({"A": np.linspace(1.0, 1.2, len(dates)), "B": np.linspace(1.0, 1.1, len(dates))}, index=dates)
    execution = _execution_panel(dates, ["A", "B"], buy_valor=1, sell_valor=1, fee=0.02)
    perturbed = execution.copy()
    future = dates[5:]
    perturbed.loc[pd.IndexSlice[future, :], "entry_fee"] = 0.15
    perturbed.loc[pd.IndexSlice[future, :], "buy_valor"] = 5
    perturbed.loc[pd.IndexSlice[future, :], "can_buy"] = False
    cfg = BacktestConfig(initial_capital=100, warmup_days=0, schedule="daily", drift_threshold=0.0)

    first = run_backtest(
        nav, meta_for_synthetic(["A", "B"]), _SequenceStrategy(), cfg, execution_meta_by_date=execution
    )
    second = run_backtest(
        nav, meta_for_synthetic(["A", "B"]), _SequenceStrategy(), cfg, execution_meta_by_date=perturbed
    )

    pd.testing.assert_frame_equal(
        first.fills[first.fills["idx"] < 5].reset_index(drop=True),
        second.fills[second.fills["idx"] < 5].reset_index(drop=True),
    )


def test_missing_profile_fails_closed_per_fund_and_keeps_last_nav_valuation():
    dates = pd.bdate_range("2026-09-22", periods=5)
    nav = pd.DataFrame({"A": [1.0, 1.1, np.nan, np.nan, np.nan], "B": 1.0}, index=dates)
    execution = _execution_panel(dates, ["A", "B"])
    execution.loc[pd.IndexSlice[dates[1:], "A"], "can_buy"] = False
    execution.loc[pd.IndexSlice[dates[1:], "A"], "can_sell"] = False
    execution.loc[pd.IndexSlice[dates[1:], "A"], "buy_reason"] = "stale_last_success_at"
    execution.loc[pd.IndexSlice[dates[1:], "A"], "sell_reason"] = "stale_last_success_at"
    cfg = BacktestConfig(initial_capital=100, warmup_days=0, schedule="daily", drift_threshold=0.0)

    result = run_backtest(
        nav,
        meta_for_synthetic(["A", "B"]),
        lambda ctx: (
            pd.Series({"A": 0.25, "B": 0.25}) if ctx.idx == 0 else pd.Series({"A": 0.0, "B": 0.5}),
            1.0,
        ),
        cfg,
        execution_meta_by_date=execution,
    )

    assert ((result.fills["code"] == "A") & (result.fills["side"] == "BUY")).any()
    assert not ((result.fills["code"] == "A") & (result.fills["side"] == "SELL")).any()
    assert ((result.fills["code"] == "B") & (result.fills["side"] == "BUY")).any()
    blocked = result.events.query("event == 'execution_blocked' and code == 'A' and action == 'SELL'")
    assert not blocked.empty and blocked.iloc[0]["reason"] == "stale_last_success_at"
    assert np.isfinite(result.equity.iloc[-1]) and result.equity.iloc[-1] > 0
    assert result.equity.iloc[-1] == result.equity.iloc[1]  # final value uses the last known A NAV


def test_pre_pit_a3_open_status_can_override_latest_closed_profile(cfg):
    dates = pd.bdate_range("2026-09-17", periods=5)
    profiles = _profiles()
    profiles["last_success_at"] = "2020-01-01 09:00"  # deliberately stale: pre-PIT A3 must ignore freshness
    profiles.loc[:, "tefas_status"] = "İşlem Görmüyor"
    profiles.loc[:, "can_buy"] = False
    profiles.loc[:, "can_sell"] = False
    nav = pd.DataFrame({"A": np.linspace(1.0, 1.1, len(dates)), "B": np.linspace(1.0, 1.05, len(dates))}, index=dates)
    d = prepare(_SyntheticStore(profiles, nav), cfg)
    result = run_backtest(
        d["nav"],
        d["meta"],
        lambda ctx: (
            pd.Series({code: float(code == ("A" if ctx.idx % 2 == 0 else "B")) for code in ctx.meta.codes}),
            1.0,
        ),
        BacktestConfig(initial_capital=100, warmup_days=0, schedule="daily", drift_threshold=0.0),
        execution_meta_by_date=d["execution_meta_by_date"],
    )

    assert "A3 varsayımlı, PIT kanıtı değil" in set(
        d["execution_meta_by_date"].xs(dates[0], level="decision_date")["status"]
    )
    assert ((result.fills["code"] == "A") & (result.fills["side"] == "BUY")).any()
    assert result.turnover > 0


def test_historical_b1_universe_ignores_current_closed_status_and_executes_pre_pit(cfg):
    dates = pd.bdate_range("2025-01-01", "2026-09-21")
    nav = pd.DataFrame({"A": 10 * 1.001 ** np.arange(len(dates))}, index=dates)
    profiles = pd.DataFrame(
        [
            {
                "snapshot_date": "2026-09-24",
                "fund_code": "A",
                "fund_class": "YAT",
                "umbrella_type": "Hisse Senedi Şemsiye Fonu",
                "category": "Hisse",
                "name": "Ornek PYŞ Hisse Fonu",
                "founder": "Ornek PYŞ",
                "founder_code": "ORNEK",
                "manager": "Ornek PYŞ",
                "first_nav_date": "2010-01-01",
                "info_ok": True,
                "hist_ok": True,
                "n_nav": len(dates),
                "tax_category": "diger",
                "withholding_rate": 0.0,
                "buy_valor": 0,
                "sell_valor": 0,
                "entry_fee": 0.0,
                "exit_fee": 0.0,
                "tefas_status": "İşlem Görmüyor",
                "can_buy": False,
                "can_sell": False,
                "last_success_at": "2026-09-24 08:00",
            }
        ]
    )
    data = prepare(_SyntheticStore(profiles, nav), cfg)
    assert bool(data["eligible"].loc["A"])
    pre = data["execution_meta_by_date"].xs(pd.Timestamp("2026-09-21"), level="decision_date")
    assert bool(pre.loc["A", "can_buy"])
    assert pre.loc["A", "status"] == "A3 varsayımlı, PIT kanıtı değil"

    cfg["backtest"]["warmup_days"] = 0
    cfg["backtest"]["schedule"] = "daily"
    cfg["legs"]["tefas"]["constraints"]["max_weight_per_fund"] = 1.0
    results = run_chain(data, cfg, top_n=1)
    b1 = next(result for result in results if result.name == "B1_topN_momentum")
    risk_fills = b1.fills.loc[b1.fills["code"].ne("CASH_PROXY")]
    assert not risk_fills.empty
    assert (risk_fills["side"] == "BUY").any()
    assert pd.to_numeric(risk_fills["units"], errors="coerce").gt(0).any()
    assert b1.weights_realized["A"].gt(0).any()
    assert np.isfinite(b1.equity.iloc[-1]) and b1.equity.iloc[-1] > 0
    assert b1.turnover > 0


def test_current_closed_status_still_blocks_b1_after_first_pit_date(cfg):
    dates = pd.bdate_range("2024-09-01", "2026-10-30")
    pit_start = pd.Timestamp("2026-09-22")
    post = dates >= pit_start
    nav = pd.DataFrame(
        {
            "A": 10 * 1.0005 ** np.arange(len(dates)),
            "B": 10 * np.where(post, 1.01 ** post.cumsum(), 1.0),
        },
        index=dates,
    )
    profiles = pd.DataFrame(
        [
            {
                "snapshot_date": day,
                "source_published_at": day + pd.Timedelta(hours=8),
                "fund_code": code,
                "fund_class": "YAT",
                "umbrella_type": "Hisse Senedi Şemsiye Fonu",
                "category": "Hisse",
                "name": f"Ornek PYŞ {code} Hisse Fonu",
                "founder": f"Ornek PYŞ {code}",
                "founder_code": code,
                "manager": f"Ornek PYŞ {code}",
                "first_nav_date": "2010-01-01",
                "info_ok": True,
                "hist_ok": True,
                "n_nav": len(dates),
                "tax_category": "diger",
                "withholding_rate": 0.0,
                "buy_valor": 0,
                "sell_valor": 0,
                "entry_fee": 0.0,
                "exit_fee": 0.0,
                "tefas_status": "İşlem Görmüyor",
                "can_buy": False,
                "can_sell": False,
                "last_success_at": day + pd.Timedelta(hours=8),
            }
            for day in dates[post]
            for code in ("A", "B")
        ]
    )
    data = prepare(_SyntheticStore(profiles, nav), cfg)
    cfg["backtest"]["warmup_days"] = 260
    cfg["backtest"]["schedule"] = "daily"
    cfg["legs"]["tefas"]["constraints"]["max_weight_per_fund"] = 1.0
    result = run_backtest(
        data["nav"],
        data["meta"],
        lambda ctx: (
            pd.Series({"A": float(ctx.date < pit_start), "B": float(ctx.date >= pit_start)}),
            1.0,
        ),
        BacktestConfig(initial_capital=100, warmup_days=260, schedule="daily", drift_threshold=0.0),
        execution_meta_by_date=data["execution_meta_by_date"],
    )

    post_events = result.events.loc[
        result.events["date"].ge(pit_start)
        & result.events["event"].eq("execution_blocked")
        & result.events["code"].eq("B")
        & result.events["action"].eq("BUY")
    ]
    assert not post_events.empty
    assert post_events["reason"].eq("status_buy_closed").all()
    post_pit_buys = result.fills.loc[
        result.fills["date"].ge(pit_start) & result.fills["code"].eq("B") & result.fills["side"].eq("BUY")
    ]
    assert post_pit_buys.empty


def test_post_pit_unknown_tefas_status_blocks_both_sides_per_fund(cfg):
    profiles = _profiles()
    profiles = profiles.loc[profiles["snapshot_date"].eq("2026-09-22")].copy()
    profiles["tefas_status"] = [None, "   "]
    profiles["can_buy"] = True
    profiles["can_sell"] = True
    profiles["last_success_at"] = "2026-09-22 09:00"
    dates = pd.to_datetime(["2026-09-22"])
    nav = pd.DataFrame({"A": [1.0], "B": [1.0]}, index=dates)

    result = prepare(_SyntheticStore(profiles, nav), cfg)["execution_meta_by_date"].xs(dates[0], level="decision_date")

    assert result.loc["A", "buy_reason"] == result.loc["A", "sell_reason"] == "unknown_tefas_status"
    assert result.loc["B", "buy_reason"] == result.loc["B", "sell_reason"] == "unknown_tefas_status"
    assert not result["can_buy"].any() and not result["can_sell"].any()


def test_post_pit_status_text_overrides_inconsistent_direction_flags(cfg):
    profiles = _profiles().query("snapshot_date == '2026-09-22'").copy()
    profiles["last_success_at"] = "2026-09-22 09:00"
    profiles["can_buy"] = True
    profiles["can_sell"] = pd.Series(True, index=profiles.index, dtype="boolean")
    profiles.loc[profiles.fund_code == "B", "can_buy"] = False
    profiles.loc[profiles.fund_code == "B", "can_sell"] = pd.NA
    profiles.loc[profiles.fund_code == "A", "tefas_status"] = "Fon Alımına Kapalı, Fon Bozumuna Açık"
    profiles.loc[profiles.fund_code == "B", "tefas_status"] = "Fon Alımına Açık, Fon Bozumuna Kapalı"
    date = pd.Timestamp("2026-09-22")
    nav = pd.DataFrame({"A": [1.0], "B": [1.0]}, index=[date])

    result = prepare(_SyntheticStore(profiles, nav), cfg)["execution_meta_by_date"].xs(date, level="decision_date")

    assert not result.loc["A", "can_buy"] and result.loc["A", "buy_reason"] == "status_buy_closed"
    assert result.loc["A", "can_sell"] and result.loc["A", "sell_reason"] == "ok"
    assert result.loc["B", "can_buy"] and result.loc["B", "buy_reason"] == "ok"
    assert not result.loc["B", "can_sell"] and result.loc["B", "sell_reason"] == "status_sell_closed"


def test_post_pit_source_fees_do_not_affect_execution_or_blocks(cfg):
    profiles = _profiles().query("fund_code == 'A' and snapshot_date == '2026-09-22'").copy()
    profiles["tefas_status"] = "İşlem Görüyor"
    profiles["can_buy"] = False
    profiles["can_sell"] = pd.Series(pd.NA, index=profiles.index, dtype="boolean")
    profiles["entry_fee"] = np.nan
    profiles["exit_fee"] = np.nan
    profiles["last_success_at"] = "2026-09-22 09:00"
    date = pd.Timestamp("2026-09-22")
    nav = pd.DataFrame({"A": [1.0]}, index=[date])
    data = prepare(_SyntheticStore(profiles, nav), cfg)
    row = data["execution_meta_by_date"].xs(date, level="decision_date").loc["A"]

    assert bool(row["can_buy"])
    assert bool(row["can_sell"])
    assert row["entry_fee"] == 0.0 and row["exit_fee"] == 0.0
    assert "fee_assumed_zero" not in row.index
    assert bool(row["status_flag_mismatch"])

    bad_fee_profile = profiles.copy()
    bad_fee_profile["entry_fee"] = "not-a-fee"
    bad_fee_profile["exit_fee"] = 99.0
    bad_fee = (
        prepare(_SyntheticStore(bad_fee_profile, nav), cfg)["execution_meta_by_date"]
        .xs(date, level="decision_date")
        .loc["A"]
    )
    assert bad_fee["buy_reason"] == "ok" and bad_fee["sell_reason"] == "ok"
    assert bad_fee["entry_fee"] == bad_fee["exit_fee"] == 0.0


def test_missing_pit_profile_does_not_create_fee_provenance(cfg):
    profiles = _profiles().query("fund_code == 'A' and snapshot_date == '2026-09-24'").copy()
    date = pd.Timestamp("2026-09-22")
    nav = pd.DataFrame({"A": [1.0]}, index=[date])

    row = (
        prepare(_SyntheticStore(profiles, nav), cfg)["execution_meta_by_date"].xs(date, level="decision_date").loc["A"]
    )

    assert row["buy_reason"] == "no_pit_snapshot"
    assert "fee_assumed_zero" not in row.index


def test_post_pit_freshness_uses_exact_morning_cutoff_age(cfg):
    profiles = _profiles().query("fund_code == 'A' and snapshot_date == '2026-09-22'").copy()
    profiles["tefas_status"] = "İşlem Görüyor"
    profiles["can_buy"] = True
    profiles["can_sell"] = True
    profiles["last_success_at"] = "2026-09-15 09:14"
    date = pd.Timestamp("2026-09-22")
    nav = pd.DataFrame({"A": [1.0]}, index=[date])

    result = prepare(_SyntheticStore(profiles, nav), cfg)["execution_meta_by_date"].xs(date, level="decision_date")

    assert result.loc["A", "buy_reason"] == "stale_last_success_at"
    assert not bool(result.loc["A", "can_buy"])


def test_pre_pit_fee_is_zero_without_a3_fee_provenance(cfg):
    profiles = _profiles()
    latest = profiles["snapshot_date"].eq("2026-09-24")
    profiles.loc[latest, ["entry_fee", "exit_fee"]] = np.nan
    date = pd.Timestamp("2026-09-21")
    nav = pd.DataFrame({"A": [1.0], "B": [1.0]}, index=[date])

    data = prepare(_SyntheticStore(profiles, nav), cfg)
    row = data["execution_meta_by_date"].xs(date, level="decision_date").loc["A"]

    assert row["execution_source"] == "A3_CURRENT_PROFILE"
    assert bool(row["can_buy"])
    assert row["entry_fee"] == 0 and row["exit_fee"] == 0
    assert "fee_assumed_zero" not in row.index


def test_post_pit_missing_tax_category_blocks_both_without_snapshot_fallback(cfg):
    profiles = _profiles().query("fund_code == 'A'").copy()
    profiles.loc[profiles.snapshot_date == "2026-09-24", "tax_category"] = "  "
    profiles.loc[:, "last_success_at"] = profiles["snapshot_date"] + " 09:00"
    date = pd.Timestamp("2026-09-24")
    nav = pd.DataFrame({"A": [1.0]}, index=[date])

    result = prepare(_SyntheticStore(profiles, nav), cfg)["execution_meta_by_date"].loc[(date, "A")]

    assert result["source_snapshot_date"] == date
    assert result["tax_category"].strip() == ""
    assert result["buy_reason"] == result["sell_reason"] == "missing_tax_category"
    assert not result["can_buy"] and not result["can_sell"]


def test_tefas_and_bes_meta_ignore_source_fees_and_us_order_cost_stays_150(cfg):
    source = pd.DataFrame(
        {
            "fund_code": ["FEE"],
            "entry_fee": [7.5],
            "exit_fee": [np.nan],
            "withholding_rate": [0.175],
            "tax_category": ["diger"],
            "tefas_status": ["İşlem Görüyor"],
        }
    )
    for leg in ("tefas", "bes"):
        meta = build_fund_meta(source, ["FEE"], cfg, leg=leg)
        assert meta.entry_fee.tolist() == [0.0]
        assert meta.exit_fee.tolist() == [0.0]
    assert source.loc[0, "entry_fee"] == 7.5 and pd.isna(source.loc[0, "exit_fee"])
    assert cfg["legs"]["us"]["costs"]["per_order_usd"] == 1.5
