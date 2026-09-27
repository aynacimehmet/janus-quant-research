"""0c-3 testleri: nakit vekili sepet slotu (P02), tarihe göre lot stopajı (P04), tasfiye değeri (§3)."""

import numpy as np
import pandas as pd
import pytest

from _panel import make_panel
from janus.backtest.costs import meta_for_synthetic
from janus.backtest.data import _execution_profiles_asof, cash_proxy_codes, cash_proxy_returns
from janus.backtest.engine import BacktestConfig, run_backtest
from janus.backtest.ledger import Ledger
from janus.strategies.baselines import CashOnly, TopNMomentum

SLOT = "CASH_PROXY"


def _fm(codes, umbrella, founder="POYRAZ PORTFÖY", n_nav=400, cat="borclanma_para_maden", wr=0.175):
    return pd.DataFrame(
        {
            "fund_code": codes,
            "umbrella_type": umbrella,
            "founder": [founder] * len(codes),
            "n_nav": [n_nav] * len(codes),
            "tax_category": [cat] * len(codes),
            "withholding_rate": [wr] * len(codes),
        }
    )


def test_cash_proxy_codes_selection(cfg):
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(0)
    nav = pd.DataFrame(
        rng.lognormal(0, 0.001, size=(300, 3)).cumprod(axis=0) * 10,
        index=idx,
        columns=["M1", "M2", "M3"],
    )
    nav.loc[idx[-20:], "M2"] = np.nan  # doluluk %93 → elenir
    fm = pd.DataFrame(
        {
            "fund_code": ["M1", "M2", "M3", "M4"],
            "umbrella_type": ["Para Piyasası Şemsiye Fonu"] * 3 + ["Hisse Senedi Şemsiye Fonu"],
            "founder_code": ["POY", "POY", "KUZ", "POY"],
            "founder": ["POYRAZ PORTFÖY", "POYRAZ PORTFÖY", "Kuzey Portföy", "POYRAZ PORTFÖY"],  # M3 kara listede
            "n_nav": [100, 500, 300, 300],
            "tax_category": ["borclanma_para_maden"] * 3 + ["pay_senedi_yogun"],
            "withholding_rate": [0.175] * 4,
        }
    )

    codes = cash_proxy_codes(nav, fm, cfg)
    assert codes == ["M1"]  # M2 doluluk, M3 kara liste, M4 şemsiye; M1 tek kalan


def test_s5_6c2_cash_proxy_allows_multiple_funds_per_founder(cfg):
    idx = pd.bdate_range("2024-01-01", periods=100)
    codes = ["A1", "A2", "A3", "B1", "C1", "D1", "E1"]
    nav = pd.DataFrame(10.0, index=idx, columns=codes)
    fm = pd.DataFrame(
        {
            "fund_code": codes,
            "umbrella_type": "Para Piyasası Şemsiye Fonu",
            "founder_code": ["A", "A", "A", "B", "C", "D", "E"],
            "founder": ["A Portföy"] * 3 + ["B Portföy", "C Portföy", "D Portföy", "E Portföy"],
            "n_nav": [100, 90, 80, 70, 60, 50, 40],
            "tax_category": "borclanma_para_maden",
            "withholding_rate": 0.175,
        }
    )

    selected = cash_proxy_codes(nav, fm, cfg, top=5)

    assert selected == ["A1", "A2", "A3", "B1", "C1"]


def test_f10_unknown_founder_code_is_excluded_without_name_or_code_fallback(cfg):
    idx = pd.bdate_range("2024-01-01", periods=100)
    nav = pd.DataFrame(10.0, index=idx, columns=["ABC1", "DEF1"])
    fm = pd.DataFrame(
        {
            "fund_code": ["ABC1", "DEF1"],
            "umbrella_type": "Para Piyasası Şemsiye Fonu",
            "founder_code": [None, "DEF"],
            "founder": ["ABC Portföy", "DEF Portföy"],
            "n_nav": [200, 100],
            "tax_category": "borclanma_para_maden",
        }
    )

    assert cash_proxy_codes(nav, fm, cfg) == ["DEF1"]


def test_f10_live_requires_profile_and_both_valors_but_history_does_not_backdate_valors(cfg):
    idx = pd.bdate_range("2024-01-01", periods=100)
    codes = ["P1", "P2", "P3", "P4"]
    nav = pd.DataFrame(10.0, index=idx, columns=codes)
    fm = pd.DataFrame(
        {
            "fund_code": codes,
            "umbrella_type": "Para Piyasası Şemsiye Fonu",
            "founder_code": ["A", "B", "C", "D"],
            "founder": ["A Portföy", "B Portföy", "C Portföy", "D Portföy"],
            "n_nav": [400, 300, 200, 100],
            "tax_category": "borclanma_para_maden",
            "isin": ["TRP1", "TRP2", "TRP3", "TRP4"],
            "snapshot_date": pd.Timestamp("2026-09-24").date(),
            "info_ok": [True, True, False, True],
            "buy_valor": [1, None, 1, 1],
            "sell_valor": [2, 2, 2, None],
            "can_buy": True,
            "can_sell": True,
            "tefas_status": "İşlem Görüyor",
            "last_success_at": pd.Timestamp("2026-09-24 09:00"),
        }
    )

    assert cash_proxy_codes(nav, fm, cfg) == codes
    with pytest.raises(ValueError, match="profile_asof"):
        cash_proxy_codes(nav, fm, cfg, require_execution_profile=True)
    assert cash_proxy_codes(
        nav,
        fm,
        cfg,
        require_execution_profile=True,
        profile_asof=pd.Timestamp("2026-09-24"),
    ) == ["P1", "P3"]
    assert (
        cash_proxy_codes(
            nav,
            fm,
            cfg,
            require_execution_profile=True,
            profile_asof=pd.Timestamp("2026-09-21"),
        )
        == []
    )


def test_live_b0_selection_ignores_missing_and_negative_source_fees(cfg):
    idx = pd.bdate_range("2026-09-01", periods=20)
    codes = ["MISSING_FEE", "NEGATIVE_FEE"]
    nav = pd.DataFrame(10.0, index=idx, columns=codes)
    fm = pd.DataFrame(
        {
            "fund_code": codes,
            "umbrella_type": "Para Piyasası Şemsiye Fonu",
            "founder_code": codes,
            "founder": codes,
            "n_nav": 20,
            "tax_category": "diger",
            "snapshot_date": pd.Timestamp("2026-09-24"),
            "buy_valor": 0,
            "sell_valor": 0,
            "entry_fee": [None, -0.01],
            "exit_fee": [None, 0.0],
            "can_buy": False,
            "can_sell": False,
            "tefas_status": "İşlem Görüyor",
            "last_success_at": pd.Timestamp("2026-09-24 08:00"),
        }
    )

    assert cash_proxy_codes(
        nav,
        fm,
        cfg,
        require_execution_profile=True,
        profile_asof="2026-09-24",
    ) == ["MISSING_FEE", "NEGATIVE_FEE"]


def test_f10_strict_live_rejects_negative_fractional_and_unknown_profile_fields(cfg):
    idx = pd.bdate_range("2026-09-01", periods=20)
    codes = ["OK", "NEG", "FRACTION", "NO_ID"]
    nav = pd.DataFrame(10.0, index=idx, columns=codes)
    fm = pd.DataFrame(
        {
            "fund_code": codes,
            "umbrella_type": "Para Piyasası Şemsiye Fonu",
            "founder_code": codes,
            "founder": codes,
            "n_nav": [20, 19, 18, 17],
            "tax_category": "diger",
            "isin": ["TR-OK", "TR-NEG", "TR-FRAC", None],
            "snapshot_date": pd.Timestamp("2026-09-24").date(),
            "info_ok": True,
            "buy_valor": [0, -1, 0.5, 0],
            "sell_valor": [0, 0, 0, 0],
            "can_buy": True,
            "can_sell": True,
            "tefas_status": "İşlem Görüyor",
            "last_success_at": pd.Timestamp("2026-09-24 09:00"),
        }
    )

    selected = cash_proxy_codes(
        nav,
        fm,
        cfg,
        require_execution_profile=True,
        profile_asof=pd.Timestamp("2026-09-24"),
    )

    assert selected == ["OK", "NO_ID"]


def test_f10_strict_live_accepts_category_when_isin_is_unknown(cfg):
    idx = pd.bdate_range("2026-09-01", periods=20)
    nav = pd.DataFrame({"P": 10.0}, index=idx)
    fm = pd.DataFrame(
        {
            "fund_code": ["P"],
            "umbrella_type": ["Para Piyasası Şemsiye Fonu"],
            "founder_code": ["F"],
            "founder": ["F"],
            "n_nav": [20],
            "tax_category": ["diger"],
            "category": ["Para Piyasası"],
            "isin": [None],
            "snapshot_date": [pd.Timestamp("2026-09-24").date()],
            "info_ok": [True],
            "buy_valor": [0],
            "sell_valor": [0],
            "can_buy": [False],  # status text is authoritative; stale stored booleans are advisory
            "can_sell": [None],
            "tefas_status": ["İşlem Görüyor"],
            "last_success_at": [pd.Timestamp("2026-09-24 09:00")],
        }
    )

    assert cash_proxy_codes(
        nav,
        fm,
        cfg,
        require_execution_profile=True,
        profile_asof=pd.Timestamp("2026-09-24"),
    ) == ["P"]
    assert (
        cash_proxy_codes(
            nav,
            fm,
            cfg,
            require_execution_profile=True,
            profile_asof=pd.Timestamp("2026-09-23"),
        )
        == []
    )


def test_f10_strict_live_requires_recent_success_and_known_status(cfg):
    idx = pd.bdate_range("2026-09-01", periods=20)
    codes = ["RECENT", "OLD", "NULL", "CLOSED"]
    nav = pd.DataFrame(10.0, index=idx, columns=codes)
    fm = pd.DataFrame(
        {
            "fund_code": codes,
            "umbrella_type": "Para Piyasası Şemsiye Fonu",
            "founder_code": codes,
            "founder": codes,
            "n_nav": 20,
            "tax_category": "diger",
            "isin": [f"TR-{code}" for code in codes],
            "snapshot_date": pd.Timestamp("2026-09-24").date(),
            "info_ok": True,
            "buy_valor": 0,
            "sell_valor": 0,
            "can_buy": [True, True, True, False],
            "can_sell": [True, True, True, True],
            "tefas_status": ["İşlem Görüyor", "İşlem Görüyor", "İşlem Görüyor", "Alımına Kapalı"],
            "last_success_at": [
                pd.Timestamp("2026-09-24 09:00"),
                pd.Timestamp("2026-09-16 09:00"),
                pd.NaT,
                pd.Timestamp("2026-09-24 09:00"),
            ],
        }
    )

    assert cash_proxy_codes(
        nav,
        fm,
        cfg,
        require_execution_profile=True,
        profile_asof=pd.Timestamp("2026-09-24 10:00"),
    ) == ["RECENT"]


def test_s5_6c_date_only_profile_asof_uses_morning_cutoff(cfg):
    nav = pd.DataFrame({"EARLY": 10.0, "LATE": 10.0}, index=pd.bdate_range("2026-09-01", periods=20))
    fm = pd.DataFrame(
        {
            "fund_code": ["EARLY", "LATE"],
            "umbrella_type": "Para Piyasası",
            "founder_code": ["A", "B"],
            "founder": ["A", "B"],
            "n_nav": [20, 19],
            "tax_category": "diger",
            "snapshot_date": [pd.Timestamp("2026-09-24").date()] * 2,
            "buy_valor": 0,
            "sell_valor": 0,
            "can_buy": True,
            "can_sell": True,
            "tefas_status": "İşlem Görüyor",
            "last_success_at": ["2026-09-24 08:59", "2026-09-24 11:00"],
            "source_published_at": ["2026-09-24 08:59", "2026-09-24 11:00"],
        }
    )

    selected = cash_proxy_codes(nav, fm, cfg, require_execution_profile=True, profile_asof=pd.Timestamp("2026-09-24"))

    assert selected == ["EARLY"]


def test_newest_available_profile_is_selected_before_freshness_and_never_falls_back(cfg):
    profiles = pd.DataFrame(
        {
            "fund_code": ["STALE", "STALE", "FUTURE", "FUTURE"],
            "snapshot_date": ["2026-09-23", "2026-09-24", "2026-09-23", "2026-09-25"],
            "last_success_at": [
                "2026-09-22 09:00",
                "2026-09-17 09:00",
                "2026-09-22 09:00",
                "2026-09-25 08:00",
            ],
            "source_published_at": [
                "2026-09-23 08:00",
                "2026-09-24 08:00",
                "2026-09-23 08:00",
                "2026-09-25 11:00",
            ],
            "buy_valor": [0, 0, 0, 0],
        }
    )

    selected = _execution_profiles_asof(profiles, "2026-09-25", "09:15")

    assert selected.set_index("fund_code")["snapshot_date"].to_dict() == {
        "STALE": pd.Timestamp("2026-09-24"),
        "FUTURE": pd.Timestamp("2026-09-23"),
    }
    assert selected.set_index("fund_code").loc["STALE", "last_success_at"] == "2026-09-17 09:00"

    # The direct strict selector receives the latest available row; stale metadata cannot
    # be replaced by an earlier fresh row. An independent eligible founder remains usable.
    nav = pd.DataFrame(10.0, index=pd.bdate_range("2026-09-01", periods=20), columns=["STALE", "VALID"])
    fm = pd.DataFrame(
        {
            "fund_code": ["STALE", "VALID"],
            "umbrella_type": "Para Piyasası",
            "founder_code": ["A", "B"],
            "founder": ["A", "B"],
            "n_nav": [20, 19],
            "tax_category": "diger",
            "snapshot_date": ["2026-09-24", "2026-09-24"],
            "buy_valor": 0,
            "sell_valor": 0,
            "can_buy": True,
            "can_sell": True,
            "tefas_status": "İşlem Görüyor",
            "last_success_at": ["2026-09-17 09:00", "2026-09-24 09:00"],
            "source_published_at": ["2026-09-24 08:00", "2026-09-24 08:00"],
        }
    )
    assert cash_proxy_codes(nav, fm, cfg, require_execution_profile=True, profile_asof="2026-09-25") == ["VALID"]
    assert (
        cash_proxy_codes(
            nav,
            fm.loc[fm["fund_code"].eq("STALE")],
            cfg,
            require_execution_profile=True,
            profile_asof="2026-09-25",
        )
        == []
    )


def test_strict_cash_proxy_rejects_unknown_status_but_keeps_other_founder(cfg):
    codes = ["UNKNOWN", "VALID"]
    nav = pd.DataFrame(10.0, index=pd.bdate_range("2026-09-01", periods=20), columns=codes)
    fm = pd.DataFrame(
        {
            "fund_code": codes,
            "umbrella_type": "Para Piyasası",
            "founder_code": ["U", "V"],
            "founder": ["U", "V"],
            "tax_category": "diger",
            "snapshot_date": "2026-09-24",
            "buy_valor": 0,
            "sell_valor": 0,
            "can_buy": True,
            "can_sell": True,
            "tefas_status": [None, "İşlem Görüyor"],
            "last_success_at": "2026-09-24 09:00",
        }
    )

    assert cash_proxy_codes(nav, fm, cfg, require_execution_profile=True, profile_asof="2026-09-24") == ["VALID"]


def test_cash_proxy_returns_equal_weight(cfg):
    idx = pd.bdate_range("2024-01-01", periods=10)
    nav = pd.DataFrame({"M1": np.linspace(1, 1.1, 10), "M2": np.linspace(1, 0.9, 10)}, index=idx)
    fm = pd.DataFrame(
        {
            "fund_code": ["M1", "M2"],
            "umbrella_type": ["Para Piyasası Şemsiye Fonu"] * 2,
            "founder_code": ["POY", "KUZ"],
            "founder": ["POYRAZ PORTFÖY"] * 2,
            "n_nav": [10, 10],
            "tax_category": ["borclanma_para_maden"] * 2,
            "withholding_rate": [0.175] * 2,
        }
    )
    codes = cash_proxy_codes(nav, fm, cfg)
    r = cash_proxy_returns(nav, codes)
    assert len(codes) == 2 and np.isfinite(r).all()


def test_f20_same_nav_real_fifo_b0_and_clipped_backtest_slot_can_differ():
    idx = pd.bdate_range("2026-09-01", periods=5)
    nav = pd.DataFrame(
        {
            "PP0": [1.0, 1.03, 1.02, 1.05, 1.08],
            "PP1": [2.0, 2.01, 2.05, 2.04, 2.10],
        },
        index=idx,
    )
    codes = ["PP0", "PP1"]

    backtest_slot = 100.0 * (1.0 + cash_proxy_returns(nav, codes)).cumprod()
    ledger = Ledger(
        meta_for_synthetic(codes, buy_valor=0, sell_valor=1, tax_rate=0.20),
        cash=100.0,
        dates=idx.to_numpy(),
    )
    for i, code in enumerate(codes):
        assert ledger.buy(i, 50.0, float(nav.iloc[0][code]), 0) is not None
    for i, code in enumerate(codes):
        assert ledger.sell(i, ledger.units[i], float(nav.iloc[-1][code]), len(idx) - 1) is not None
    real_fund_ledger_equity = ledger.equity(nav.iloc[-1].to_numpy(float), None, 0.0)

    assert ledger.receivable_total() > 0  # satış T+1 alacak; FIFO satış vergisi net tutarı etkiler
    assert not np.isclose(real_fund_ledger_equity, float(backtest_slot.iloc[-1]), rtol=1e-10, atol=1e-10)


def test_b0_slot_mtm_and_liquidation():
    """Ekle-3: B0 = slot MTM; cash_index = B0 slot defteri; vergi yalnızca tasfiye değerinde görünür."""
    nav = make_panel(n_funds=2, days=300, drift=0.0)
    slot_px = pd.Series(np.linspace(1.0, 1.05, len(nav)), index=nav.index)  # ~%5 brüt kazanç
    cfg = BacktestConfig(initial_capital=100, warmup_days=10)
    meta = meta_for_synthetic(list(nav.columns))
    meta = type(meta)(
        codes=meta.codes,
        buy_valor=meta.buy_valor,
        sell_valor=meta.sell_valor,
        entry_fee=meta.entry_fee,
        exit_fee=meta.exit_fee,
        tax_rate=meta.tax_rate,
        tax_unknown=meta.tax_unknown,
        equity_intensive=meta.equity_intensive,
        can_buy=meta.can_buy,
        can_sell=meta.can_sell,
        tax_category=meta.tax_category,
        tax_schedule=((pd.Timestamp("2016-01-01").date(), {"default": 0.10}),),  # deterministik slot stopajı
    )
    res = run_backtest(nav, meta, CashOnly(), cfg, cash_nav=slot_px, cash_category="borclanma_para_maden")
    assert np.allclose(res.equity.to_numpy(), res.cash_index.to_numpy(), rtol=1e-9)  # B0 = slot MTM
    assert res.turnover == 0  # slot işlemleri turnover'a girmez (Ekle-2)
    assert (res.fills["code"] == SLOT).all() and not res.fills.empty
    assert res.unrealized_tax > 0 and res.liquidation_value < res.equity.iloc[-1]  # dönem sonu bozum vergisi


def test_strategy_cash_is_slot_position():
    """P02: strateji nakdi slot pozisyonu; risky alış slot bozumuyla finanse edilir; özdeşlik korunur."""
    nav = make_panel(n_funds=3, days=400, drift=0.0005)
    slot_px = pd.Series(1.0002 ** np.arange(len(nav)), index=nav.index)
    cfg = BacktestConfig(initial_capital=100, warmup_days=260)
    res = run_backtest(
        nav,
        meta_for_synthetic(list(nav.columns)),
        TopNMomentum(n=2, max_weight=0.5),
        cfg,
        cash_returns=pd.Series(0.0002, index=nav.index),
        cash_nav=slot_px,
        cash_category="borclanma_para_maden",
    )
    assert (res.weights.sum(axis=1) <= 1 + 1e-9).all() and (res.weights.max(axis=1) <= 0.5 + 1e-9).all()
    assert (res.equity > 0).all() and res.equity.iloc[0] == 100
    assert (res.fills["code"] == SLOT).any()  # sweep/bozum defterde
    risky = res.fills[res.fills["code"] != SLOT]
    assert not risky.empty and res.turnover > 0  # turnover yalnızca risky


def test_dd_trigger_moves_risky_to_slot_same_day():
    """Ekle-4: DD tetiklendiğinde slot hedef payı aynı gün artar (risky→slot T+0)."""
    nav = make_panel(n_funds=3, days=500, drift=0.0)
    nav.iloc[300:, :] *= np.linspace(1, 0.6, 200)[:, None]
    slot_px = pd.Series(1.0002 ** np.arange(len(nav)), index=nav.index)
    cfg = BacktestConfig(initial_capital=100, warmup_days=260, dd_trigger=0.12)
    res = run_backtest(
        nav, meta_for_synthetic(list(nav.columns)), TopNMomentum(n=3, max_weight=1.0), cfg, cash_nav=slot_px
    )
    ev = res.events
    t0 = int(ev.loc[ev["event"] == "dd_trigger", "idx"].iloc[0])
    reb = ev[ev["event"] == "rebalance"].set_index("idx")
    assert float(reb.loc[t0, "locked_weight"]) == 0.0
    # tetik günündeki hedefte slot payı, risky maruziyet düşümü kadar artar (risky ağırlık = 1 − slot)
    w = res.weights
    assert (w[SLOT] > 0).any()
    assert w[SLOT].iloc[-1] > w[SLOT].iloc[0]  # çöküş sonrası slot payı artar


def test_lot_tax_rate_by_purchase_date():
    """P04: 09.07.2025 öncesi/sonrası lot farklı oran (config yedek schedule)."""
    meta = meta_for_synthetic(["F0"])
    meta = type(meta)(
        codes=meta.codes,
        buy_valor=meta.buy_valor,
        sell_valor=meta.sell_valor,
        entry_fee=meta.entry_fee,
        exit_fee=meta.exit_fee,
        tax_rate=meta.tax_rate,
        tax_unknown=meta.tax_unknown,
        equity_intensive=meta.equity_intensive,
        can_buy=meta.can_buy,
        can_sell=meta.can_sell,
        tax_category=meta.tax_category,
        tax_schedule=(
            (pd.Timestamp("2016-01-01").date(), {"default": 0.05}),
            (pd.Timestamp("2025-07-09").date(), {"default": 0.10}),
        ),
    )
    cal = pd.bdate_range("2025-06-01", periods=40)
    led = Ledger(meta, cash=100.0, dates=cal.to_numpy())
    led.buy(0, 50.0, 10.0, 0)  # 02.06.2025 → eski rejim
    led.buy(0, 50.0, 10.0, 30)  # 15.07.2025 → yeni rejim
    rates = [lot.tax_rate for lot in led.lots[0]]
    assert rates[0] == 0.05 and rates[1] == 0.10


def test_unrealized_tax_and_liquidation_identity():
    from janus.backtest.ledger import Lot

    meta = meta_for_synthetic(["F0"], tax_rate=0.2)
    led = Ledger(meta, cash=50.0)
    led.units[0] = 10.0
    led.lots[0] = [Lot(10.0, 8.0, 0, 0.2, 0)]
    nav = np.array([10.0])
    ut = led.unrealized_tax(nav)
    assert np.isclose(ut, 0.2 * 10 * (10.0 - 8.0))  # kârlı lot stopajı
    assert np.isclose(led.equity(nav, None, 0.0) - ut, 50.0 + 100.0 - ut)
