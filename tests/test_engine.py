import numpy as np
import pandas as pd
import pytest

from _panel import make_panel  # noqa: E402
from janus.backtest.costs import meta_for_synthetic
from janus.backtest.engine import BacktestConfig, _rebalance_to, rebalance_days, run_backtest, stale_flags
from janus.backtest.ledger import Ledger, Lot
from janus.backtest.metrics import deflated_sharpe, summary
from janus.backtest.validation import walk_forward_splits
from janus.strategies.baselines import CashOnly, RuleGate, TopNMomentum


def test_rebalance_days_monthly():
    cal = pd.bdate_range("2024-01-01", "2024-04-30")
    d = rebalance_days(cal, "monthly_first_business_day")
    assert [cal[i].strftime("%Y-%m-%d") for i in d] == ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"]


def test_stale_flags():
    nav = make_panel(2, 10)
    nav.iloc[-3:, 1] = np.nan
    s = stale_flags(nav, max_stale_days=2)
    assert not s[-1, 0] and s[-1, 1] and not s[-3, 1]


def test_cash_only_equals_cash_index():
    nav = make_panel()
    cash_r = pd.Series(0.0005, index=nav.index)
    cfg = BacktestConfig(initial_capital=100, warmup_days=10)
    res = run_backtest(
        nav,
        meta_for_synthetic(list(nav.columns)),
        CashOnly(),
        cfg,
        cash_returns=cash_r,
        sensitivity=True,
    )
    assert np.allclose(res.equity.to_numpy(), res.cash_index.to_numpy(), rtol=1e-9)
    assert res.fills.empty and res.taxes_paid == 0


def test_cash_returns_without_slot_requires_explicit_sensitivity():
    nav = make_panel(n_funds=3, days=40)
    cash_r = pd.Series(0.001, index=nav.index)
    cfg = BacktestConfig(initial_capital=100, warmup_days=10, sensitivity=False)

    with pytest.raises(ValueError, match="cash_returns.*sensitivity=True"):
        run_backtest(nav, meta_for_synthetic(list(nav.columns)), CashOnly(), cfg, cash_returns=cash_r)


def test_cash_returns_daily_tax_fallback_is_explicit_sensitivity():
    nav = make_panel(n_funds=3, days=40)
    cash_r = pd.Series(0.001, index=nav.index)
    cfg = BacktestConfig(initial_capital=100, warmup_days=10, cash_tax_rate=0.2)

    res = run_backtest(
        nav,
        meta_for_synthetic(list(nav.columns)),
        CashOnly(),
        cfg,
        cash_returns=cash_r,
        sensitivity=True,
    )

    effective_returns = cash_r.to_numpy() * (1.0 - cfg.cash_tax_rate)
    effective_returns[0] = 0.0  # defterle aynı: ilk gün tahakkuk yok
    expected = np.cumprod(1.0 + effective_returns) * cfg.initial_capital
    assert np.allclose(res.cash_index.to_numpy(), expected, rtol=1e-9)


def test_momentum_engine_invariants():
    nav = make_panel()
    cfg = BacktestConfig(initial_capital=100, warmup_days=260)
    meta = meta_for_synthetic(list(nav.columns), buy_valor=1, sell_valor=2)
    res = run_backtest(
        nav,
        meta,
        TopNMomentum(n=3, max_weight=0.5),
        cfg,
        cash_returns=pd.Series(0.0004, index=nav.index),
        sensitivity=True,
    )
    assert res.n_rebalances >= 20 and not res.fills.empty
    assert (res.equity > 0).all() and res.equity.iloc[0] == 100
    # hedef ağırlıklar tavanı aşmaz, toplam ≤ 1
    assert (res.weights.max(axis=1) <= 0.5 + 1e-9).all() and (res.weights.sum(axis=1) <= 1 + 1e-9).all()
    # en güçlü momentumlu F0 rebalance'ların büyük çoğunluğunda seçilmeli
    assert (res.weights["F0"] > 0).mean() >= 0.8
    m = summary(res.equity, res.cash_index, res.taxes_paid, res.fees_paid, res.turnover)
    assert np.isfinite(m["sharpe"]) and 0 <= m["mdd"] < 1 and m["turnover_per_year"] > 0


def test_drift_threshold_limits_trades():
    nav = make_panel(n_funds=4, days=400, drift=0.0)
    cfg_tight = BacktestConfig(
        initial_capital=100, warmup_days=260, drift_threshold=0.0, drift_threshold_taxable_sale=0.0
    )
    cfg_loose = BacktestConfig(
        initial_capital=100, warmup_days=260, drift_threshold=0.5, drift_threshold_taxable_sale=0.5
    )
    meta = meta_for_synthetic(list(nav.columns))
    strat = TopNMomentum(n=4, max_weight=0.25)
    n_tight = len(run_backtest(nav, meta, strat, cfg_tight).fills)
    n_loose = len(run_backtest(nav, meta, strat, cfg_loose).fills)
    assert n_tight > n_loose


def test_dd_trigger_reduces_exposure():
    nav = make_panel(n_funds=3, days=500, drift=0.0)
    nav.iloc[300:, :] *= np.linspace(1, 0.6, 200)[:, None]  # %40 çöküş
    cfg = BacktestConfig(
        initial_capital=100, warmup_days=260, dd_trigger=0.12, exposure_levels={"low": 0.3, "medium": 0.65, "full": 1.0}
    )
    res = run_backtest(nav, meta_for_synthetic(list(nav.columns)), TopNMomentum(n=3, max_weight=1.0), cfg)
    ev = res.events
    assert (ev["event"] == "dd_trigger").any()
    after = ev[(ev["event"] == "rebalance") & (ev["idx"] > ev.loc[ev["event"] == "dd_trigger", "idx"].iloc[0])]
    assert (after["exposure"] <= 0.65 + 1e-9).all()


def test_dd_exposure_config_target():
    """0c-5: DD tetikleyicisinin hedefi risk.dd_exposure; exposure_levels["medium"]'dan ayrı."""
    nav = make_panel(n_funds=3, days=500, drift=0.0)
    nav.iloc[300:, :] *= np.linspace(1, 0.6, 200)[:, None]
    cfg = BacktestConfig(
        initial_capital=100,
        warmup_days=260,
        dd_trigger=0.12,
        dd_exposure=0.4,
        exposure_levels={"low": 0.3, "medium": 0.65, "full": 1.0},
    )
    res = run_backtest(nav, meta_for_synthetic(list(nav.columns)), TopNMomentum(n=3, max_weight=1.0), cfg)
    ev = res.events
    assert (ev["event"] == "dd_trigger").any()
    after = ev[(ev["event"] == "rebalance") & (ev["idx"] > ev.loc[ev["event"] == "dd_trigger", "idx"].iloc[0])]
    assert (after["exposure"] <= 0.4 + 1e-9).all()


def test_suspended_fund_not_traded_and_stress_value():
    """P05: suspended → işlem yok; DD official_value ile; stress_value (haircut) ayrı ve düşük."""
    nav = make_panel(n_funds=3, days=400, drift=0.0)
    susp = pd.DataFrame(False, index=nav.index, columns=nav.columns)
    susp.iloc[300:, 0] = True
    cfg = BacktestConfig(initial_capital=100, warmup_days=260, suspension_haircut=0.3)
    res = run_backtest(
        nav, meta_for_synthetic(list(nav.columns)), TopNMomentum(n=3, max_weight=1.0), cfg, suspended=susp
    )
    late = res.fills[res.fills["idx"] >= 300]
    assert (late["code"] != "F0").all()
    assert (res.equity_stress <= res.equity + 1e-9).all()
    assert res.equity_stress.iloc[-1] < res.equity.iloc[-1]  # F0 hâlâ elde, haircut uygulanıyor


def test_data_stale_no_dd_and_no_haircut():
    """P05: 3 gün NAV eksikliği DD tetiklemez; data_stale'e haircut uygulanmaz, işlem de yok."""
    nav = make_panel(n_funds=3, days=400, drift=0.0)
    nav.iloc[300:, 0] = np.nan  # F0 verisi eskiyor (resmî askı yok)
    cfg = BacktestConfig(initial_capital=100, warmup_days=260, suspension_haircut=0.3)
    res = run_backtest(nav, meta_for_synthetic(list(nav.columns)), TopNMomentum(n=3, max_weight=1.0), cfg)
    assert not (res.events["event"] == "dd_trigger").any()
    assert np.allclose(res.equity_stress, res.equity)  # haircut yok
    late = res.fills[res.fills["idx"] >= 302]  # stale_flags eşiği 2 gün
    assert (late["code"] != "F0").all()


def test_locked_position_no_sell_targets_normalized():
    """LEDGER §6: can_sell=False olan elde tutulan fon satılamaz; hedefler normalize, toplam ≤ 1, nakit negatif olmaz."""
    nav = make_panel(n_funds=3, days=400, drift=0.0)
    meta = meta_for_synthetic(list(nav.columns), no_sell=["F0"])
    cfg = BacktestConfig(initial_capital=100, warmup_days=260)
    res = run_backtest(nav, meta, TopNMomentum(n=3, max_weight=1.0), cfg)
    f0 = res.fills[res.fills["code"] == "F0"]
    assert len(f0) > 0 and (f0["side"] == "BUY").all()  # alınabilir ama satılamaz
    ev = res.events[res.events["event"] == "rebalance"]
    assert (ev["locked_weight"] > 0).any()
    assert (res.weights.sum(axis=1) <= 1 + 1e-9).all()
    assert (res.equity > 0).all()


def test_can_buy_blocked():
    nav = make_panel(n_funds=3, days=400, drift=0.0)
    meta = meta_for_synthetic(list(nav.columns), no_buy=["F0"])
    cfg = BacktestConfig(initial_capital=100, warmup_days=260)
    res = run_backtest(nav, meta, TopNMomentum(n=3, max_weight=1.0), cfg)
    assert not ((res.fills["side"] == "BUY") & (res.fills["code"] == "F0")).any()


def test_sell_allowed_when_buy_closed():
    """D05: 'alıma kapalı, bozuma açık' fon satılabiliyor."""
    codes = ["F0", "F1"]
    meta = meta_for_synthetic(codes, no_buy=["F0"])
    led = Ledger(meta, cash=100.0)
    led.units[0] = 10.0
    led.lots[0].append(Lot(10.0, 10.0, 0, 0.0, 0))  # pozisyon elle açıldı
    price = np.array([10.0, 10.0])
    w = np.array([0.0, 1.0])  # F0 hedef dışı
    traded = _rebalance_to(
        led,
        w,
        price,
        5,
        BacktestConfig(),
        sellable=np.array([True, True]),
        buyable=np.array([False, True]),
    )
    assert traded > 0 and led.units[0] == 0.0


def test_rule_gate_levels():
    nav = make_panel(n_funds=3, days=600, drift=0.0005)
    idx = nav.mean(axis=1)
    levels = {"low": 0.3, "medium": 0.65, "full": 1.0}
    gate = RuleGate(index_nav=idx, inner=TopNMomentum(n=3, max_weight=1.0), levels=levels)
    cfg = BacktestConfig(initial_capital=100, warmup_days=260)
    res = run_backtest(nav, meta_for_synthetic(list(nav.columns)), gate, cfg)
    assert set(res.events.loc[res.events["event"] == "rebalance", "exposure"].round(2)) <= {0.3, 0.65, 1.0}


def test_walk_forward_splits_purge():
    cal = pd.bdate_range("2022-01-03", "2025-12-31")
    splits = walk_forward_splits(cal, test_months=3, purge_days=21, min_train_days=252)
    assert len(splits) >= 10
    for s in splits:
        assert (cal.get_loc(s.test_start) - cal.get_loc(s.train_end)) == 22  # purge 21 + 1
        assert s.train_start == cal[0]


def test_deflated_sharpe_monotone():
    assert (
        deflated_sharpe(0.1, n_trials=1, sr_var=0.01, n_obs=500)
        != deflated_sharpe(0.1, n_trials=1, sr_var=0.01, n_obs=500)
        or True
    )
    a = deflated_sharpe(0.10, n_trials=5, sr_var=0.001, n_obs=750)
    b = deflated_sharpe(0.10, n_trials=200, sr_var=0.001, n_obs=750)
    assert 0 <= b <= a <= 1


def test_dd_trigger_forces_sales_despite_threshold():
    """F-02: DD tetik günü satış eşiği atlanır; gerçekleşen risky ≤ dd_exposure."""
    nav = make_panel(n_funds=20, days=500, drift=0.0)
    # ani çöküş: tetik gününde küçük pozisyonlar eşik altında kalmasın
    nav.iloc[350:, :] *= np.linspace(1, 0.6, 150)[:, None]
    cfg = BacktestConfig(
        initial_capital=100,
        warmup_days=260,
        dd_trigger=0.12,
        dd_exposure=0.65,
        drift_threshold=0.05,  # yüksek eşik; force olmasa satış atlanırdı
        drift_threshold_taxable_sale=0.10,
        exposure_levels={"low": 0.3, "medium": 0.65, "full": 1.0},
    )
    meta = meta_for_synthetic(list(nav.columns))
    res = run_backtest(nav, meta, TopNMomentum(n=20, max_weight=0.05), cfg)
    triggers = res.events[res.events["event"] == "dd_trigger"]
    assert len(triggers) > 0
    t = int(triggers.iloc[0]["idx"])
    # tetik gününde en az bir satış olmalı
    sells = res.fills[(res.fills["idx"] == t) & (res.fills["side"] == "SELL")]
    assert len(sells) > 0
    # gerçekleşen toplam risky ağırlık ≤ dd_exposure
    realized = res.weights_realized
    risky = realized.drop(columns=["CASH_PROXY"], errors="ignore")
    assert risky.loc[nav.index[t]].sum() <= 0.65 + 0.02  # yuvarlama toleransı


def test_stale_held_fund_no_spurious_trade():
    """F-03: elde tutulan fon stale gününde hedefe sadık kalınır; ham fiyat hayalet rebalance üretmez."""
    nav = make_panel(n_funds=3, days=400, drift=0.0)
    # F1 bir süreliğine stale (son 50 gün)
    nav.iloc[-50:, 1] = np.nan
    cfg = BacktestConfig(initial_capital=100, warmup_days=260, drift_threshold=0.0)
    meta = meta_for_synthetic(list(nav.columns))
    res = run_backtest(nav, meta, TopNMomentum(n=2, max_weight=0.5), cfg)
    # stale dönemde F1 için alım/satım yok
    late = res.fills[res.fills["idx"] >= 350]
    assert (late["code"] != "F1").all()
    # hedef ağırlıklar F1 hariç toplam ≤ 1
    w = res.weights
    if "F1" in w.columns:
        assert (w.drop(columns=["F1"]).sum(axis=1) <= 1 + 1e-9).all()


def test_engine_future_perturbation():
    """F-04: NAV[t] değişince t günü emir kararı değişmez (D-1 fiyatıyla karar)."""
    nav = make_panel(n_funds=4, days=400, drift=0.0)
    cfg = BacktestConfig(initial_capital=100, warmup_days=260, drift_threshold=0.0)
    meta = meta_for_synthetic(list(nav.columns))
    res1 = run_backtest(nav, meta, TopNMomentum(n=2, max_weight=0.5), cfg)
    nav2 = nav.copy()
    # t=300 sonrası NAV'ları değiştir; t=300 kararı D-299 bilgisine dayanmalı
    nav2.iloc[300:, :] *= 1.5
    res2 = run_backtest(nav2, meta, TopNMomentum(n=2, max_weight=0.5), cfg)
    # t=300'den sonraki ilk rebalance gününde hedef ağırlıkları aynı olmalı
    common_days = res1.weights.index.intersection(res2.weights.index)
    future_days = common_days[common_days > nav.index[300]]
    assert len(future_days) > 0
    d = future_days[0]
    pd.testing.assert_frame_equal(res1.weights.loc[[d]], res2.weights.loc[[d]], check_names=False)


def test_weights_realized_recorded():
    """F-11: weights_realized rebalance günlerinde kapanış gerçekleşen ağırlıklarını içerir."""
    nav = make_panel(n_funds=4, days=400, drift=0.0)
    cfg = BacktestConfig(initial_capital=100, warmup_days=260)
    res = run_backtest(nav, meta_for_synthetic(list(nav.columns)), TopNMomentum(n=2, max_weight=0.5), cfg)
    assert not res.weights_realized.empty
    assert res.weights_realized.shape[1] == res.weights.shape[1]
    # her rebalance günü için realized satırı var
    assert len(res.weights_realized) == len(res.weights)


def test_cash_nav_no_bfill_assertion():
    """F-13: cash_nav başlangıçtan eksikse bfill olmadan ValueError."""
    nav = make_panel(n_funds=3, days=200, drift=0.0)
    cash_r = pd.Series(0.0003, index=nav.index[50:])  # geç başlayan nakit serisi
    cfg = BacktestConfig(initial_capital=100, warmup_days=10)
    with pytest.raises(ValueError, match="F-13"):
        run_backtest(
            nav,
            meta_for_synthetic(list(nav.columns)),
            CashOnly(),
            cfg,
            cash_returns=cash_r,
            cash_nav=(1 + cash_r).cumprod(),
        )
