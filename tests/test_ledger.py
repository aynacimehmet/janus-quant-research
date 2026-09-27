import numpy as np

from janus.backtest.costs import meta_for_synthetic
from janus.backtest.ledger import Ledger


def test_accounting_identity_and_valor():
    meta = meta_for_synthetic(["A", "B"], buy_valor=1, sell_valor=2, tax_rate=0.175)
    led = Ledger(meta, cash=100.0)
    p0 = np.array([10.0, 20.0])
    led.buy(0, 40.0, 10.0, idx=0)  # 4 pay A
    led.buy(1, 40.0, 20.0, idx=0)  # 2 pay B
    assert abs(led.cash - 20.0) < 1e-9 and abs(led.equity(p0) - 100.0) < 1e-9
    # D+0: alınan pay henüz satılamaz (buy_valor=1)
    assert led.sell(0, 4.0, 12.0, idx=0) is None
    # D+1: A 12'ye satılır → brüt 48, kâr 8, vergi 1.4, net 46.6 → alacak D+3
    f = led.sell(0, 4.0, 12.0, idx=1)
    assert f is not None and abs(f.tax - 1.4) < 1e-9 and abs(f.realized_gain - 8.0) < 1e-9
    assert abs(led.cash - 20.0) < 1e-9 and abs(led.receivable_total() - 46.6) < 1e-9
    p1 = np.array([12.0, 20.0])
    assert abs(led.equity(p1) - (20.0 + 46.6 + 40.0)) < 1e-9  # nakit + alacak + B
    led.settle(2)
    assert abs(led.cash - 20.0) < 1e-9  # henüz değil
    led.settle(3)
    assert abs(led.cash - 66.6) < 1e-9  # valör doldu


def test_fifo_tax_and_loss_no_tax():
    meta = meta_for_synthetic(["A"], buy_valor=0, sell_valor=0, tax_rate=0.10)
    led = Ledger(meta, cash=100.0)
    led.buy(0, 50.0, 10.0, idx=0)  # 5 pay @10
    led.buy(0, 30.0, 15.0, idx=1)  # 2 pay @15
    f = led.sell(0, 6.0, 12.0, idx=2)  # FIFO: 5 pay @10 (kâr 10) + 1 pay @15 (zarar 3)
    assert abs(f.realized_gain - 7.0) < 1e-9
    assert abs(f.tax - 1.0) < 1e-9  # vergi yalnızca kârlı lotta: 10 × 0.10
    assert abs(led.units[0] - 1.0) < 1e-9 and abs(led.cash - (20.0 + 72.0 - 1.0)) < 1e-9


def test_fees_and_suspension_haircut():
    meta = meta_for_synthetic(["A"], buy_valor=0, sell_valor=0, tax_rate=0.0)
    meta.entry_fee[0], meta.exit_fee[0] = 0.01, 0.02
    led = Ledger(meta, cash=101.0)
    led.buy(0, 101.0, 10.0, idx=0)
    assert abs(led.units[0] - 10.0) < 1e-9 and abs(led.fees_paid - 1.0) < 1e-9
    p = np.array([10.0])
    assert abs(led.equity(p, np.array([True]), haircut=0.3) - 70.0) < 1e-9
    f = led.sell(0, 10.0, 10.0, idx=1)
    assert abs(f.gross - 98.0) < 1e-9 and abs(led.fees_paid - 3.0) < 1e-9


def test_cash_accrual_taxed():
    led = Ledger(meta_for_synthetic(["A"]), cash=100.0, cash_tax_rate=0.175)
    led.accrue_cash(0.001)
    assert abs(led.cash - 100.0825) < 1e-9
