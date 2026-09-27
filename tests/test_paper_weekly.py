from __future__ import annotations

import pandas as pd
import pytest

from janus.data.store import Store
from janus.paper.core import _ensure_paper_tables
from janus.paper.weekly import _load_real_b0_equity, compute_b0_tracking


def _track(actual: pd.Series, proxy: pd.Series, start: pd.Timestamp, end: pd.Timestamp, epoch: pd.Timestamp):
    return compute_b0_tracking(actual, proxy, epoch_start=epoch, start=start, end=end)


def test_b0_internal_buy_sell_and_settlement_do_not_create_returns():
    dates = pd.bdate_range("2026-09-01", periods=5)
    components = pd.DataFrame(
        {
            "fund_value": [0.0, 100.0, 0.0, 0.0, 0.0],
            "cash": [100.0, 0.0, 100.0, 0.0, 100.0],
            "receivables": [0.0, 0.0, 0.0, 100.0, 0.0],
        },
        index=dates,
    )
    actual = components.sum(axis=1)
    proxy = pd.Series(100.0, index=dates)

    result = _track(actual, proxy, dates[0], dates[-1], dates[0])

    assert actual.eq(100.0).all()
    assert result["status"] == "değerlendirildi"
    assert result["real_return"] == pytest.approx(0.0)
    assert result["proxy_return"] == pytest.approx(0.0)
    assert result["difference_pp"] == pytest.approx(0.0)


def test_b0_weekly_has_no_fee_drag_but_retains_tax_and_settlement_effect():
    dates = pd.bdate_range("2026-09-01", periods=3)
    components = pd.DataFrame(
        {
            "fund_value": [100.0, 0.0, 0.0],
            "cash": [0.0, 0.0, 90.0],
            "receivables": [0.0, 90.0, 0.0],
        },
        index=dates,
    )
    actual = components.sum(axis=1)
    proxy = pd.Series(100.0, index=dates)

    result = _track(actual, proxy, dates[0], dates[-1], dates[0])

    assert actual.iloc[1] == actual.iloc[2] == 90.0
    assert result["real_return"] == pytest.approx(-0.10)
    assert result["proxy_return"] == pytest.approx(0.0)
    assert result["difference_pp"] == pytest.approx(-10.0)


def test_b0_tracking_does_not_evaluate_a_week_crossing_init_reset():
    dates = pd.bdate_range("2026-09-01", periods=8)
    actual = pd.Series([100.0, 101.0, 102.0, 100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
    proxy = pd.Series(100.0, index=dates)
    epoch = dates[4]

    crossing = _track(actual, proxy, dates[0], dates[-1], epoch)
    post_reset = _track(actual, proxy, dates[4], dates[-1], epoch)

    assert crossing["status"] == "değerlendirilemedi"
    assert "sınırını kesiyor" in crossing["reason"]
    assert post_reset["status"] == "değerlendirildi"
    assert post_reset["real_return"] == pytest.approx(104.0 / 101.0 - 1.0)


def test_b0_tracking_requires_two_common_dates_without_forward_fill():
    dates = pd.bdate_range("2026-09-01", periods=3)
    actual = pd.Series([100.0, 101.0], index=dates[:2])
    proxy = pd.Series([100.0, 100.5], index=dates[1:])

    result = _track(actual, proxy, dates[0], dates[-1], dates[0])

    assert result["status"] == "değerlendirilemedi"
    assert result["reason"] == "iki ortak geçerli tarih yok"


def test_b0_tracking_rejects_missing_value_on_a_common_date():
    dates = pd.bdate_range("2026-09-01", periods=3)
    actual = pd.Series([100.0, float("nan"), 102.0], index=dates)
    proxy = pd.Series([100.0, 101.0, 102.0], index=dates)

    result = _track(actual, proxy, dates[0], dates[-1], dates[0])

    assert result["status"] == "değerlendirilemedi"
    assert result["reason"] == "ortak tarihlerde eksik/geçersiz equity"


def test_real_b0_equity_uses_cash_receivables_and_rejects_risky_attribution(tmp_path):
    store = Store(tmp_path / "weekly.duckdb")
    _ensure_paper_tables(store)
    dates = pd.bdate_range("2026-09-01", periods=2)
    store.con.execute(
        "CREATE TABLE janus_pilot_epoch "
        "(epoch_id INTEGER, started_at TIMESTAMP, initialized_at TIMESTAMP, published_at TIMESTAMP, available_from TIMESTAMP)"
    )
    store.con.execute(
        "INSERT INTO janus_pilot_epoch VALUES (1, ?, ?, ?, ?)",
        [date.to_pydatetime() for date in [dates[0]] * 4],
    )
    store.con.executemany(
        "INSERT INTO paper_equity (portfolio_name,date,equity,cash,receivables,risky_value,slot_value,nav_asof) "
        "VALUES ('live', ?, ?, ?, ?, 0, ?, ?)",
        [
            (day.date(), 100.0, cash, receivable, slot, day.date())
            for day, cash, receivable, slot in [
                (dates[0], 100.0, 0.0, 0.0),
                (dates[1], 0.0, 100.0, 0.0),
            ]
        ],
    )
    store.con.execute(
        "INSERT INTO paper_b0_memberships VALUES (?, ?, 'B0F', 'PYSA', 0, 0)",
        [dates[0].date(), dates[0].date()],
    )

    actual, epoch, reason = _load_real_b0_equity(store, dates[-1])
    assert reason is None
    assert epoch == dates[0]
    assert actual.tolist() == pytest.approx([100.0, 100.0])

    store.con.execute(
        "INSERT INTO paper_fills VALUES ('risk-fill', 'p', 'RISK', 'BUY', 1, 1, 1, 0, 0, 0, ?)",
        [dates[1].date()],
    )
    _, _, reason = _load_real_b0_equity(store, dates[-1])
    assert reason == "risky fill nakit/alacak atfını belirsiz kılıyor"
    store.close()
