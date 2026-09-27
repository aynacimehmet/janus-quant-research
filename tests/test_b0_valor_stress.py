"""B0's own executable-fund ledger must report the sell-valor +1 stress."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from janus.backtest.costs import meta_for_synthetic
from janus.backtest.robustness import b0_fund_valor_stress


def _execution_profiles(dates: pd.DatetimeIndex, codes: list[str]) -> pd.DataFrame:
    index = pd.MultiIndex.from_product([dates, codes], names=["decision_date", "fund_code"])
    return pd.DataFrame(
        {
            "buy_valor": 0,
            "sell_valor": 1,
            "entry_fee": 0.01,
            "exit_fee": 0.01,
            "tax_category": "new",
            "can_buy": True,
            "can_sell": True,
            "status": "İşlem Görüyor",
            "execution_source": "PIT",
            "buy_reason": "ok",
            "sell_reason": "ok",
        },
        index=index,
    )


def test_b0_lot_tax_category_is_profiled_at_buy_and_immutable_afterward():
    dates = pd.bdate_range("2026-09-22", periods=8)
    nav = pd.DataFrame({"P1": np.linspace(100.0, 110.0, len(dates))}, index=dates)
    codes = ["P1"]
    profiles = _execution_profiles(dates, codes)
    profiles.loc[(dates[0], "P1"), "tax_category"] = "old"
    changed_later = profiles.copy()
    changed_later.loc[changed_later.index.get_level_values("decision_date") > dates[0], "tax_category"] = "changed"

    static_meta = meta_for_synthetic(codes, buy_valor=0, sell_valor=1, tax_rate=0.20)
    static_meta = replace(
        static_meta,
        tax_category=np.array(["new"], dtype=object),
        tax_schedule=(
            (
                pd.Timestamp("2026-01-01").date(),
                {"old": 0.10, "new": 0.0, "changed": 0.05},
            ),
        ),
    )

    baseline = b0_fund_valor_stress(nav, codes, static_meta, profiles, select_end=dates[3])
    later_changed = b0_fund_valor_stress(nav, codes, static_meta, changed_later, select_end=dates[3])
    base_row = baseline.query("scenario == 'fon-baz' and period == 'full'").iloc[0]
    changed_row = later_changed.query("scenario == 'fon-baz' and period == 'full'").iloc[0]

    units = 100.0 / 100.0
    expected_tax = 0.10 * max(units * 110.0 - 100.0, 0.0)
    expected_receivable = units * 110.0 - expected_tax
    expected_cagr = (expected_receivable / 100.0) ** (252 / (len(dates) - 1)) - 1

    assert base_row["status"] == changed_row["status"] == "ok"
    assert np.isclose(base_row["taxes_paid"], expected_tax)
    assert np.isclose(base_row["receivables_final"], expected_receivable)
    assert np.isclose(base_row["ending_equity"], expected_receivable)
    assert np.isclose(base_row["cagr"], expected_cagr)
    assert np.isclose(changed_row["taxes_paid"], base_row["taxes_paid"])
    assert np.isclose(changed_row["cagr"], base_row["cagr"])
    assert base_row["fees_paid"] == 0
    assert base_row["remaining_lots"] == 0
    assert base_row["settle_idx"] == len(dates)


def test_b0_profile_issue_status_names_fund_field_and_date():
    """Eksik/geçersiz üye profili tüm satırları düşürürken neden fon/alan/tarih bazında kanıtlanmalı."""
    dates = pd.bdate_range("2026-09-22", periods=8)
    nav = pd.DataFrame(
        {"P1": np.linspace(1.0, 1.08, len(dates)), "P2": np.linspace(2.0, 2.1, len(dates))},
        index=dates,
    )
    codes = ["P1", "P2"]
    profiles = _execution_profiles(dates, codes)
    profiles.loc[(dates[0], "P1"), "sell_valor"] = np.nan
    profiles.loc[(dates[-1], "P2"), "tax_category"] = "   "

    rows = b0_fund_valor_stress(
        nav,
        codes,
        meta_for_synthetic(codes, buy_valor=0, sell_valor=1, tax_rate=0.20),
        profiles,
        select_end=dates[3],
    )

    statuses = set(rows["status"])
    assert len(statuses) == 1
    status = statuses.pop()
    assert status.startswith("değerlendirilemedi: ilk alım/son satış gününde profil eksik veya geçersiz [")
    assert "P1" in status and "sell_valor" in status and str(dates[0].date()) in status
    assert "P2" in status and "tax_category" in status and str(dates[-1].date()) in status
    assert rows["cagr"].isna().all()
    assert rows["fees_paid"].isna().all()


@pytest.mark.parametrize("missing_at", ["first", "terminal"])
@pytest.mark.parametrize("blank", [False, True])
def test_b0_missing_or_blank_tax_category_makes_fund_lot_stress_unavailable(missing_at: str, blank: bool):
    dates = pd.bdate_range("2026-09-22", periods=8)
    nav = pd.DataFrame({"P1": np.linspace(1.0, 1.08, len(dates))}, index=dates)
    profiles = _execution_profiles(dates, ["P1"])
    day = dates[0] if missing_at == "first" else dates[-1]
    if blank:
        profiles.loc[(day, "P1"), "tax_category"] = "  "
    else:
        profiles.loc[(day, "P1"), "tax_category"] = np.nan

    rows = b0_fund_valor_stress(
        nav,
        ["P1"],
        meta_for_synthetic(["P1"], buy_valor=0, sell_valor=1, tax_rate=0.20),
        profiles,
        select_end=dates[3],
    )

    assert rows["status"].str.startswith("değerlendirilemedi:").all()
    assert rows["cagr"].isna().all()
    assert rows["taxes_paid"].isna().all()


def test_b0_valid_member_profiles_produce_numbers_with_zero_tefas_commission():
    """Geçerli profil varsa baz/+1 satırları sayısaldır; ham non-zero fee execution komisyonu değildir (ADR-27)."""
    dates = pd.bdate_range("2026-09-22", periods=8)
    nav = pd.DataFrame(
        {"P1": np.linspace(1.0, 1.08, len(dates)), "P2": np.linspace(2.0, 2.04, len(dates))},
        index=dates,
    )
    codes = ["P1", "P2"]
    profiles = _execution_profiles(dates, codes)
    profiles["entry_fee"] = 0.05
    profiles["exit_fee"] = 0.07

    rows = b0_fund_valor_stress(
        nav,
        codes,
        meta_for_synthetic(codes, buy_valor=0, sell_valor=1, tax_rate=0.20),
        profiles,
        select_end=dates[4],
    )
    full = rows.query("period == 'full'").set_index("scenario")
    for scenario in ("fon-baz", "satış valörü +1"):
        row = full.loc[scenario]
        assert row["status"] == "ok"
        assert pd.notna(row["cagr"])
        assert row["fees_paid"] == 0
        assert row["taxes_paid"] > 0
    assert full.loc["satış valörü +1", "settle_idx"] == len(dates) + 1


def test_b0_fixed_terminal_full_only_reports_net_fund_cagr_and_periods_unavailable():
    dates = pd.bdate_range("2026-09-22", periods=8)
    nav = pd.DataFrame(
        {"P1": np.linspace(1.0, 1.08, len(dates)), "P2": np.linspace(2.0, 2.04, len(dates))},
        index=dates,
    )
    codes = ["P1", "P2"]
    profiles = _execution_profiles(dates, codes)

    rows = b0_fund_valor_stress(
        nav,
        codes,
        meta_for_synthetic(codes, buy_valor=0, sell_valor=1, tax_rate=0.20),
        profiles,
        select_end=str(dates[4].date()),
    )

    base_select = rows.query("scenario == 'fon-baz' and period == 'select'").iloc[0]
    stress_select = rows.query("scenario == 'satış valörü +1' and period == 'select'").iloc[0]
    base_external = rows.query("scenario == 'fon-baz' and period == 'external'").iloc[0]
    stress_external = rows.query("scenario == 'satış valörü +1' and period == 'external'").iloc[0]
    full_base = rows.query("scenario == 'fon-baz' and period == 'full'").iloc[0]
    full_stress = rows.query("scenario == 'satış valörü +1' and period == 'full'").iloc[0]
    for row in (base_select, stress_select, base_external, stress_external):
        assert np.isnan(row["cagr"])
        assert row["status"] == "değerlendirilemedi: dönem sonu bağımsız tasfiye/yeniden giriş yok"
    assert base_select["assumption_label"] == "PIT yürütme profili"
    assert full_base["status"] == full_stress["status"] == "ok"
    assert full_base["settle_idx"] == len(dates)  # terminal sale at final index + T+1
    assert full_stress["settle_idx"] == len(dates) + 1
    assert np.isclose(full_base["cagr"], full_stress["cagr"])
    gross_terminal_multiple = np.mean(nav.iloc[-1].to_numpy() / nav.iloc[0].to_numpy())
    gross_cagr = gross_terminal_multiple ** (252 / (len(dates) - 1)) - 1
    assert full_base["cagr"] < gross_cagr  # entry/exit fees and realized tax are in the net equity
    assert full_base["fees_paid"] == 0 and full_base["taxes_paid"] > 0
    assert "dönem sonu tasfiye" in full_base["event_note"]


def test_b0_explicit_synthetic_reinvestment_misses_plus_one_day_of_positive_cash_return():
    dates = pd.bdate_range("2026-09-22", periods=8)
    nav = pd.DataFrame(
        {"P1": np.linspace(1.0, 1.08, len(dates)), "P2": np.linspace(2.0, 2.04, len(dates))},
        index=dates,
    )
    codes = ["P1", "P2"]
    profiles = _execution_profiles(dates, codes)
    cash_returns = pd.Series(0.01, index=dates)

    rows = b0_fund_valor_stress(
        nav,
        codes,
        meta_for_synthetic(codes, buy_valor=0, sell_valor=1, tax_rate=0.20),
        profiles,
        select_end=str(dates[4].date()),
        synthetic_liquidation_idx=2,
        cash_returns=cash_returns,
    )

    sensitivity = rows.query("mode == 'synthetic_reinvestment'").set_index("scenario")
    base = sensitivity.loc["fon-baz"]
    plus_one = sensitivity.loc["satış valörü +1"]
    assert base["settle_idx"] == 3
    assert plus_one["settle_idx"] == 4
    assert plus_one["cagr"] < base["cagr"]
    assert "varsayımsal yeniden yatırım" in plus_one["event_note"]
    assert "vergili günlük nakit yaklaşımı" in plus_one["event_note"]
    assert "sentetik" in plus_one["assumption_label"].lower()
    assert "gerçek fon-lot nakit getirisi değildir" in plus_one["assumption_label"]


def test_b0_synthetic_reinvestment_zero_returns_preserves_ledger_identity_and_audit_fields():
    dates = pd.bdate_range("2026-09-22", periods=8)
    nav = pd.DataFrame(
        {"P1": np.linspace(1.0, 1.08, len(dates)), "P2": np.linspace(2.0, 2.04, len(dates))},
        index=dates,
    )
    codes = ["P1", "P2"]
    profiles = _execution_profiles(dates, codes)

    rows = b0_fund_valor_stress(
        nav,
        codes,
        meta_for_synthetic(codes, buy_valor=0, sell_valor=1, tax_rate=0.20),
        profiles,
        select_end=str(dates[4].date()),
        synthetic_liquidation_idx=2,
        cash_returns=pd.Series(0.0, index=dates),
    )

    sensitivity = rows.query("mode == 'synthetic_reinvestment'").set_index("scenario")
    base = sensitivity.loc["fon-baz"]
    plus_one = sensitivity.loc["satış valörü +1"]
    assert np.isclose(base["cagr"], plus_one["cagr"])
    assert base["settle_idx"] != plus_one["settle_idx"]
    for row in (base, plus_one):
        assert row["equity_identity_max_abs"] < 1e-9
        assert np.isclose(
            row["ending_equity"], row["settled_cash"] + row["receivables_final"] + row["position_value_final"]
        )
        assert row["remaining_lots"] == 0
        assert row["units_available_at_liquidation"] > 0
        assert row["sellable_units"] == 0
        assert row["fees_paid"] == 0
        assert row["taxes_paid"] > 0
        assert row["settled_cash"] > 0


def test_b0_receivables_do_not_accrue_the_return_on_their_settlement_day():
    dates = pd.bdate_range("2026-09-22", periods=8)
    nav = pd.DataFrame({"P1": np.ones(len(dates)), "P2": np.full(len(dates), 2.0)}, index=dates)
    returns = pd.Series(0.0, index=dates)
    returns.iloc[3] = 0.01  # base settlement day; +1 scenario still holds a receivable

    rows = b0_fund_valor_stress(
        nav,
        ["P1", "P2"],
        meta_for_synthetic(["P1", "P2"], buy_valor=0, sell_valor=1, tax_rate=0.20),
        _execution_profiles(dates, ["P1", "P2"]),
        select_end=str(dates[4].date()),
        synthetic_liquidation_idx=2,
        cash_returns=returns,
    )
    sensitivity = rows.query("mode == 'synthetic_reinvestment'").set_index("scenario")
    base = sensitivity.loc["fon-baz"]
    plus_one = sensitivity.loc["satış valörü +1"]
    assert base["settle_idx"] == 3 and plus_one["settle_idx"] == 4
    assert np.isclose(base["cagr"], plus_one["cagr"])
