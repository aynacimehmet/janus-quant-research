import pandas as pd

from janus.data.quality import (
    execution_status_masks,
    expected_last_nav_date,
    flag_bad_nav,
    flag_data_stale,
    flag_suspended_status,
    quality_summary,
    stale_days,
    trade_status_masks,
)


def test_execution_status_masks_ignore_persisted_flags_and_report_mismatch():
    frame = pd.DataFrame(
        {
            "fund_code": ["OPEN", "BUY_CLOSED", "BUY_OPEN", "UNKNOWN"],
            "tefas_status": [
                "İşlem Görüyor",
                "Fon Alımına Kapalı, Fon Bozumuna Açık",
                "Fon Alımına Açık",
                "tanınmayan durum",
            ],
            "can_buy": [False, True, False, True],
            "can_sell": [None, False, True, True],
        }
    )

    buy, sell, known, mismatch = execution_status_masks(frame)

    assert buy.tolist() == [True, False, True, False]
    assert sell.tolist() == [True, True, False, False]
    assert known.tolist() == [True, True, True, False]
    assert mismatch.tolist() == [True, True, True, True]


def test_stale_days(nav_wide):
    sd = stale_days(nav_wide)
    assert sd["A"] == 0 and sd["C"] == 3 and sd["D"] == 1


def test_flag_data_stale_is_not_suspension(nav_wide):
    s = flag_data_stale(nav_wide, max_stale_days=2)
    assert bool(s["C"]) and not bool(s["D"]) and not bool(s["A"]) and s.name == "data_stale"


def test_flag_bad_nav(nav_wide):
    bad = nav_wide.copy()
    bad.iloc[2, 0] = 0.0
    bad.iloc[5, 1] = bad.iloc[4, 1] * 1.5
    flags = flag_bad_nav(bad)
    assert flags["nonpositive"].any() and flags["jump"].any()


def test_expected_last_nav_date_calendar():
    assert expected_last_nav_date("2026-09-23 09:15") == pd.Timestamp("2026-09-22")  # yayın saatinden önce → dün
    assert expected_last_nav_date("2026-09-23 10:30") == pd.Timestamp("2026-09-23")  # yayın sonrası → bugün
    assert expected_last_nav_date("2026-09-21 09:00") == pd.Timestamp("2026-09-18")  # Pzt sabah → Cum
    assert expected_last_nav_date("2026-09-19 12:00") == pd.Timestamp("2026-09-18")  # Cmt → Cum
    cfg = {"calendar": {"holidays": ["2026-09-22"], "nav_publish_time": "10:00"}}
    assert expected_last_nav_date("2026-09-23 09:00", cfg) == pd.Timestamp("2026-09-21")


def test_trade_status_masks_and_suspended():
    fm = pd.DataFrame(
        {
            "fund_code": ["G", "N", "BB", "BA", "X"],
            "tefas_status": [
                "TEFAS'ta işlem görüyor",
                "TEFAS'ta İşlem Görmüyor",
                "Fon Alımına Kapalı, Fon Bozumuna Kapalı",
                "Fon Alımına Kapalı, Fon Bozumuna Açık",
                None,
            ],
        }
    )
    can_buy, can_sell = trade_status_masks(fm)
    assert can_buy.tolist() == [True, False, False, False, True]
    assert can_sell.tolist() == [True, False, False, True, True]
    susp = flag_suspended_status(fm)
    assert susp.tolist() == [False, False, True, False, False]  # yalnızca alım VE bozum kapalı


def _cfg():
    return {
        "legs": {"tefas": {"universe": {"max_stale_days": 2, "min_universe_fresh_ratio": 0.5, "source_stale_days": 2}}},
        "calendar": {"holidays": []},
    }


def _fm(codes):
    return pd.DataFrame(
        {
            "fund_code": codes,
            "tefas_status": ["TEFAS'ta işlem görüyor"] * len(codes),
            "info_ok": [True] * len(codes),
            "hist_ok": [True] * len(codes),
        }
    )


def test_source_stale_when_whole_panel_old(nav_wide):
    # panel 2026-09-01..2026-09-14; karar 2026-09-18 (Cum) → beklenen 17 Eyl; panel 3 iş günü geride
    q = quality_summary(nav_wide, _fm(list("ABCD")), _cfg(), asof="2026-09-18 12:00")
    assert q["source_stale"] and not q["ok"] and q["fresh_days_behind"] == 4  # beklenen 18 Eyl, panel 14 Eyl
    assert q["fresh_ratio"] >= 0.5  # fonlar kendi içinde taze olabilir; kaynak yine de eski (D04)


def test_fresh_when_panel_matches_reference(nav_wide):
    q = quality_summary(
        nav_wide, _fm(list("ABCD")), _cfg(), asof="2026-09-15 09:00"
    )  # Salı sabah → beklenen 14 Eyl (Pzt)
    assert not q["source_stale"] and q["fresh_days_behind"] == 0
    assert q["n_data_stale"] == 1 and q["data_stale_codes"] == ["C"] and q["n_suspended"] == 0


def test_fresh_ratio_uses_eligible_only(nav_wide):
    elig = pd.Series({"A": True, "B": True, "C": False, "D": False})
    q = quality_summary(nav_wide, _fm(list("ABCD")), _cfg(), asof="2026-09-15 09:00", eligible=elig)
    assert q["n_eligible"] == 2 and q["fresh_ratio"] == 1.0 and q["ok"]


def test_suspended_scope_eligible_or_held(nav_wide):
    fm = _fm(list("ABCD"))
    fm.loc[fm.fund_code == "D", "tefas_status"] = "Fon Alımına Kapalı, Fon Bozumuna Kapalı"
    elig = pd.Series({"A": True, "B": True, "C": True, "D": False})
    q0 = quality_summary(nav_wide, fm, _cfg(), asof="2026-09-15 09:00", eligible=elig)
    assert q0["n_suspended"] == 0  # D eligible değil, elde de yok
    q1 = quality_summary(nav_wide, fm, _cfg(), asof="2026-09-15 09:00", eligible=elig, held=["D"])
    assert q1["n_suspended"] == 1 and q1["suspended_codes"] == ["D"]  # elde tutulan askıdaki fon görünür
