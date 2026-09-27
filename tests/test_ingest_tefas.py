import pandas as pd
import pytest

from janus.config import load_config
from janus.data.ingest_tefas import derive_founder, history_to_long, ingest_tefas
from janus.data.quality import flag_data_stale, quality_summary, trade_status_masks
from janus.data.store import Store
from janus.data.tefas_client import FakeTefasClient
from janus.portfolio.universe import phase1_mask


@pytest.fixture
def cfg_tefas_only():
    """TEFAS ingest testleri EMK çekmeden çalışsın (regresyon koruması)."""
    c = load_config()
    c["ingest"]["fund_types"] = ["YAT"]
    return c


def test_derive_founder():
    s = pd.Series(
        [
            "POYRAZ PORTFÖY BİRİNCİ HİSSE SENEDİ FONU",
            "Kuzey Portföy Yönetimi A.Ş. Fon Sepeti Fonu",
            "Ege Finans Portföy X",
            "Bir Şey Fonu",
        ]
    )
    out = derive_founder(s).tolist()
    assert (
        out[0] == "POYRAZ PORTFÖY"
        and out[1] == "Kuzey Portföy"
        and out[2] == "Ege Finans Portföy"
        and out[3] == "Bir Şey"
    )


def test_initial_then_incremental_is_idempotent(tmp_path, cfg_tefas_only):
    st = Store(tmp_path / "t.duckdb")
    client = FakeTefasClient(n=12, days=400)
    r1 = ingest_tefas(st, client, cfg_tefas_only, mode="initial", snapshot_date="2026-09-22")
    assert r1["status"] == "partial"  # F05 info hatası simüle edildi
    assert r1["n_processed"] == 12 and r1["n_hist_ok"] == 12 and r1["n_info_ok"] == 11
    n_rows_1 = st.con.execute("SELECT count(*) FROM fund_nav").fetchone()[0]
    r2 = ingest_tefas(st, client, cfg_tefas_only, mode="incremental", snapshot_date="2026-09-22")
    n_rows_2 = st.con.execute("SELECT count(*) FROM fund_nav").fetchone()[0]
    assert n_rows_2 == n_rows_1  # upsert: satır sayısı değişmez
    fm = st.latest_fund_master()
    assert len(fm) == 12 and fm["snapshot_date"].nunique() == 1  # aynı gün snapshot tekil
    assert fm.loc[fm.fund_code == "F07", "n_nav"].item() > 0
    assert pd.Timestamp(fm.set_index("fund_code").loc["F07", "first_nav_date"]) > pd.Timestamp(
        fm.set_index("fund_code").loc["F00", "first_nav_date"]
    )  # genç fon
    assert r2["mode"] == "incremental" and st.last_run("ingest_tefas")["summary"]["mode"] == "incremental"


def test_universe_and_quality_end_to_end(tmp_path, cfg_tefas_only):
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=400), cfg_tefas_only, mode="initial", snapshot_date="2026-09-22")
    wide = st.nav_wide()
    fm = st.latest_fund_master()
    cfg_tefas_only["legs"]["tefas"]["universe"]["min_universe_fresh_ratio"] = 0.9  # 12 fonluk sentetik evren
    q = quality_summary(wide, fm, cfg_tefas_only, asof="2026-09-23 09:00")  # panel son günü 22.09 (Sal) → taze
    assert q["ok"] and not q["source_stale"]
    # F03 3 günlük NAV boşluğu var; hafta sonu kesişince bazen 1 iş günü geride görünür.
    assert q["n_data_stale"] in (0, 1)
    if q["n_data_stale"]:
        assert q["data_stale_codes"] == ["F03"]
    assert q["n_suspended"] == 0  # F09 "İşlem Görmüyor" askı değil
    stale = flag_data_stale(wide.tail(30), 2)
    can_buy, _ = trade_status_masks(fm)
    m = phase1_mask(fm, cfg_tefas_only, asof="2026-09-22", suspended=stale, can_buy=can_buy)
    # dışarı: serbest (F01,F05,F09), Polaris (F02,F08), Kuzey (F03 zaten askıda, F09 serbest), genç F07, F03 askıda, F09 durum kapalı
    assert not m["F01"] and not m["F02"] and not m["F03"] and not m["F07"] and not m["F08"] and not m["F09"]
    assert m["F00"] and m["F04"] and m["F06"] and m["F10"]
    assert m.sum() == 5


def test_valor_and_fee_parsing(tmp_path, cfg_tefas_only):
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=6, days=300), cfg_tefas_only, mode="initial", snapshot_date="2026-09-22")
    fm = st.latest_fund_master().set_index("fund_code")
    assert fm.loc["F01", "buy_valor"] == 1 and fm.loc["F03", "sell_valor"] == 2 and fm.loc["F01", "exit_fee"] == 0
    assert fm.loc["F00", "founder"] == "POYRAZ PORTFÖY" and fm.loc["F02", "founder_code"] == "POL"
    assert fm["hist_ok"].all() and bool(fm.loc["F05", "info_ok"]) is False


def test_f10_last_success_at_is_recorded_and_preserved_on_carry(tmp_path, cfg_tefas_only, monkeypatch):
    st = Store(tmp_path / "f10.duckdb")
    client = FakeTefasClient(n=6, days=300)
    ingest_tefas(st, client, cfg_tefas_only, mode="initial", snapshot_date="2026-09-22")
    first = st.latest_fund_master().set_index("fund_code")
    assert pd.notna(first.loc["F00", "last_success_at"])
    assert first.loc["F00", "last_success_source"] == "fetch"
    success_time = first.loc["F00", "last_success_at"]

    def fail_profile(_code):
        raise RuntimeError("synthetic profile failure")

    monkeypatch.setattr(client, "fund_info", fail_profile)
    ingest_tefas(st, client, cfg_tefas_only, mode="incremental", snapshot_date="2026-09-23")
    carried = st.latest_fund_master().set_index("fund_code")
    assert carried.loc["F00", "last_success_at"] == success_time
    assert carried.loc["F00", "last_success_source"] == "fetch"
    assert not bool(carried.loc["F00", "info_fetched"])


def test_profile_success_backfill_is_idempotent_and_only_updates_successful_rows(tmp_path, cfg_tefas_only):
    st = Store(tmp_path / "backfill.duckdb")
    ingest_tefas(st, FakeTefasClient(n=6, days=300), cfg_tefas_only, mode="initial", snapshot_date="2026-09-22")
    st.con.execute("UPDATE fund_master SET last_success_at=NULL WHERE info_fetched IS TRUE")
    assert st.backfill_profile_success_at(dry_run=True) == 5
    assert st.con.execute("SELECT count(*) FROM fund_master WHERE last_success_at IS NOT NULL").fetchone()[0] == 0
    changed = st.backfill_profile_success_at()
    assert changed == 5
    assert (
        st.con.execute(
            "SELECT count(*) FROM fund_master WHERE info_fetched IS TRUE AND last_success_at IS NULL"
        ).fetchone()[0]
        == 0
    )
    assert st.backfill_profile_success_at() == 0
    assert set(
        st.con.execute("SELECT DISTINCT last_success_source FROM fund_master WHERE info_fetched IS TRUE").df()[
            "last_success_source"
        ]
    ) == {"proxy_ingest_time"}
    assert (
        st.con.execute(
            "SELECT count(*) FROM fund_master WHERE info_fetched IS FALSE AND last_success_at IS NOT NULL"
        ).fetchone()[0]
        == 0
    )
    st.close()


def test_f10_nav_ingest_excludes_nan_and_nonpositive_prices():
    hist = pd.DataFrame(
        {"Price": [1.0, float("nan"), 0.0, -1.0, 2.0]},
        index=pd.date_range("2026-09-21", periods=5),
    )

    nav = history_to_long("F00", hist, pd.Timestamp("2026-09-25").to_pydatetime())

    assert nav["price"].tolist() == [1.0, 2.0]
    assert nav["price"].notna().all() and nav["price"].gt(0).all()


def test_trade_status_masks_keep_legacy_blank_open_but_strict_masks_fail_closed():
    from janus.data.quality import strict_trade_status_masks

    fm = pd.DataFrame(
        {
            "fund_code": ["OPEN", "BUY_CLOSED", "SELL_CLOSED", "BOTH", "NOT_TRADING", "NULL", "BLANK", "UNKNOWN"],
            "tefas_status": [
                "İşlem Görüyor",
                "Fon Alımına Kapalı, Fon Bozumuna Açık",
                "Fon Alımına Açık, Fon Bozumuna Kapalı",
                "Alımına ve Bozumuna Kapalı",
                "İşlem Görmüyor",
                None,
                "   ",
                "bilinmeyen",
            ],
        }
    )
    legacy_buy, legacy_sell = trade_status_masks(fm)
    assert legacy_buy["NULL"] and legacy_sell["BLANK"]

    can_buy, can_sell, known = strict_trade_status_masks(fm)
    assert can_buy["OPEN"] and can_sell["OPEN"] and known["OPEN"]
    assert not can_buy["BUY_CLOSED"] and can_sell["BUY_CLOSED"]
    assert can_buy["SELL_CLOSED"] and not can_sell["SELL_CLOSED"]
    assert not can_buy["BOTH"] and not can_sell["BOTH"]
    assert not can_buy["NOT_TRADING"] and not can_sell["NOT_TRADING"]
    assert not known[["NULL", "BLANK", "UNKNOWN"]].any()
    assert not can_buy[["NULL", "BLANK", "UNKNOWN"]].any()
    assert not can_sell[["NULL", "BLANK", "UNKNOWN"]].any()


def test_ingested_buy_closed_status_preserves_sell_eligibility():
    from janus.data.ingest_tefas import build_fund_master

    listing = pd.DataFrame(
        [{"fund_code": "X", "name": "X FONU", "umbrella_type": "Para Piyasası", "fund_class": "YAT"}]
    )
    info = pd.DataFrame(
        [
            {
                "fund_code": "X",
                "isin": "TRX",
                "category": "Para Piyasası",
                "tefas_status": "Fon Alımına Kapalı, Fon Bozumuna Açık",
                "fund_size": 1,
                "investor_count": 1,
                "risk_value": 1,
                "buy_valor": 0,
                "sell_valor": 0,
                "entry_fee": 0,
                "exit_fee": 0,
            }
        ]
    )
    result = build_fund_master(
        listing,
        pd.DataFrame(columns=["fund_code", "founder_code", "applied_fee", "max_expense_ratio"]),
        info,
        pd.DataFrame(columns=["fund_code", "first_nav_date", "last_nav_date", "n_nav"]),
        pd.Timestamp("2026-09-25"),
        pd.Timestamp("2026-09-25"),
    )
    assert not bool(result.loc[0, "can_buy"])
    assert bool(result.loc[0, "can_sell"])
