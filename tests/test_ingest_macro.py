from janus.data.ingest_evds import FakeMacro, ingest_macro
from janus.data.store import Store


def test_macro_ingest_and_pit_view(tmp_path, cfg):
    st = Store(tmp_path / "t.duckdb")
    res = ingest_macro(st, FakeMacro(), cfg)
    assert res["status"] == "ok" and set(res["counts"]) == {"usdtry", "eurtry", "cpi_index", "policy_rate"}
    wide_all = st.macro_wide()
    assert {"usdtry", "cpi_index", "policy_rate"} <= set(wide_all.columns)
    # PIT: TÜFE 35 gün gecikmeli → 22.09 itibarıyla Ağustos (01.08+35g=05.09) görünür, Eylül görünmez
    wide_asof = st.macro_wide(asof="2026-09-22")
    cpi = wide_asof["cpi_index"].dropna()
    assert cpi.index.max().strftime("%Y-%m") == "2026-08"
    # idempotent
    n1 = st.con.execute("SELECT count(*) FROM macro").fetchone()[0]
    ingest_macro(st, FakeMacro(), cfg)
    assert st.con.execute("SELECT count(*) FROM macro").fetchone()[0] == n1
