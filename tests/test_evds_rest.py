import pandas as pd

from janus.data.ingest_evds import EvdsRestMacro, FakeMacro, ingest_macro
from janus.data.store import Store


def test_rest_parse_json():
    payload = {
        "totalCount": 3,
        "items": [
            {"Tarih": "01-09-2026", "TP_DK_USD_A": "41.25", "TP_FG_J0": None, "UNIXTIME": {"$numberLong": "1"}},
            {"Tarih": "02-09-2026", "TP_DK_USD_A": "41.30", "TP_FG_J0": None, "UNIXTIME": {"$numberLong": "2"}},
            {"Tarih": "03-09-2026", "TP_DK_USD_A": None, "TP_FG_J0": "2500.5", "UNIXTIME": {"$numberLong": "3"}},
        ],
    }
    wide = EvdsRestMacro.parse(payload, ["TP.DK.USD.A", "TP.FG.J0"])
    assert list(wide.columns) == ["TP.DK.USD.A", "TP.FG.J0"]
    assert wide.index.name == "Date" and wide.index[0] == pd.Timestamp("2026-09-01")
    assert wide["TP.DK.USD.A"].iloc[1] == 41.30 and wide["TP.FG.J0"].iloc[2] == 2500.5


class BrokenMacro:
    def evds(self, codes, period, frequency):
        raise ConnectionError("simulated")

    def policy_rate_history(self, period):
        raise ConnectionError("simulated")


def test_fallback_fetcher_used(tmp_path, cfg):
    st = Store(tmp_path / "t.duckdb")
    res = ingest_macro(st, [BrokenMacro(), FakeMacro()], cfg)
    assert res["status"] == "ok" and res["n_rows"] > 0


def test_all_fetchers_fail_is_recorded(tmp_path, cfg):
    st = Store(tmp_path / "t.duckdb")
    res = ingest_macro(st, [BrokenMacro()], cfg)
    assert res["status"] == "failed" and len(res["failures"]) == 4
