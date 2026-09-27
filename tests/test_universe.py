import pandas as pd

from janus.portfolio.universe import phase1_mask


def test_phase1_filters(fund_master, cfg):
    m = phase1_mask(fund_master, cfg, asof=pd.Timestamp("2026-09-21"))
    assert m["AAA"]  # YAT, hisse şemsiye, temiz kurucu, yaş > 1y
    assert not m["BBB"]  # Serbest şemsiye → dışarı
    assert not m["CCC"]  # kurucu kara listede (Polaris)
    assert not m["DDD"]  # EMK → Faz-1 dışı
    assert not m["EEE"]  # yönetici kara listede (Kuzey)
    assert not m["FFF"]  # yaş < 365 gün


def test_whitelist_applies(fund_master, cfg):
    cfg["legs"]["tefas"]["universe"]["founder_whitelist"] = ["Lodos Portföy"]
    m = phase1_mask(fund_master, cfg, asof=pd.Timestamp("2026-09-21"))
    assert m.sum() == 1 and m["AAA"]


def test_status_pattern_excludes(fund_master, cfg):
    fm = fund_master.copy()
    fm["tefas_status"] = ["İşlem Görüyor", "", "", "", "", ""]
    fm.loc[0, "tefas_status"] = "İşlem Görmüyor"
    assert not phase1_mask(fm, cfg, asof=pd.Timestamp("2026-09-21"))["AAA"]
