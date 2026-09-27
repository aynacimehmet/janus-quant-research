import pandas as pd

from janus.portfolio.universe import phase1_mask, tr_fold


def test_tr_fold():
    assert tr_fold("ALFA CAPİTAL PORTFÖY") == "alfa capital portföy"
    assert tr_fold("IŞIK") == "ışık" and tr_fold("LODOS PORTFÖY") == "lodos portföy"


def test_blacklist_catches_turkish_capital_i(cfg):
    fm = pd.DataFrame(
        {
            "fund_code": ["X1", "X2"],
            "fund_class": ["YAT", "YAT"],
            "umbrella_type": ["Hisse Senedi Şemsiye Fonu"] * 2,
            "category": ["Hisse Senedi Fonu"] * 2,
            "name": ["ALFA CAPİTAL PORTFÖY HİSSE SENEDİ FONU", "LODOS PORTFÖY HİSSE SENEDİ FONU"],
            "founder": ["ALFA CAPİTAL PORTFÖY", "LODOS PORTFÖY"],
            "tefas_status": ["TEFAS'ta işlem görüyor"] * 2,
            "first_nav_date": ["2021-01-04"] * 2,
        }
    )
    m = phase1_mask(fm, cfg, asof="2026-09-22")
    assert not m["X1"] and m["X2"]


def test_status_vocabulary(cfg):
    base = {
        "fund_class": "YAT",
        "umbrella_type": "Değişken Şemsiye Fonu",
        "category": "Değişken Fon",
        "founder": "POYRAZ PORTFÖY",
        "name": "POYRAZ PORTFÖY DEĞİŞKEN FON",
        "first_nav_date": "2020-01-02",
    }
    statuses = [
        "TEFAS'ta İşlem Görmüyor",
        "TEFAS'ta işlem görüyor",
        "Fon Alımına Kapalı, Fon Bozumuna Kapalı",
        "Fon Alımına Kapalı, Fon Bozumuna Açık",
        None,
    ]
    fm = pd.DataFrame([{**base, "fund_code": f"S{i}", "tefas_status": st} for i, st in enumerate(statuses)])
    m = phase1_mask(fm, cfg, asof="2026-09-22")
    assert m.tolist() == [
        False,
        True,
        False,
        False,
        True,
    ]  # None durum → dışlanmaz (profil hatası; askı tespiti ayrıca bakar)
