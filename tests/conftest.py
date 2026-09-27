import numpy as np
import pandas as pd
import pytest

from janus.config import load_config


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def nav_wide():
    idx = pd.bdate_range("2026-09-01", periods=10)
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.lognormal(0, 0.01, size=(10, 4)).cumprod(axis=0) * 10, index=idx, columns=list("ABCD"))
    df.loc[idx[-3:], "C"] = np.nan  # C: 3 gündür NAV yok → askıda
    df.loc[idx[-1], "D"] = np.nan  # D: 1 gün → normal gecikme
    return df


@pytest.fixture
def fund_master():
    """S0 sentetik fund_master (inception_date ile yaş)."""
    return pd.DataFrame(
        {
            "fund_code": ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
            "fund_class": ["YAT", "YAT", "YAT", "EMK", "YAT", "YAT"],
            "umbrella_type": [
                "Hisse Senedi Şemsiye Fonu",
                "Serbest Şemsiye Fonu",
                "Para Piyasası Şemsiye Fonu",
                "Hisse Senedi Emeklilik",
                "Değişken Şemsiye Fonu",
                "Karma Şemsiye Fonu",
            ],
            "name": [
                "LODOS PORTFÖY HİSSE FONU",
                "POYRAZ PORTFÖY SERBEST FON",
                "POLARİS PORTFÖY PARA PİYASASI FONU",
                "ANADOLU HAYAT EMEKLİLİK HİSSE",
                "GÜNDOĞAN PORTFÖY DEĞİŞKEN FON",
                "BATI PORTFÖY KARMA FON",
            ],
            "founder": [
                "Lodos Portföy",
                "Poyraz Portföy",
                "Polaris Portföy Yönetimi A.Ş.",
                "Marmara Hayat",
                "Gündoğan Portföy",
                "Batı Portföy",
            ],
            "manager": [
                "Lodos Portföy",
                "Poyraz Portföy",
                "Polaris Portföy",
                "Marmara Hayat",
                "Kuzey Portföy",
                "Batı Portföy",
            ],
            "inception_date": ["2015-01-01", "2018-01-01", "2019-01-01", "2012-01-01", "2016-01-01", "2026-06-01"],
        }
    )
