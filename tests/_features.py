"""S3b-2 test yardımcısı: sentetik feature paneli (S3b-1 build_features üzerinden)."""

from __future__ import annotations

import pandas as pd

from _panel import make_panel
from janus.features.fund_features import build_features


def make_fm(codes, extra=None):
    fm = pd.DataFrame(
        {
            "fund_code": codes,
            "umbrella_type": "Hisse",
            "withholding_rate": 0.175,
            "tefas_status": "",
            "founder": "K",
            "manager": "",
            "name": "Fon " + pd.Series(codes),
        }
    )
    if extra:
        for k, v in extra.items():
            fm[k] = v
    return fm


def make_features(n_funds=6, days=500, seed=0):
    nav = make_panel(n_funds=n_funds, days=days, seed=seed)
    cal = nav.index
    macro = pd.DataFrame(
        {
            "policy_rate": 0.4,
            "d_policy_63": 0.0,
            "cpi_yoy": 0.3,
            "real_rate": 0.1,
            "usdtry_ret63": 0.05,
            "usdtry_vol21": 0.1,
        },
        index=cal,
    )
    market = pd.DataFrame({"eq_trend63": 0.1, "eq_vol21": 0.2, "breadth200": 0.5}, index=cal)
    return build_features(nav, make_fm(list(nav.columns)), pd.Series(0.0002, index=cal), macro, market)
