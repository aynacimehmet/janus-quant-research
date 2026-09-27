"""S3b-1 testleri: gelecek perturbasyonu, beyaz liste, maske ayrımı, y formülü, CLI, performans."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from _panel import make_panel
from janus.cli import app
from janus.features.fund_features import FEATURE_COLUMNS, build_features, validate_columns

CAL = pd.bdate_range("2023-01-02", periods=800)


def make_macro(cal, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "policy_rate": 0.4 + rng.normal(0, 0.001, len(cal)).cumsum() * 0.01,
            "d_policy_63": 0.0,
            "cpi_yoy": 0.3,
            "real_rate": 0.1,
            "usdtry_ret63": 0.05,
            "usdtry_vol21": 0.1,
        },
        index=cal,
    )


def make_market(cal, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"eq_trend63": rng.normal(0, 0.1, len(cal)), "eq_vol21": 0.2, "breadth200": 0.5}, index=cal)


def make_fm(codes, status="", umbrella="Hisse", tax=0.175, extra=None):
    fm = pd.DataFrame(
        {
            "fund_code": codes,
            "fund_class": "YAT",
            "umbrella_type": umbrella,
            "withholding_rate": tax,
            "tefas_status": status,
            "founder": "Kurucu A",
            "manager": "",
            "name": "Fon " + pd.Series(codes),
        }
    )
    if extra:
        for k, v in extra.items():
            fm[k] = v
    return fm


def test_y_formula_manual():
    # Elle hesap: t=5 için y = log(NAV[t+22]/NAV[t+1]) - Σ_{h=t+2..t+22} log(1+cash_h) (TEMPORAL §4)
    nav = pd.DataFrame({"F0": np.linspace(10, 12, 40)}, index=pd.bdate_range("2024-01-02", periods=40))
    cash = pd.Series(0.001, index=nav.index)
    df = build_features(nav, make_fm(["F0"]), cash, make_macro(nav.index), make_market(nav.index))
    row = df[df["feature_asof"] == nav.index[5]].iloc[0]
    ln = np.log(nav["F0"])
    expected = (ln.iloc[27] - ln.iloc[6]) - 21 * np.log1p(0.001)
    assert row["y"] == pytest.approx(expected)
    assert row["decision_at"] == nav.index[6]
    assert row["label_end"] == nav.index[27]
    assert row["label_available_at"] == nav.index[28]


def test_future_perturbation_bit_identical():
    nav = make_panel(n_funds=8, days=600, seed=3)
    t0 = 400
    base = build_features(
        nav,
        make_fm(list(nav.columns)),
        pd.Series(0.0002, index=nav.index),
        make_macro(nav.index),
        make_market(nav.index),
    )
    nav2 = nav.copy()
    nav2.iloc[t0 + 1 :] *= 1.01  # gelecek NAV perturbasyonu
    macro2 = make_macro(nav.index)
    macro2.iloc[t0 + 1 :] += 0.01
    fm2 = make_fm(list(nav.columns), extra={"aum_now": 1e9, "investor_count": 100})  # tarihçesiz meta (H02)
    pert = build_features(nav2, fm2, pd.Series(0.0002, index=nav.index), macro2, make_market(nav.index))
    cut = nav.index[t0]
    cols = [c for c in base.columns if c != "y"]  # y, geleceğe uzanan penceresi olan son 22 satırda değişir
    b = base[base["feature_asof"] <= cut][cols].reset_index(drop=True)
    p = pert[pert["feature_asof"] <= cut][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(b, p, check_exact=True)


def test_whitelist_rejects_unknown():
    with pytest.raises(ValueError, match="aum_now"):
        validate_columns(["log_ret_1", "aum_now"])
    validate_columns(FEATURE_COLUMNS)  # beyaz liste kendisi geçer


def test_masks_separate_inference_rows():
    nav = make_panel(n_funds=6, days=800, seed=4)
    df = build_features(
        nav,
        make_fm(list(nav.columns)),
        pd.Series(0.0002, index=nav.index),
        make_macro(nav.index),
        make_market(nav.index),
    )
    last = df[df["feature_asof"] >= nav.index[-22]]
    assert last["label_ready"].eq(False).all()
    assert last["y"].isna().all()  # çıkarım satırları y'siz (H03)
    assert last["feature_ready"].eq(True).all()  # X dolu
    mid = df[df["feature_asof"] == nav.index[400]]
    assert mid["label_ready"].eq(True).all() and mid["y"].notna().all()


def test_terminal_nav_row_maps_to_next_configured_session_but_has_no_label():
    cal = pd.bdate_range(end="2026-09-25", periods=300)
    nav = pd.DataFrame({"F0": np.linspace(10.0, 15.0, len(cal))}, index=cal)
    cfg = {"calendar": {"holidays": ["2026-09-28"]}, "legs": {"tefas": {"universe": {}}}}

    features = build_features(
        nav,
        make_fm(["F0"]),
        pd.Series(0.0002, index=cal),
        make_macro(cal),
        make_market(cal),
        cfg=cfg,
    )
    terminal = features.loc[features["feature_asof"] == cal[-1]].iloc[0]
    latest_mature_label = features.loc[features["feature_asof"] == cal[-23]].iloc[0]

    assert terminal["decision_at"] == pd.Timestamp("2026-09-29")
    assert pd.isna(terminal["label_end"])
    assert pd.isna(terminal["label_available_at"])
    assert pd.isna(terminal["y"])
    assert latest_mature_label["label_available_at"] == pd.Timestamp("2026-09-29")


def test_eligible_and_policy_flags():
    nav = make_panel(n_funds=4, days=600, seed=5)
    fm = make_fm(list(nav.columns), status=["", "Fon Alımına Kapalı", "", "TEFAS'ta İşlem Görmüyor"])
    cfg = {
        "legs": {
            "tefas": {"universe": {"founder_blacklist": ["Başka Kurucu"], "exclude_status_patterns": ["görmüyor"]}}
        }
    }
    df = build_features(
        nav, fm, pd.Series(0.0, index=nav.index), make_macro(nav.index), make_market(nav.index), cfg=cfg
    )
    last = df[df["feature_asof"] == nav.index[-1]].set_index("fund_code")
    assert last["eligible_at_decision"].all()  # NAV var, geçmiş >= 252, stale değil
    assert last.loc["F1", "policy_today_excluded"]  # alımına kapalı
    assert last.loc["F3", "policy_today_excluded"]  # işlem görmüyor
    assert not last.loc["F0", "policy_today_excluded"]
    # geçmiş 252'den az olan erken satırlar eligible değil
    early = df[df["feature_asof"] == nav.index[100]]
    assert not early["eligible_at_decision"].any()


def test_validate_columns_blocks_injected_feature():
    # beyaz liste dışı kolon üreticiye sızdırılamaz: build çıktısı yalnızca whitelisted + zaman/maske/y
    nav = make_panel(n_funds=3, days=300, seed=5)
    df = build_features(
        nav, make_fm(list(nav.columns)), pd.Series(0.0, index=nav.index), make_macro(nav.index), make_market(nav.index)
    )
    validate_columns(
        [
            c
            for c in df.columns
            if c not in FEATURE_COLUMNS
            and c
            not in (
                "feature_asof",
                "fund_code",
                "decision_at",
                "label_end",
                "label_available_at",
                "feature_ready",
                "label_ready",
                "eligible_at_decision",
                "policy_today_excluded",
                "y",
            )
        ]
    )


def test_performance_539x1000():
    import time

    nav = make_panel(n_funds=539, days=1000, seed=7)
    t = time.perf_counter()
    df = build_features(
        nav,
        make_fm(list(nav.columns)),
        pd.Series(0.0002, index=nav.index),
        make_macro(nav.index),
        make_market(nav.index),
    )
    dt = time.perf_counter() - t
    assert len(df) == 539 * 1000
    assert dt < 60.0, f"çok yavaş: {dt:.1f}s"


class FakeStore:
    """CLI testi için salt-okunur snapshot benzeri sahte Store."""

    snapshot_asof = pd.Timestamp("2026-09-23 18:00")

    def __init__(self, nav, fm):
        self._nav, self._fm = nav, fm

    def latest_fund_master(self):
        return self._fm

    def nav_wide(self):
        return self._nav

    def log_run(self, *a, **k):  # salt-okunur: yazma engeli
        raise RuntimeError("salt-okunur Store: yazma engellendi (log_run)")


def test_cli_features_build(tmp_path, monkeypatch):
    import janus.cli as cli

    nav = make_panel(n_funds=5, days=400, seed=9)
    fm = make_fm(list(nav.columns))
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: FakeStore(nav, fm))
    monkeypatch.setattr(cli, "load_config", lambda: {"legs": {"tefas": {"universe": {}}}, "store": {}})
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    import janus.features.macro as feat_macro

    monkeypatch.setattr(
        feat_macro, "macro_long", lambda store: pd.DataFrame(columns=["series", "date", "value", "available_from"])
    )
    result = CliRunner().invoke(app, ["features", "build"])
    assert result.exit_code == 0, result.output
    out = tmp_path / "data" / "features" / "fund_features.parquet"
    assert out.exists()
    df = pd.read_parquet(out)
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert (tmp_path / "data" / "features" / "build_summary.json").exists()  # runs yazılamadı → JSON
    # S3b-1b-ek: özet metrikleri — yeni anahtarlar, aralıklar, eski alanlar korunur
    import json as _json

    summary = _json.loads((tmp_path / "data" / "features" / "build_summary.json").read_text())
    for key in ("label_ready_ratio", "eligible_at_decision_ratio", "elapsed_seconds"):
        assert key in summary
    assert 0.0 <= summary["label_ready_ratio"] <= 1.0
    assert 0.0 <= summary["eligible_at_decision_ratio"] <= 1.0
    assert summary["elapsed_seconds"] >= 0.0
    for key in ("n_rows", "n_funds", "feature_asof_min", "feature_asof_max", "asof", "snapshot_asof"):
        assert key in summary
    assert summary["n_rows"] == len(df)
    # --asof kırpma
    result2 = CliRunner().invoke(app, ["features", "build", "--asof", str(nav.index[300].date())])
    assert result2.exit_code == 0, result2.output
    df2 = pd.read_parquet(out)
    assert df2["feature_asof"].max() <= pd.Timestamp(nav.index[300])
