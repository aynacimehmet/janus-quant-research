"""S5-0b: artımlı boru hattı eşitlik testleri (sentetik 3 günlük pencere, bit düzeyinde)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import janus.cli as cli
import janus.features.macro as feat_macro
from _panel import make_panel
from janus.cli import (
    _features_build_impl,
    _predictions_build_impl,
    _predictions_calibrate_impl,
    _select_impl,
    app,
)
from janus.models.conformal import calibrate_predictions
from janus.models.walkforward import refit_dates, run_walkforward

MK = {"num_boost_round": 20}
CFG = {"legs": {"tefas": {"universe": {"max_stale_days": 2}}}}


def _make_macro(cal):
    return pd.DataFrame(
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


def _make_fm(codes):
    return pd.DataFrame(
        {
            "fund_code": codes,
            "fund_class": "YAT",
            "umbrella_type": "Hisse",
            "withholding_rate": 0.175,
            "tefas_status": "",
            "founder": "K",
            "manager": "",
            "name": "Fon " + pd.Series(codes),
        }
    )


class _FakeStore:
    snapshot_asof = pd.Timestamp("2026-09-24 10:00")

    def __init__(self, nav, fm):
        self._nav, self._fm = nav, fm

    def latest_fund_master(self):
        return self._fm

    def nav_wide(self):
        return self._nav

    def log_run(self, *a, **k):
        pass


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """400 günlük features + 3 günlük uzatma; her ikisi için aynı seed."""
    nav400 = make_panel(n_funds=6, days=400, seed=11)
    nav403 = make_panel(n_funds=6, days=403, seed=11)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    fm = _make_fm(list(nav400.columns))
    st400 = _FakeStore(nav400, fm)
    st403 = _FakeStore(nav403, fm)
    _features_build_impl(nav400, fm, CFG, st400, tmp_path / "features.parquet", incremental=False)
    full403 = _features_build_impl(nav403, fm, CFG, st403, tmp_path / "full403.parquet", incremental=False)
    return {
        "nav400": nav400,
        "nav403": nav403,
        "fm": fm,
        "full403": full403,
        "tmp_path": tmp_path,
    }


def test_features_incremental_bit_identical(env):
    out = env["tmp_path"] / "features.parquet"  # mevcut 400 günlük dosya
    st = _FakeStore(env["nav403"], env["fm"])
    summary = _features_build_impl(env["nav403"], env["fm"], CFG, st, out, incremental=True)
    assert summary["incremental"] is True
    assert summary["new_dates"] == 3 * 6  # 3 yeni iş günü × 6 fon
    inc = pd.read_parquet(out)
    full = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    pd.testing.assert_frame_equal(inc, full, check_exact=True)
    summary2 = _features_build_impl(env["nav403"], env["fm"], CFG, st, out, incremental=True)
    assert summary2["new_dates"] == 0


def test_features_incremental_idempotent_no_new_dates(tmp_path, monkeypatch):
    nav = make_panel(n_funds=5, days=300, seed=13)
    fm = _make_fm(list(nav.columns))
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    out = tmp_path / "features.parquet"
    st = _FakeStore(nav, fm)
    _features_build_impl(nav, fm, CFG, st, out, incremental=False)
    summary = _features_build_impl(nav, fm, CFG, st, out, incremental=True)
    assert summary["new_dates"] == 0
    assert summary["incremental"] is True


def test_predictions_incremental_bit_identical(env):
    feats400 = pd.read_parquet(env["tmp_path"] / "features.parquet")
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    out = env["tmp_path"] / "predictions.parquet"
    _predictions_build_impl(feats400, out, model_kwargs=MK, incremental=False)
    summary = _predictions_build_impl(feats403, out, model_kwargs=MK, incremental=True)
    assert summary["incremental"] is True
    assert summary["new_dates"] == 3 * 6
    full_out = env["tmp_path"] / "pred_full.parquet"
    _predictions_build_impl(feats403, full_out, model_kwargs=MK, incremental=False)
    inc = pd.read_parquet(out)
    full = pd.read_parquet(full_out)
    pd.testing.assert_frame_equal(inc, full, check_exact=True)


def test_calibrate_incremental_bit_identical(env):
    feats400 = pd.read_parquet(env["tmp_path"] / "features.parquet")
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds400 = run_walkforward(feats400, model_kwargs=MK)
    preds403 = run_walkforward(feats403, model_kwargs=MK)
    cal_cfg = {"gamma": 0.05, "alpha_min": 0.02, "alpha_max": 0.5, "n_min": 200, "calib_window": 126}
    out = env["tmp_path"] / "calibrated.parquet"
    _predictions_calibrate_impl(preds400, feats400, 0.2, out, cal_cfg, incremental=False)
    summary = _predictions_calibrate_impl(preds403, feats403, 0.2, out, cal_cfg, incremental=True)
    assert summary["incremental"] is True
    assert summary["new_dates"] == 3 * 6
    full_out = env["tmp_path"] / "cal_full.parquet"
    _predictions_calibrate_impl(preds403, feats403, 0.2, full_out, cal_cfg, incremental=False)
    inc = pd.read_parquet(out)
    full = pd.read_parquet(full_out)
    pd.testing.assert_frame_equal(inc, full, check_exact=True)


def test_select_creates_parquet(env):
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds403 = run_walkforward(feats403, model_kwargs=MK)
    cal403 = calibrate_predictions(preds403, feats403, miscoverage_target=0.2, gamma=0.05)
    date = str(pd.to_datetime(cal403["decision_at"]).max().date())
    out = env["tmp_path"] / f"selection_{date}.parquet"
    cfg = {
        "gamma": 0.05,
        "alpha_min": 0.02,
        "alpha_max": 0.5,
        "n_min": 200,
        "calib_window": 126,
        "fdr_q_grid": [0.10, 0.20, 0.30],
    }
    summary = _select_impl(date, preds403, feats403, cal403, out, cfg)
    assert out.exists()
    sel = pd.read_parquet(out)
    assert set(sel.columns) >= {"fund_code", "p_value", "selected_q10", "selected_q20", "selected_q30", "lower", "q50"}
    assert summary["date"] == date


def test_cli_features_build_incremental(tmp_path, monkeypatch):
    """CLI'de --incremental flagi çalışır ve süre/özeti loglar."""
    nav = make_panel(n_funds=5, days=300, seed=12)
    fm = _make_fm(list(nav.columns))
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: _FakeStore(nav, fm))
    monkeypatch.setattr(cli, "load_config", lambda: CFG)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    runner = CliRunner()
    r1 = runner.invoke(app, ["features", "build"])
    assert r1.exit_code == 0, r1.output
    out = tmp_path / "data" / "features" / "fund_features.parquet"
    assert out.exists()
    r2 = runner.invoke(app, ["features", "build", "--incremental"])
    assert r2.exit_code == 0, r2.output
    assert "new_dates=0" in r2.output


# --- S5-6e/F17: artımlı hat bütünlüğü regresyonları --------------------------------------------


def test_features_incremental_stale_input_blocks(env):
    """Girdi maddeleşmiş dosyanın gerisindeyse hat 'stale' ile durur (F17)."""
    out = env["tmp_path"] / "full403.parquet"
    before = pd.read_parquet(out)
    st = _FakeStore(env["nav400"], env["fm"])
    summary = _features_build_impl(env["nav400"], env["fm"], CFG, st, out, incremental=True)
    assert summary["status"] == "stale"
    assert summary["input_regressed"] is True
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=False)


def test_features_incremental_detects_nav_revision(env):
    """Geçmiş NAV revizyonu → fail-closed; dosya değişmez (F17/PO-A)."""
    out = env["tmp_path"] / "features.parquet"
    before = pd.read_parquet(out)
    nav_rev = env["nav403"].copy()
    nav_rev.iloc[300, 0] = nav_rev.iloc[300, 0] * 1.05  # prefix (overlap öncesi) revizyonu
    st = _FakeStore(nav_rev, env["fm"])
    with pytest.raises(RuntimeError, match="fail-closed"):
        _features_build_impl(nav_rev, env["fm"], CFG, st, out, incremental=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=False)


def test_features_incremental_detects_new_fund(env):
    """Geçmişli yeni fon → fail-closed; dosya değişmez (F17/PO-A)."""
    out = env["tmp_path"] / "features.parquet"
    before = pd.read_parquet(out)
    nav_new = env["nav403"].copy()
    nav_new["F07"] = nav_new.iloc[:, 0].to_numpy() * 0.9
    fm_new = _make_fm([*nav_new.columns])
    st = _FakeStore(nav_new, fm_new)
    with pytest.raises(RuntimeError, match="fail-closed"):
        _features_build_impl(nav_new, fm_new, CFG, st, out, incremental=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=False)


def test_features_incremental_maturing_label_matches_full(tmp_path, monkeypatch):
    """>22 iş günü uzatma olgunlaşan etiketi overlap'ta tazeler; full ile bit-eşit (F17)."""
    nav400 = make_panel(n_funds=6, days=400, seed=21)
    nav430 = make_panel(n_funds=6, days=430, seed=21)
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    fm = _make_fm(list(nav400.columns))
    out = tmp_path / "features.parquet"
    _features_build_impl(nav400, fm, CFG, _FakeStore(nav400, fm), out, incremental=False)
    t450 = make_panel(n_funds=6, days=430, seed=21)
    summary = _features_build_impl(nav430, fm, CFG, _FakeStore(nav430, fm), out, incremental=True)
    full = tmp_path / "full430.parquet"
    _features_build_impl(nav430, fm, CFG, _FakeStore(t450, fm), full, incremental=False)
    inc = pd.read_parquet(out)
    ref = pd.read_parquet(full)
    pd.testing.assert_frame_equal(inc, ref, check_exact=True)
    assert summary["new_dates"] == 30 * 6


def test_features_revision_in_overlap_matches_full(env):
    """Overlap içindeki (yakın) NAV revizyonu full ile eşit kalmalı (F17)."""
    out = env["tmp_path"] / "features.parquet"
    nav_rev = env["nav403"].copy()
    nav_rev.iloc[-2, 0] = nav_rev.iloc[-2, 0] * 1.05
    st = _FakeStore(nav_rev, env["fm"])
    _features_build_impl(nav_rev, env["fm"], CFG, st, out, incremental=True)
    inc = pd.read_parquet(out)
    full = env["tmp_path"] / "full_rev.parquet"
    _features_build_impl(nav_rev, env["fm"], CFG, _FakeStore(nav_rev, env["fm"]), full, incremental=False)
    pd.testing.assert_frame_equal(inc, pd.read_parquet(full), check_exact=True)


def test_predictions_train_max_t_d23_validator(env):
    """train_max_t gerçek D−23 işlem günü sınırını sağlamalı (TEMPORAL §9.3, F17)."""
    feats = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds = run_walkforward(feats, model_kwargs=MK)
    ok, n_bad = cli._train_max_t_ok(preds, feats)
    assert ok is True and n_bad == 0
    bad = preds.head(1).copy()
    bad["train_max_t"] = bad["decision_at"]  # D−23 ihlali
    ok2, n_bad2 = cli._train_max_t_ok(pd.concat([preds.iloc[1:], bad], ignore_index=True), feats)
    assert ok2 is False and n_bad2 == 1


def test_predictions_incremental_detects_revision_and_d23(env):
    """Maddeleşmiş prediction ön eki değişirse fail-closed; dosya değişmez (F17/PO-A)."""
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds = run_walkforward(feats403, model_kwargs=MK)
    out = env["tmp_path"] / "predictions.parquet"
    stale = preds.copy()
    stale.loc[0, "q10"] = stale.loc[0, "q10"] + 1.0  # maddeleşmiş dosyada içerik farkı
    stale.to_parquet(out, index=False)
    before = pd.read_parquet(out)
    with pytest.raises(RuntimeError, match="fail-closed"):
        _predictions_build_impl(feats403, out, model_kwargs=MK, incremental=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=False)


def test_predictions_incremental_uses_real_d23_validator(env):
    """Yazılan prediction'lar gerçek D−23 sınırını sağlar (summary bayrağı)."""
    feats = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    out = env["tmp_path"] / "predictions.parquet"
    summary = _predictions_build_impl(feats, out, model_kwargs=MK, incremental=False)
    assert summary["train_max_t_ok"] is True
    assert summary["train_max_t_violations"] == 0


def test_calibrate_alpha_prefix_append_only(env):
    """α_D öneki append-only: mevcut karar günleri değişmez (TEMPORAL §6, F17)."""
    feats400 = pd.read_parquet(env["tmp_path"] / "features.parquet")
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds400 = run_walkforward(feats400, model_kwargs=MK)
    preds403 = run_walkforward(feats403, model_kwargs=MK)
    cal_cfg = {"gamma": 0.05, "alpha_min": 0.02, "alpha_max": 0.5, "n_min": 200, "calib_window": 126}
    out = env["tmp_path"] / "calibrated.parquet"
    _predictions_calibrate_impl(preds400, feats400, 0.2, out, cal_cfg, incremental=False)
    before = pd.read_parquet(out)
    summary = _predictions_calibrate_impl(preds403, feats403, 0.2, out, cal_cfg, incremental=True)
    assert summary["revision_detected"] is False
    inc = pd.read_parquet(out)
    emax = pd.to_datetime(before["decision_at"]).max()
    keys = ["decision_at", "fund_code"]
    pb = before.sort_values(keys).reset_index(drop=True)
    pa = inc[pd.to_datetime(inc["decision_at"]) <= emax].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(pb[["alpha_D"]], pa[["alpha_D"]].reset_index(drop=True), check_exact=True)


def test_calibrate_revision_fails_closed(env):
    """Kalibrasyon α öneki revize olursa sessizce karışım yazılmaz; fail-closed (F17)."""
    feats400 = pd.read_parquet(env["tmp_path"] / "features.parquet")
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds400 = run_walkforward(feats400, model_kwargs=MK)
    preds403 = run_walkforward(feats403, model_kwargs=MK).copy()
    cal_cfg = {"gamma": 0.05, "alpha_min": 0.02, "alpha_max": 0.5, "n_min": 200, "calib_window": 126}
    out = env["tmp_path"] / "calibrated.parquet"
    _predictions_calibrate_impl(preds400, feats400, 0.2, out, cal_cfg, incremental=False)
    before = pd.read_parquet(out)
    emax = pd.to_datetime(before["decision_at"]).max()
    pmask = pd.to_datetime(preds403["decision_at"]) <= emax
    preds403.loc[pmask, "q10"] = preds403.loc[pmask, "q10"] - 2.0  # prefix alpha'yı değiştirir
    with pytest.raises(RuntimeError, match="append-only"):
        _predictions_calibrate_impl(preds403, feats403, 0.2, out, cal_cfg, incremental=True)
    # Dosya değişmemiş: ön ek korunmuş
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=False)


def test_month_transition_refit_preserved(tmp_path, monkeypatch):
    """Ay geçişinde refit takvimi/model_id korunur; incremental full ile eşit (F17)."""
    nav400 = make_panel(n_funds=6, days=400, seed=31)
    nav435 = make_panel(n_funds=6, days=435, seed=31)  # >1 ay geçiş
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    fm = _make_fm(list(nav400.columns))
    fdir = tmp_path / "features"
    _features_build_impl(nav400, fm, CFG, _FakeStore(nav400, fm), fdir / "features.parquet", incremental=False)
    feats435 = _features_build_impl(nav435, fm, CFG, _FakeStore(nav435, fm), fdir / "f435.parquet", incremental=False)
    assert feats435["n_rows"] > 0
    feats403 = pd.read_parquet(fdir / "features.parquet")
    feats_rev = pd.read_parquet(fdir / "f435.parquet")
    out = tmp_path / "predictions.parquet"
    _predictions_build_impl(feats403, out, model_kwargs=MK, incremental=False)
    summary = _predictions_build_impl(feats_rev, out, model_kwargs=MK, incremental=True)
    full = tmp_path / "pred_full.parquet"
    _predictions_build_impl(feats_rev, full, model_kwargs=MK, incremental=False)
    inc = pd.read_parquet(out)
    pd.testing.assert_frame_equal(inc, pd.read_parquet(full), check_exact=True)
    assert summary["new_dates"] > 0
    month = pd.to_datetime(inc["model_id"]).dt.to_period("M")
    assert inc.groupby(month)["model_id"].nunique().max() == 1
    # model_id = R = ayın ilk işlem günü (TEMPORAL §5)
    refits = set(
        pd.to_datetime(refit_dates(pd.DatetimeIndex(np.sort(pd.to_datetime(feats_rev["decision_at"]).unique()))))
    )
    assert set(pd.to_datetime(inc["model_id"])) <= refits


# --- S5-R2/ADR-0029: canonical prefix bütünlüğü ve terminal doğrulama sırası --------------------


def _bad_latest_train_max_t(monkeypatch):
    """run_walkforward'ı sarıp en yeni karar satırına D−23 ihlali enjekte eder."""
    import janus.models.walkforward as wf

    real = wf.run_walkforward

    def bad(features, model_kwargs=None):
        p = real(features, model_kwargs=model_kwargs).copy()
        i = pd.to_datetime(p["decision_at"]).idxmax()
        p.loc[i, "train_max_t"] = p.loc[i, "decision_at"]  # D−23 ihlali
        return p

    monkeypatch.setattr(wf, "run_walkforward", bad)


def test_predictions_build_d23_append_fail_closed(env, monkeypatch):
    """A1: append yolunda D−23 ihlali yazımdan önce ret; canonical dosya değişmez (ADR-0029/2)."""
    feats400 = pd.read_parquet(env["tmp_path"] / "features.parquet")
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    out = env["tmp_path"] / "predictions.parquet"
    _predictions_build_impl(feats400, out, model_kwargs=MK, incremental=False)
    before = pd.read_parquet(out)
    _bad_latest_train_max_t(monkeypatch)
    with pytest.raises(RuntimeError, match="train_max_t"):
        _predictions_build_impl(feats403, out, model_kwargs=MK, incremental=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=True)


def test_predictions_build_d23_full_fail_closed(env, monkeypatch):
    """A2: full yolda D−23 ihlali yazımdan önce ret; mevcut canonical dosya değişmez (ADR-0029/2)."""
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    out = env["tmp_path"] / "predictions.parquet"
    _predictions_build_impl(feats403, out, model_kwargs=MK, incremental=False)
    before = pd.read_parquet(out)
    _bad_latest_train_max_t(monkeypatch)
    with pytest.raises(RuntimeError, match="train_max_t"):
        _predictions_build_impl(feats403, out, model_kwargs=MK, incremental=False)
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=True)


def _cal_cfg():
    return {"gamma": 0.05, "alpha_min": 0.02, "alpha_max": 0.5, "n_min": 200, "calib_window": 126}


def test_calibrate_q50_prefix_perturbation_fail_closed(env):
    """B1: prefix q50-only değişimi (α sabit) guard'ı geçemez; canonical değişmez (ADR-0029/3)."""
    feats400 = pd.read_parquet(env["tmp_path"] / "features.parquet")
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds400 = run_walkforward(feats400, model_kwargs=MK)
    preds403 = run_walkforward(feats403, model_kwargs=MK).copy()
    out = env["tmp_path"] / "calibrated.parquet"
    _predictions_calibrate_impl(preds400, feats400, 0.2, out, _cal_cfg(), incremental=False)
    before = pd.read_parquet(out)
    emax = pd.to_datetime(before["decision_at"]).max()
    mask = pd.to_datetime(preds403["decision_at"]) <= emax
    preds403.loc[mask, "q50"] = preds403.loc[mask, "q50"] + 3.0  # yalnız canonical prefix q50
    with pytest.raises(RuntimeError, match="append-only"):
        _predictions_calibrate_impl(preds403, feats403, 0.2, out, _cal_cfg(), incremental=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=True)


def test_calibrate_train_max_t_prefix_perturbation_fail_closed(env):
    """B2: prefix train_max_t-only değişimi (α sabit) guard'ı geçemez; canonical değişmez (ADR-0029/3)."""
    feats400 = pd.read_parquet(env["tmp_path"] / "features.parquet")
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds400 = run_walkforward(feats400, model_kwargs=MK)
    preds403 = run_walkforward(feats403, model_kwargs=MK).copy()
    out = env["tmp_path"] / "calibrated.parquet"
    _predictions_calibrate_impl(preds400, feats400, 0.2, out, _cal_cfg(), incremental=False)
    before = pd.read_parquet(out)
    emax = pd.to_datetime(before["decision_at"]).max()
    mask = pd.to_datetime(preds403["decision_at"]) <= emax
    preds403.loc[mask, "train_max_t"] = preds403.loc[mask, "train_max_t"] - pd.Timedelta(days=5)
    with pytest.raises(RuntimeError, match="append-only"):
        _predictions_calibrate_impl(preds403, feats403, 0.2, out, _cal_cfg(), incremental=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=True)


def test_calibrate_materialized_y_change_fail_closed(env):
    """B3: maddeleşmiş `y` değişimi ret; canonical değişmez (ADR-0029/1)."""
    feats400 = pd.read_parquet(env["tmp_path"] / "features.parquet")
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds400 = run_walkforward(feats400, model_kwargs=MK)
    preds403 = run_walkforward(feats403, model_kwargs=MK)
    out = env["tmp_path"] / "calibrated.parquet"
    _predictions_calibrate_impl(preds400, feats400, 0.2, out, _cal_cfg(), incremental=False)
    tampered = pd.read_parquet(out)
    emax = pd.to_datetime(tampered["decision_at"]).max()
    nan_mask = (pd.to_datetime(tampered["decision_at"]) <= emax) & tampered["y"].isna()
    assert nan_mask.any()
    tampered.loc[tampered.index[nan_mask][0], "y"] = -9.0  # dolu→farklı gibi davranır
    tampered.to_parquet(out, index=False)
    stored = pd.read_parquet(out)
    with pytest.raises(RuntimeError, match="append-only"):
        _predictions_calibrate_impl(preds403, feats403, 0.2, out, _cal_cfg(), incremental=True)
    pd.testing.assert_frame_equal(pd.read_parquet(out), stored, check_exact=True, check_dtype=True)


def test_calibrate_y_maturation_accepted_matches_full(env):
    """B3: temiz append'te tek yönlü `y` olgunlaşması kabul; incremental==full (ADR-0029/1)."""
    feats400 = pd.read_parquet(env["tmp_path"] / "features.parquet")
    feats403 = pd.read_parquet(env["tmp_path"] / "full403.parquet")
    preds400 = run_walkforward(feats400, model_kwargs=MK)
    preds403 = run_walkforward(feats403, model_kwargs=MK)
    out = env["tmp_path"] / "calibrated.parquet"
    _predictions_calibrate_impl(preds400, feats400, 0.2, out, _cal_cfg(), incremental=False)
    summary = _predictions_calibrate_impl(preds403, feats403, 0.2, out, _cal_cfg(), incremental=True)
    assert summary["revision_detected"] is False
    full_out = env["tmp_path"] / "cal_full.parquet"
    _predictions_calibrate_impl(preds403, feats403, 0.2, full_out, _cal_cfg(), incremental=False)
    pd.testing.assert_frame_equal(pd.read_parquet(out), pd.read_parquet(full_out), check_exact=True)
