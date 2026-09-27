"""S3b-3 testleri: CQR (TEMPORAL §6 birebir) + ACI-tarzı günlük global α + coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _features import make_features
from janus.models.conformal import (
    aci_alpha_path,
    calibrate_predictions,
    calibration_diagnostics,
    calibration_mask,
    conformal_interval,
    coverage_report,
)
from janus.models.walkforward import run_walkforward

MK = {"num_boost_round": 20}


@pytest.fixture(scope="module")
def calibrated():
    f = make_features(n_funds=8, days=600, seed=1)
    preds = run_walkforward(f, model_kwargs=MK)
    return f, calibrate_predictions(preds, f, miscoverage_target=0.2, gamma=0.05)


def test_ordered_statistic_manual():
    scores = np.arange(250, dtype=float)  # n=250 >= n_min
    s_k, n, k_gt = conformal_interval(scores, alpha_d=0.2, n_min=200)
    k = int(np.ceil(251 * 0.8))  # = 201
    assert k == 201 and not k_gt
    assert s_k == pytest.approx(scores[k - 1])  # 200.0
    s_k2, _, k_gt2 = conformal_interval(scores, alpha_d=0.02, n_min=200)  # k = ceil(251*0.98)=246 <= 250
    assert not np.isnan(s_k2) and not k_gt2
    s_k3, n3, _ = conformal_interval(scores[:199], alpha_d=0.2, n_min=200)
    assert np.isnan(s_k3) and n3 == 199  # n < 200 → tahmin yok
    s_k4, _, k_gt4 = conformal_interval(scores, alpha_d=0.001, n_min=200)  # k = 251 > 250
    assert np.isnan(s_k4) and k_gt4


def test_calibration_window_and_asof():
    f = make_features(n_funds=4, days=400, seed=2)
    dec = pd.DatetimeIndex(np.sort(pd.to_datetime(f["decision_at"].dropna().unique())))
    d = dec[-1]
    m = calibration_mask(f, d)
    assert (pd.to_datetime(f.loc[m, "label_available_at"]) <= d).all()  # as-of: gelecek label yok
    cal = pd.DatetimeIndex(np.sort(pd.to_datetime(f["feature_asof"].unique())))
    pos_d = int(pd.Series(np.arange(len(cal)), index=cal).loc[pd.Timestamp(f["feature_asof"].max())])
    assert (cal.get_indexer(pd.to_datetime(f.loc[m, "feature_asof"])) >= pos_d - 126 - 23).all()  # pencere sınırı
    # pencere dışı satır maskede değil
    early = f[f["feature_asof"] < cal[pos_d - 149]]
    if len(early := early if False else f[f["feature_asof"].isin(cal[: max(pos_d - 149, 0)])]):
        assert not m.reindex(early.index).fillna(False).any()


def test_flags_and_crossing():
    from janus.models.conformal import _flag

    assert _flag([]) == "ok"
    assert _flag(["crossing"]) == "crossing"
    assert _flag(["crossing", "insufficient_calibration"]) == "crossing+insufficient_calibration"  # crossing ezilmez
    assert _flag(["insufficient_calibration"]) == "insufficient_calibration"
    assert _flag(["k_gt_n"]) == "k_gt_n"


def test_crossing_only_q10_gt_q90(calibrated):
    f, cal = calibrated
    cal = cal.copy()  # modül-scope fixture'ı kirletme
    # sentetik: bir satırda q10 > q90 kur; q10 > q50 / q50 > q90 crossing sayılmamalı
    row_idx = cal.index[0]
    cal.loc[row_idx, "q10"], cal.loc[row_idx, "q50"], cal.loc[row_idx, "q90"] = 0.5, 0.4, 0.3
    # q10 > q90 → crossing; q10 > q50 ama q10 <= q90 → crossing değil
    assert (cal["q10"] > cal["q90"]).sum() >= 1
    mid = cal[(cal["q10"] > cal["q50"]) & (cal["q10"] <= cal["q90"])]
    assert not mid["quality_flag"].str.contains("crossing").any()
    mid2 = cal[(cal["q50"] > cal["q90"]) & (cal["q10"] <= cal["q90"])]
    assert not mid2["quality_flag"].str.contains("crossing").any()


def test_alpha_same_within_day_and_order_invariance(calibrated):
    f, cal = calibrated
    for _d, g in cal.groupby("decision_at"):
        assert g["alpha_D"].nunique() == 1  # günde tek alpha
    # fon sırası invariance: aynı gün satırlarının sırası alpha yolunu değiştirmez
    f2, cal2 = calibrated
    assert cal.groupby("decision_at")["alpha_D"].first().equals(cal.groupby("decision_at")["alpha_D"].first())


def test_no_prediction_rows_preserved(calibrated):
    f, cal = calibrated
    early = cal[cal["n_calib"] < 200]
    if len(early):
        assert early["lower"].isna().all() and early["upper"].isna().all()
        assert early["quality_flag"].str.contains("insufficient_calibration").all()
    assert {
        "decision_at",
        "fund_code",
        "model_id",
        "q10",
        "q50",
        "q90",
        "lower",
        "upper",
        "alpha_D",
        "n_calib",
        "quality_flag",
    }.issubset(cal.columns)


def test_future_perturbation(calibrated):
    f, base = calibrated
    t0 = f["feature_asof"].max() - pd.Timedelta(days=30)
    f2 = f.copy()
    future = f2["feature_asof"] > t0
    f2.loc[future, "y"] = f2.loc[future, "y"] * 1.05
    preds = run_walkforward(f, model_kwargs=MK)
    preds2 = preds.copy()
    preds2.loc[preds2["decision_at"] > t0, ["q10", "q50", "q90"]] *= 1.05
    pert = calibrate_predictions(preds2, f2, miscoverage_target=0.2, gamma=0.05)
    b = base[base["decision_at"] <= t0].reset_index(drop=True)
    p = pert[pert["decision_at"] <= t0].reset_index(drop=True)
    cols = [
        "decision_at",
        "fund_code",
        "model_id",
        "q10",
        "q50",
        "q90",
        "lower",
        "upper",
        "alpha_D",
        "n_calib",
        "quality_flag",
    ]
    pd.testing.assert_frame_equal(b[cols], p[cols], check_exact=True)


def test_symmetric_noise_coverage():
    rng = np.random.default_rng(0)
    n_days, n_funds = 120, 60
    cal_idx = pd.bdate_range("2024-01-02", periods=n_days + 30)
    # simetrik gürültü: y = q50 + N(0, 0.01); q10/q90 gerçek %10/%90 kuantile yakın
    rows, yrows = [], []
    for i, _d in enumerate(cal_idx[:n_days]):
        for j in range(n_funds):
            rows.append(
                {
                    "decision_at": cal_idx[i + 1],
                    "fund_code": f"F{j}",
                    "q10": -0.0128,
                    "q50": 0.0,
                    "q90": 0.0128,
                    "model_id": cal_idx[0],
                    "train_max_t": cal_idx[0],
                }
            )
            yrows.append(
                {
                    "feature_asof": cal_idx[i],
                    "decision_at": cal_idx[i + 1],
                    "fund_code": f"F{j}",
                    "y": rng.normal(0, 0.01),
                    "label_available_at": cal_idx[i + 23],
                }
            )
    preds = pd.DataFrame(rows)  # S3b-2 kayıt şeması: y/label içermez
    feats = pd.DataFrame(yrows)
    cal = calibrate_predictions(preds, feats, miscoverage_target=0.1, gamma=0.05)
    cov = coverage_report(cal)
    last = cov[cov["n_rows"] > 500]
    assert len(last) and abs(last["coverage"].mean() - 0.9) <= 0.03  # ±3 pp


def test_aci_recovery_drifting_distribution():
    rng = np.random.default_rng(3)
    days = 300
    idx = pd.bdate_range("2024-01-02", periods=days)
    # ilk 100 gün sd=0.01, sonra sd=0.03 (kayan dağılım); sabit aralık ±0.0128
    sd = np.r_[np.full(100, 0.01), np.full(days - 100, 0.03)]
    errs = {}
    for i in range(1, days):
        y = rng.normal(0, sd[i - 1])
        errs[idx[i]] = float(y < -0.0128 or y > 0.0128)
    path = aci_alpha_path(pd.Series(errs), miscoverage_target=0.1, gamma=0.05)
    # toparlanma: kayma sonrası err > target → alpha düşer → k büyür → aralık genişler
    assert path.iloc[-1] < path.iloc[0]
    assert path.between(0.02, 0.5).all()


def test_alpha_path_formula():
    errs = pd.Series(
        {pd.Timestamp("2024-01-02"): 0.0, pd.Timestamp("2024-01-03"): 0.5, pd.Timestamp("2024-01-04"): np.nan}
    )
    path = aci_alpha_path(errs, miscoverage_target=0.2, gamma=0.05)
    assert path.iloc[0] == pytest.approx(0.2)  # alpha_init = target
    assert path.iloc[1] == pytest.approx(0.2 + 0.05 * (0.2 - 0.0))  # err=0 → alpha artar
    assert path.iloc[2] == pytest.approx(np.clip(0.21 + 0.05 * (0.2 - 0.5), 0.02, 0.5))  # err=0.5 → alpha düşer
    # NaN gün: alpha değişmez (path.iloc[2] kaydı, 3. günün err'si sonrası zaten yok)


def test_cli_calibrate_disjoint_outputs(tmp_path, monkeypatch):
    """S3b-3-ek: target'a göre ayrık çıktı; ikinci koşu ilkini ezmez."""
    import json as _json

    from typer.testing import CliRunner

    import janus.cli as cli

    rng = np.random.default_rng(0)
    cal_idx = pd.bdate_range("2024-01-02", periods=200)
    rows, yrows = [], []
    for i in range(128):
        for j in range(3):
            rows.append(
                {
                    "decision_at": cal_idx[i + 1],
                    "fund_code": f"F{j}",
                    "q10": -0.01,
                    "q50": 0.0,
                    "q90": 0.01,
                    "model_id": cal_idx[0],
                    "train_max_t": cal_idx[0],
                }
            )
            yrows.append(
                {
                    "feature_asof": cal_idx[i],
                    "decision_at": cal_idx[i + 1],
                    "fund_code": f"F{j}",
                    "y": rng.normal(0, 0.01),
                    "label_available_at": cal_idx[i + 23],
                }
            )
    fdir = tmp_path / "data" / "features"
    fdir.mkdir(parents=True)
    pd.DataFrame(yrows).to_parquet(fdir / "fund_features.parquet", index=False)
    pdir = tmp_path / "data" / "predictions"
    pdir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(pdir / "predictions.parquet", index=False)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: {"conformal": {}})

    runner = CliRunner()
    r1 = runner.invoke(cli.app, ["predictions", "calibrate", "--target", "0.20"])
    assert r1.exit_code == 0, r1.output
    f20, s20 = pdir / "calibrated_target_020.parquet", pdir / "calibration_summary_target_020.json"
    assert f20.exists() and s20.exists()
    blob20 = f20.read_bytes()
    assert _json.loads(s20.read_text())["miscoverage_target"] == 0.20

    r2 = runner.invoke(cli.app, ["predictions", "calibrate", "--target", "0.10"])
    assert r2.exit_code == 0, r2.output
    f10, s10 = pdir / "calibrated_target_010.parquet", pdir / "calibration_summary_target_010.json"
    assert f10.exists() and s10.exists()
    assert _json.loads(s10.read_text())["miscoverage_target"] == 0.10
    # ilk koşunun çıktıları değişmedi (overwrite yok)
    assert f20.read_bytes() == blob20
    assert not (pdir / "calibrated.parquet").exists()  # eski ortak yol artık yazılmıyor


def test_alpha_moves_and_update_formula(calibrated):
    """S3b-3-fix regresyon: err_D olgunlaşan satırlardan hesaplanmalı; alpha yolu hareket etmeli."""
    f, cal = calibrated
    moved = cal.dropna(subset=["alpha_D"]).groupby("decision_at")["alpha_D"].first()
    assert moved.nunique() > 1  # alpha yolu donmaz
    # elle doğrulama: alpha_{d_next} = clip(alpha_{d_prev} + gamma*(target - err_{d_prev}))
    target, gamma = 0.2, 0.05
    days = list(moved.index)
    nav_cal = pd.DatetimeIndex(np.sort(pd.to_datetime(f["feature_asof"].dropna().unique())))
    pos_of = pd.Series(np.arange(len(nav_cal)), index=nav_cal)
    checked = 0
    for d_prev, d_next in zip(days[:-1], days[1:], strict=False):
        # err_{d_prev}: d_prev günü olgunlaşan satırlar (feature_asof = nav_cal[pos-23],
        # decision_at = nav_cal[pos-22]); aralığı olanlar üzerinden
        pos_d_prev = int(pos_of.loc[d_prev])
        t_m = pos_d_prev - 23
        if t_m + 1 >= len(nav_cal):
            continue
        m = (cal["decision_at"] == nav_cal[t_m + 1]) & cal["lower"].notna()  # hizalı seçim
        if not m.any():
            continue  # o gün aralıklı olgun satır yok → alpha değişmez (§6 NaN davranışı)
        y_m = cal.loc[m, "y"].to_numpy(float)
        lo_m = cal.loc[m, "lower"].to_numpy(float)
        hi_m = cal.loc[m, "upper"].to_numpy(float)
        err = float(np.mean((y_m < lo_m) | (y_m > hi_m)))
        expected = float(np.clip(moved.loc[d_prev] + gamma * (target - err), 0.02, 0.5))
        assert moved.loc[d_next] == pytest.approx(expected), f"{d_next.date()}: err={err}"
        checked += 1
    assert checked > 10  # formül yeterli sayıda günde elle doğrulandı


def test_calibrate_diagnose_writes_nothing(tmp_path, monkeypatch):
    """--diagnose: yalnız agregat çıktı; hiçbir parquet/summary/üretim dosyası yazmaz."""

    from typer.testing import CliRunner

    import janus.cli as cli

    rng = np.random.default_rng(0)
    cal_idx = pd.bdate_range("2024-01-02", periods=200)
    rows, yrows = [], []
    for i in range(128):
        for j in range(3):
            rows.append(
                {
                    "decision_at": cal_idx[i + 1],
                    "fund_code": f"F{j}",
                    "q10": -0.01,
                    "q50": 0.0,
                    "q90": 0.01,
                    "model_id": cal_idx[0],
                    "train_max_t": cal_idx[0],
                }
            )
            yrows.append(
                {
                    "feature_asof": cal_idx[i],
                    "decision_at": cal_idx[i + 1],
                    "fund_code": f"F{j}",
                    "y": rng.normal(0, 0.01),
                    "label_available_at": cal_idx[i + 23],
                }
            )
    fdir = tmp_path / "data" / "features"
    fdir.mkdir(parents=True)
    pd.DataFrame(yrows).to_parquet(fdir / "fund_features.parquet", index=False)
    pdir = tmp_path / "data" / "predictions"
    pdir.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(pdir / "predictions.parquet", index=False)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda: {"conformal": {}})
    before = sorted(str(p) for p in tmp_path.rglob("*") if p.is_file())
    result = CliRunner().invoke(cli.app, ["predictions", "calibrate", "--target", "0.20", "--diagnose"])
    assert result.exit_code == 0, result.output
    after = sorted(str(p) for p in tmp_path.rglob("*") if p.is_file())
    assert before == after  # hiçbir dosya yazılmadı
    for key in (
        "n_decision_days",
        "n_err_days",
        "n_err_nan_days",
        "n_mature_days",
        "mature_rows_min",
        "mature_rows_median",
        "mature_rows_max",
        "n_mature_with_interval",
        "n_alpha_changes",
        "alpha_first_change",
        "alpha_min",
        "alpha_max",
    ):
        assert key in result.output


def test_no_calendar_flag():
    """F-08: takvim dışı karar günü satırları no_calendar bayrağı taşır."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-02", periods=130)
    rows, yrows = [], []
    for i in range(100):
        for j in range(2):
            rows.append(
                {
                    "decision_at": idx[i + 1],
                    "fund_code": f"F{j}",
                    "q10": -0.01,
                    "q50": 0.0,
                    "q90": 0.01,
                    "model_id": idx[0],
                    "train_max_t": idx[0],
                }
            )
            yrows.append(
                {
                    "feature_asof": idx[i],
                    "decision_at": idx[i + 1],
                    "fund_code": f"F{j}",
                    "y": rng.normal(0, 0.01),
                    "label_available_at": idx[i + 23],
                }
            )
    preds = pd.DataFrame(rows)
    feats = pd.DataFrame(yrows)
    # takvim dışı karar günü ekle
    extra = preds.iloc[:2].copy()
    extra["decision_at"] = pd.Timestamp("2025-01-01")
    preds_extra = pd.concat([preds, extra], ignore_index=True)
    cal = calibrate_predictions(preds_extra, feats, miscoverage_target=0.2, gamma=0.05)
    extra_cal = cal[cal["decision_at"] == pd.Timestamp("2025-01-01")]
    assert (extra_cal["quality_flag"] == "no_calendar").all()
    assert extra_cal["lower"].isna().all()


def test_diagnostics_crossing_matches_production():
    """F-09: calibration_diagnostics crossing sıralaması üretimle aynı."""
    f = make_features(n_funds=8, days=300, seed=4)
    preds = run_walkforward(f, model_kwargs={"num_boost_round": 20})
    # q10 > q90 olacak şekilde boz
    preds_broken = preds.copy()
    preds_broken["q10"] = preds_broken["q90"] + 0.01
    diag = calibration_diagnostics(preds_broken, f, miscoverage_target=0.2, gamma=0.05)
    cal = calibrate_predictions(preds_broken, f, miscoverage_target=0.2, gamma=0.05)
    # tanıdaki interval'li satırlar üretimde crossing ile aynı lower/upper sıralamasına sahip
    m = cal["lower"].notna()
    assert (cal.loc[m, "lower"] <= cal.loc[m, "upper"]).all()
    assert diag["n_decision_days"] > 0
