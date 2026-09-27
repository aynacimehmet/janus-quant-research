"""S3b-5-2 testleri: conformal p-değeri (H0: y ≤ 0), BH adım-up, FDR kontrolü, FdrHRP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _features import make_features
from _panel import make_panel
from janus.backtest.costs import meta_for_synthetic
from janus.backtest.engine import BacktestConfig, run_backtest
from janus.models.conformal import calibrate_predictions
from janus.models.conformal_selection import FdrHRP, bh_select, conformal_pvalues, selected_positive_rate
from janus.models.walkforward import run_walkforward

MK = {"num_boost_round": 20}


@pytest.fixture(scope="module")
def s3b():
    f = make_features(n_funds=8, days=600, seed=1)
    preds = run_walkforward(f, model_kwargs=MK)
    cal = calibrate_predictions(preds, f, miscoverage_target=0.2, gamma=0.05)
    nav = make_panel(n_funds=8, days=600, seed=1)
    return f, preds, cal, nav


def test_pvalue_monotonic_in_q50():
    """Monotonluk: daha yüksek q50 → daha küçük p (y değil)."""
    rng = np.random.default_rng(0)
    days = 150
    idx = pd.bdate_range("2024-01-02", periods=days + 30)
    rows, yrows = [], []
    for i in range(days):
        for j in range(3):
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
                    "y": rng.normal(-0.01, 0.01),
                    "label_available_at": idx[i + 23],
                }
            )
    feats = pd.DataFrame(yrows)
    preds = pd.DataFrame(rows)
    # aynı kalibrasyon setinde iki test birimi: q50 düşük vs yüksek
    test_day = idx[100]  # takvim içindeki karar günü (son karar günü feature takvimi dışında olabilir)
    base = preds[preds["decision_at"] == test_day].copy()
    lo = base.copy()
    lo["q50"] = -0.02
    hi = base.copy()
    hi["q50"] = 0.05
    pv_lo = conformal_pvalues(pd.concat([preds[preds["decision_at"] != test_day], lo]), feats)
    pv_hi = conformal_pvalues(pd.concat([preds[preds["decision_at"] != test_day], hi]), feats)
    p_lo = pv_lo[pv_lo["decision_at"] == test_day]["p_value"].mean()
    p_hi = pv_hi[pv_hi["decision_at"] == test_day]["p_value"].mean()
    assert p_hi < p_lo  # büyük q50 → küçük p → aday


def test_bh_step_up_manual():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20}, name="p")
    # sıralı: 0.01, 0.03, 0.04, 0.20; eşikler q=0.10: 0.025, 0.05, 0.075, 0.10 → p_(2)=0.03 ≤ 0.05, p_(3)=0.04 ≤ 0.075, p_(4)=0.20 > 0.10 → k=3
    assert bh_select(p, q=0.10) == ["a", "c", "b"]
    assert bh_select(pd.Series({"x": 0.5, "y": 0.9}), q=0.20) == []  # boş küme
    assert bh_select(pd.Series(dtype=float), q=0.20) == []
    assert bh_select(pd.Series({"x": np.nan}), q=0.20) == []  # NaN p → aday değil


def test_fdr_control_repeated():
    """Tekrarlı sentetik koşuda seçilen kümede gerçekleşen (y ≤ 0) oranı ortalamada ≤ q.

    Fon 0–9: sinyal (y > 0, q50 = y — oracle; yalnız test). Fon 10–39: H0 (y < 0, q50 = 0 → p ≈ 1).
    """
    rng = np.random.default_rng(7)
    q = 0.20
    fdp = []
    for _rep in range(10):
        days, funds = 60, 40
        idx = pd.bdate_range("2024-01-02", periods=days + 30)
        rows, yrows = [], []
        for i in range(days):
            for j in range(funds):
                y = rng.normal(0.10, 0.005) if j < 10 else rng.normal(-0.02, 0.01)
                q50 = y if j < 10 else 0.0  # oracle sinyal (yalnız test); null'da q50 = 0
                rows.append(
                    {
                        "decision_at": idx[i + 1],
                        "fund_code": f"F{j}",
                        "q10": q50 - 0.01,
                        "q50": q50,
                        "q90": q50 + 0.01,
                        "model_id": idx[0],
                        "train_max_t": idx[0],
                    }
                )
                yrows.append(
                    {
                        "feature_asof": idx[i],
                        "decision_at": idx[i + 1],
                        "fund_code": f"F{j}",
                        "y": y,
                        "label_available_at": idx[i + 23],
                    }
                )
        feats = pd.DataFrame(yrows)
        pv = conformal_pvalues(pd.DataFrame(rows), feats)
        for d, g in pv.groupby("decision_at"):
            sel = bh_select(g.set_index("fund_code")["p_value"], q)
            if sel:
                yv = feats[(feats["decision_at"] == d) & feats["fund_code"].isin(sel)]["y"]
                fdp.append(float((yv <= 0).mean()))
    assert len(fdp) > 100  # yeterli seçim günü
    assert np.mean(np.array(fdp)) <= q + 0.05  # ampirik FDR ≤ q (toleranslı)


def test_clipped_variant_smaller_p():
    rng = np.random.default_rng(1)
    days, funds = 120, 30
    idx = pd.bdate_range("2024-01-02", periods=days + 30)
    rows, yrows = [], []
    for i in range(days):
        for j in range(funds):
            q50 = -0.03 + 0.002 * j  # negatif q50 → V̂ > 0 → clipped etkili
            rows.append(
                {
                    "decision_at": idx[i + 1],
                    "fund_code": f"F{j}",
                    "q10": q50 - 0.01,
                    "q50": q50,
                    "q90": q50 + 0.01,
                    "model_id": idx[0],
                    "train_max_t": idx[0],
                }
            )
            yrows.append(
                {
                    "feature_asof": idx[i],
                    "decision_at": idx[i + 1],
                    "fund_code": f"F{j}",
                    "y": rng.normal(0, 0.02),
                    "label_available_at": idx[i + 23],
                }
            )
    feats = pd.DataFrame(yrows)
    preds = pd.DataFrame(rows)
    p0 = conformal_pvalues(preds, feats, clipped=False)
    p1 = conformal_pvalues(preds, feats, clipped=True)
    ok = p0["p_value"].notna() & p1["p_value"].notna()
    assert ok.any()
    assert (p1.loc[ok, "p_value"] <= p0.loc[ok, "p_value"] + 1e-12).all()  # clipped daha güçlü (p küçük)
    assert (p1.loc[ok, "p_value"] < p0.loc[ok, "p_value"]).mean() > 0.3  # çoğu birimde sıkılaşır


def test_future_perturbation_pvalues(s3b):
    f, preds, _, _ = s3b
    t0 = f["feature_asof"].max() - pd.Timedelta(days=30)
    f2 = f.copy()
    f2.loc[f2["feature_asof"] > t0, "y"] *= 1.05
    p1 = conformal_pvalues(preds, f)
    p2 = conformal_pvalues(preds, f2)
    b = p1[pd.to_datetime(p1["decision_at"]) <= t0].reset_index(drop=True)
    p = p2[pd.to_datetime(p2["decision_at"]) <= t0].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        b[["decision_at", "fund_code", "p_value", "n_calib"]],
        p[["decision_at", "fund_code", "p_value", "n_calib"]],
        check_exact=True,
    )


def test_fdr_hrp_empty_set_cash(s3b):
    f, preds, cal, nav = s3b
    st = FdrHRP(calibrated=cal, predictions=preds, features=f, fdr_q=1e-9)  # hiçbir gün küme dolmaz
    cash_r = pd.Series(0.0003, index=nav.index)
    res = run_backtest(
        nav,
        meta_for_synthetic(list(nav.columns)),
        st,
        BacktestConfig(initial_capital=100, warmup_days=260),
        cash_returns=cash_r,
        cash_nav=(1 + cash_r).cumprod(),
    )
    risky = res.weights.drop(columns=["CASH_PROXY"], errors="ignore")
    assert (risky.sum(axis=1) < 1e-9).all()
    assert np.allclose(res.equity.to_numpy(), res.cash_index.to_numpy(), rtol=1e-9)


def test_fdr_hrp_selects_and_positive_rate(s3b):
    f, preds, cal, nav = s3b
    st = FdrHRP(calibrated=cal, predictions=preds, features=f, fdr_q=0.5)
    meta = meta_for_synthetic(list(nav.columns))
    cash_r = pd.Series(0.0003, index=nav.index)
    run_backtest(
        nav,
        meta,
        st,
        BacktestConfig(initial_capital=100, warmup_days=260),
        cash_returns=cash_r,
        cash_nav=(1 + cash_r).cumprod(),
    )
    assert st._bh and any(v for v in st._bh.values())
    rate, n = selected_positive_rate(cal, st._history)
    assert n > 0 and np.isfinite(rate)
    # seçim geçmişi BH kümesiyle uyumlu: seçilen fonlar o günün kümesinde
    for d, chosen in st._history.items():
        if chosen:
            assert set(chosen) <= st._bh.get(pd.Timestamp(d), set())


def test_fdr_hrp_validation():
    with pytest.raises(ValueError, match="predictions"):
        FdrHRP(calibrated=pd.DataFrame({"decision_at": [1], "fund_code": ["F"], "lower": [0.1]}))


def _suite_cfg() -> dict:
    """select-suite build_strategies için minimal cfg (test_select_suite CFG ile aynı yapı)."""
    return {
        "legs": {
            "tefas": {
                "constraints": {
                    "max_weight_per_fund": 0.25,
                    "max_weight_per_founder": 0.30,
                    "max_funds_per_founder": 3,
                    "max_funds_per_hrp_cluster": 3,
                }
            }
        },
        "hrp": {
            "cov_days": 126,
            "cluster_distance": 0.4,
            "linkage": "single",
            "buffer_mult": 2.0,
            "tax_aware": True,
            "refresh_every": 3,
            "smooth_lambda": 0.5,
        },
        "validation": {"start_dates": 3},
    }


@pytest.fixture()
def env_fdr():
    """Küçük panel + gerçek build_features çerçevesi (cash_excess_21 kolonu dahil) + sentetik q'lar."""
    f = make_features(n_funds=6, days=400, seed=3)
    preds = run_walkforward(f, model_kwargs=MK)
    cal = calibrate_predictions(preds, f, miscoverage_target=0.2, gamma=0.05)
    nav = make_panel(n_funds=6, days=400, seed=3)
    from janus.backtest.costs import meta_for_synthetic

    cash_r = pd.Series(0.0003, index=nav.index)
    d = {
        "nav": nav,
        "fund_master": pd.DataFrame({"fund_code": list(nav.columns), "founder": "K"}),
        "meta": meta_for_synthetic(list(nav.columns)),
        "eligible": pd.Series(True, index=nav.columns),
        "cash_codes": [],
        "cash_returns": cash_r,
        "cash_nav": (1 + cash_r).cumprod(),
        "cash_category": None,
        "cash_rate": 0.175,
        "equity_index": nav.iloc[:, 0],
        "snapshot_asof": None,
    }
    return d, cal, preds, f


def test_fdr_q_grid_rows(env_fdr):
    """S3b-5-3: q ızgarası satırları (q10/q30) + p-değerleri bir kez hesaplanır."""
    d, cal, preds, feats = env_fdr
    from janus.backtest.select_suite import build_strategies

    strats = build_strategies(
        d,
        _suite_cfg(),
        top_n=3,
        calibrated=cal,
        predictions=preds,
        features=feats,
        kill_switch=False,
        fdr_q=0.20,
        fdr_q_grid=(0.10, 0.20, 0.30),
        features_pit=feats,
    )
    names = [x.name for x in strats]
    assert "B2c_fdr_hrp" in names and "B2c_fdr_hrp_q10" in names and "B2c_fdr_hrp_q30" in names
    assert "B2b_ks2" in names and "B2c_m_ks2" in names  # kill-switch v2 satırları
    qmap = {x.name: x.fdr_q for x in strats if x.name.startswith("B2c_fdr")}
    assert qmap["B2c_fdr_hrp_q10"] == 0.10 and qmap["B2c_fdr_hrp_q30"] == 0.30
    # p-değerleri önbellek: aynı frame nesnesi
    pvs = [id(x.pvalues) for x in strats if x.name.startswith("B2c_fdr")]
    assert len(set(pvs)) == 1


def test_ks2_cash_fallback(env_fdr):
    """Kill-switch v2: seçilen fonların olgunlaşmış 63g ortalama nakit-fazlası < 0 → NAKİT (momentum değil)."""
    d, cal, preds, feats = env_fdr
    nav = d["nav"]
    meta = d["meta"]
    # cash_excess_21'i seçilen fonlar için negatif yap: tüm fonlar negatif → her seçim nakde düşer
    f_neg = feats.copy()
    f_neg["cash_excess_21"] = -0.01
    st = FdrHRP(calibrated=cal, predictions=preds, features=f_neg, fdr_q=0.5, ks2=True, ks2_features=f_neg)
    cash_r = pd.Series(0.0003, index=nav.index)
    res = run_backtest(
        nav,
        meta,
        st,
        BacktestConfig(initial_capital=100, warmup_days=260),
        cash_returns=cash_r,
        cash_nav=(1 + cash_r).cumprod(),
    )
    risky = res.weights.drop(columns=["CASH_PROXY"], errors="ignore")
    assert (risky.sum(axis=1) < 1e-9).all()  # tüm rebalance günlerinde nakit
    assert st.ks2_days > 0
    assert np.allclose(res.equity.to_numpy(), res.cash_index.to_numpy(), rtol=1e-9)
    # pozitif nakit-fazlası → normal seçim
    f_pos = feats.copy()
    f_pos["cash_excess_21"] = 0.01
    st2 = FdrHRP(calibrated=cal, predictions=preds, features=f_pos, fdr_q=0.5, ks2=True, ks2_features=f_pos)
    res2 = run_backtest(
        nav,
        meta,
        st2,
        BacktestConfig(initial_capital=100, warmup_days=260),
        cash_returns=cash_r,
        cash_nav=(1 + cash_r).cumprod(),
    )
    assert st2.ks2_days == 0  # tetiklenmedi
    assert (res2.weights.drop(columns=["CASH_PROXY"], errors="ignore").sum(axis=1) > 0).any()


def test_ks2_momentum_fallback_is_cash_not_momentum(env_fdr):
    """B2b-ks2: tetiklenince momentum'a değil NAKİTe düşer (kill-switch v1'den fark)."""
    d, cal, preds, feats = env_fdr
    nav = d["nav"]
    f_neg = feats.copy()
    f_neg["cash_excess_21"] = -0.01
    from janus.strategies.portfolio import MomentumHRP

    st = MomentumHRP(n=3, ks2=True, ks2_features=f_neg, eligible=d["eligible"])
    cash_r = pd.Series(0.0003, index=nav.index)
    res = run_backtest(
        nav,
        d["meta"],
        st,
        BacktestConfig(initial_capital=100, warmup_days=260),
        cash_returns=cash_r,
        cash_nav=(1 + cash_r).cumprod(),
    )
    risky = res.weights.drop(columns=["CASH_PROXY"], errors="ignore")
    assert (risky.sum(axis=1) < 1e-9).all()  # momentum seçimi YOK, nakit
    assert st.ks2_days > 0


def test_nan_q50_ghost_pvalue():
    """F-01: NaN q50 test satırı hayalet p-değerine dönüşmemeli; p=NaN, flag=missing_q50."""
    rng = np.random.default_rng(0)
    days, funds = 150, 3
    idx = pd.bdate_range("2024-01-02", periods=days + 30)
    rows, yrows = [], []
    for i in range(days):
        for j in range(funds):
            rows.append(
                {
                    "decision_at": idx[i + 1],
                    "fund_code": f"F{j}",
                    "q10": -0.01,
                    "q50": 0.0 if j > 0 else np.nan,  # F0 q50 NaN
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
                    "y": rng.normal(-0.01, 0.01),
                    "label_available_at": idx[i + 23],
                }
            )
    preds = pd.DataFrame(rows)
    feats = pd.DataFrame(yrows)
    pv = conformal_pvalues(preds, feats)
    f0 = pv[pv["fund_code"] == "F0"]
    # yeterli kalibrasyonlu günlerde NaN q50 → missing_q50; erken günler insufficient_calibration olabilir
    f0_ok = f0[f0["n_calib"] >= 200]
    assert f0_ok["p_value"].isna().all()
    assert (f0_ok["quality_flag"] == "missing_q50").all()
    f1 = pv[pv["fund_code"] == "F1"]
    assert f1["p_value"].notna().any()


def test_pvalue_laa_filter_defense():
    """F-07: label_available_at > D olan satırlar kalibrasyona girmemeli."""
    rng = np.random.default_rng(1)
    days, funds = 80, 3
    idx = pd.bdate_range("2024-01-02", periods=days + 30)
    rows, yrows = [], []
    for i in range(days):
        for j in range(funds):
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
            # bazı satırların label_available_at'ını geleceğe taşı (özellikle son günler)
            laa = idx[i + 23] if i < days - 10 else idx[-1]
            yrows.append(
                {
                    "feature_asof": idx[i],
                    "decision_at": idx[i + 1],
                    "fund_code": f"F{j}",
                    "y": rng.normal(-0.01, 0.01),
                    "label_available_at": laa,
                }
            )
    preds = pd.DataFrame(rows)
    feats = pd.DataFrame(yrows)
    pv = conformal_pvalues(preds, feats)
    # son karar günü için kalibrasyon satırları laa <= d olmalı
    last_d = preds["decision_at"].max()
    used = feats[feats["label_available_at"] <= last_d]
    # p-değeri hesaplanan günlerde n_calib, laa filtresi uygulanmış satır sayısıyla uyumlu olmalı
    assert (pv["n_calib"] <= len(used)).all()
