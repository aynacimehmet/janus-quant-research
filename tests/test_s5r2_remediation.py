"""S5-R2/ADR-0029 kalıcı regresyon: canonical prefix bütünlüğü, terminal doğrulama ve panel takvim.

Kapsam:
- PROOF A/B/C/D: `train_max_t <= D-23` hesabının takvimi gerçek NAV işlem-günü takvimiyle
  (panel ekseni) eşdeğerliğini sentetik olarak kanıtlar (ZORUNLU kontrol).
- PROOF E: `_frames_equal(check_dtype=True)` dtype-only farkta fail-closed davranışı.
- PROOF F: BES artımlı yolunda kapsam değişimi canonical'ı guard öncesi BUDAMAZ; fail-closed
  yükselir ve kanonik dosya byte/değer+dtype sabit kalır (ADR-0029/4).
- B2: configured iş günü ekseni dışındaki panel tarihleri `_features_build_impl` /
  `_predictions_build_impl` yolunda fail-closed; geçerli iş günü paneli + terminal kabul (GREEN).

Salt sentetik veri; data/, reports/, mlruns/ okunmaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import janus.features.macro as feat_macro
import janus.features.market as feat_market
from _panel import make_panel
from janus.cli import (
    _features_build_impl,
    _frames_equal,
    _predictions_build_impl,
    _train_max_t_ok,
)
from janus.config import load_config
from janus.data.quality import next_business_day
from janus.data.store import Store
from janus.features.fund_features import build_features
from janus.models.walkforward import run_walkforward

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


def _make_market(cal):
    return pd.DataFrame({"eq_trend63": 0.0, "eq_vol21": 0.1, "breadth200": 0.5}, index=cal)


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


def _panel_with_holidays(days=430, seed=11):
    """Gerçekçi işlem-günü takvimi: bdate_range + panel ekseninden düşürülmüş tatiller."""
    nav = make_panel(n_funds=6, days=days, seed=seed)
    idx = pd.DatetimeIndex(nav.index)
    return nav.reindex(idx.delete([60, 140, 260]))


def test_proof_abcd_train_max_t_calendar_equivalence():
    nav = _panel_with_holidays()
    cal = pd.DatetimeIndex(nav.index)
    n = len(cal)
    fm = _make_fm(list(nav.columns))
    cash = pd.Series(0.0002, index=cal)
    feats = build_features(nav, fm, cash, _make_macro(cal), _make_market(cal), cfg=CFG, leg="tefas")

    # PROOF A — feature_asof panel eksenini birebir kapsar (union-of-stack argümanının ampirik kanıtı)
    fa = pd.DatetimeIndex(pd.to_datetime(feats["feature_asof"]).dropna().unique()).sort_values()
    assert list(fa) == list(cal), "feature_asof != panel ekseni"

    preds = run_walkforward(feats, model_kwargs=MK)
    assert len(preds) > 0

    terminal = next_business_day(cal[-1], CFG)
    da = pd.DatetimeIndex(pd.to_datetime(preds["decision_at"]).dropna().unique()).sort_values()
    assert set(da) <= set(cal[1:]) | {terminal}

    # Doğrulayıcı takvimi = panel ∪ {terminal} (küme eşitliği)
    fa_arr = pd.to_datetime(feats["feature_asof"]).dropna().to_numpy()
    da_arr = pd.to_datetime(preds["decision_at"]).dropna().to_numpy()
    vcal = pd.DatetimeIndex(np.unique(np.concatenate([fa_arr, da_arr]))).sort_values()
    assert list(vcal) == list(cal) + [terminal], "validator takvimi panel ∪ {terminal} değil"

    # PROOF B — bağımsız sözleşme kontrolü: panel pozisyonlarıyla t <= D-23
    pos_panel = {d: i for i, d in enumerate(cal)}

    def contract_check(p):
        for _, row in p.iterrows():
            d, t = pd.Timestamp(row["decision_at"]), pd.Timestamp(row["train_max_t"])
            p_d = pos_panel.get(d, n if d == terminal else None)
            p_t = pos_panel.get(t)
            assert p_d is not None and p_t is not None, f"takvim dışı tarih d={d} t={t}"
            if p_t > p_d - 23:
                return False
        return True

    ok, n_bad = _train_max_t_ok(preds, feats)
    assert ok and n_bad == 0
    assert contract_check(preds), "bağımsız panel-posisyon kontrolü başarısız"

    # PROOF C — tam sınır: 1 pozisyon ihlali yakalanır; D-23 tam sınır yasaldır
    mid = len(preds) // 2
    row = preds.iloc[mid]
    d = pd.Timestamp(row["decision_at"])
    p_d = pos_panel.get(d, n if d == terminal else None)
    assert p_d >= 40
    viol = preds.copy()
    viol.loc[viol.index[mid], "train_max_t"] = cal[p_d - 22]
    ok_v, n_v = _train_max_t_ok(viol, feats)
    assert not ok_v and n_v >= 1, "D-22 (1 pozisyon) ihlali yakalanmadı"
    assert not contract_check(viol)
    legal = preds.copy()
    legal.loc[legal.index[mid], "train_max_t"] = cal[p_d - 23]
    ok_l, n_l = _train_max_t_ok(legal, feats)
    assert ok_l and n_l == 0, "D-23 tam sınırı yanlışlıkla ihlal sayıldı"
    assert contract_check(legal)

    # Sözleşme özdeşliği: label_available_at = cal[t+23] (sonca bir önceki satırda terminal)
    sub = feats[feats["fund_code"] == nav.columns[0]]
    for _, r in sub.iloc[::40].iterrows():
        t = pd.Timestamp(r["feature_asof"])
        laa = pd.Timestamp(r["label_available_at"])
        p_t = pos_panel[t]
        if pd.notna(laa):
            if p_t + 23 < n:
                assert laa == cal[p_t + 23]
            else:
                assert p_t == n - 23 and laa == terminal

    # PROOF D1 — panel ekseninde EKSİK gün guardı sıkılaştırır (yanlış pozitif yönü, muhafazakâr).
    # Doğrulayıcı takvimi feature_asof ∪ decision_at BİRLİKESİDİR: tarihin takvimden gerçekten
    # çıkması için iki çerçevede de olmaması gerekir (tutarlı panel boşluğu senaryosu).
    missing_date = cal[p_d - 10]
    feats_missing = feats[pd.to_datetime(feats["feature_asof"]) != missing_date]
    legal_missing = legal[pd.to_datetime(legal["decision_at"]) != missing_date]
    ok_m, n_m = _train_max_t_ok(legal_missing, feats_missing)
    assert not ok_m and n_m >= 1, "eksik panel günü yasal D-23 satırını ihlal saymalı (sıkı yön)"

    # PROOF D2 — panel ekseninde FAZLADAN (işlem günü olmayan) gün guardı zayıflatır:
    # t..D arasına tek sahte tarih girince gerçek D-22 ihlali geçer (kirlilik yönü).
    spurious = cal[p_d - 22] + pd.Timedelta(days=1)
    while spurious in cal:
        spurious += pd.Timedelta(days=1)
    assert cal[p_d - 22] < spurious < d, "sahte tarih t..D penceresine düşmeli"
    fake = pd.DataFrame({c: [np.nan] for c in feats.columns})
    fake["feature_asof"] = [spurious]
    feats_polluted = pd.concat([feats, fake], ignore_index=True)
    ok_p, n_p = _train_max_t_ok(viol, feats_polluted)
    assert ok_p and n_p == 0, "sahte panel günü D-22 ihlalini geçirdi — guard zayıfladı"


def test_proof_e_frames_equal_dtype_fail_closed():
    keys = ["feature_asof", "fund_code"]
    base = pd.DataFrame(
        {"feature_asof": pd.to_datetime(["2026-09-01", "2026-09-01"]), "fund_code": ["A", "B"], "value": [1, 2]}
    )
    assert _frames_equal(base, base.copy(), keys) is True
    float_variant = base.copy()
    float_variant["value"] = float_variant["value"].astype(float)
    assert _frames_equal(base, float_variant, keys) is False, "dtype-only fark fail-closed değil"
    extra = base.copy()
    extra["extra"] = 1
    assert _frames_equal(base, extra, keys) is False
    assert _frames_equal(base, base.iloc[1:], keys) is False
    with_nan = pd.DataFrame({"feature_asof": pd.to_datetime(["2026-09-01"]), "fund_code": ["A"], "value": [np.nan]})
    assert _frames_equal(with_nan, with_nan.copy(), keys) is True


def test_proof_f_bes_scope_change_fails_closed_without_pruning(tmp_path, monkeypatch):
    """PROOF F — BES: kapsam değişimi canonical'ı guard öncesi budamaz (ADR-0029/4).

    Kapsam dışı fon (E00/OKS) eski canonical'da duruyor; yeniden üretilen önek onu içermiyor.
    Guard fail-closed yükseltir ve kanonik dosya byte/değer+dtype bakımından DEĞİŞMEZ; satır budanmaz.
    """
    cfg = load_config()
    cfg["ingest"]["fund_types"] = ["EMK"]
    cfg["legs"]["bes"]["exclude_umbrella_patterns"] = ["oks"]
    fm = pd.DataFrame(
        {
            "fund_code": ["E00", "E05", "E10"],
            "fund_class": ["EMK", "EMK", "EMK"],
            "umbrella_type": ["OKS", "Değişken", "Para Piyasası"],
            "category": ["OKS", "Değişken", "Para Piyasası"],
            "withholding_rate": 0.0,
        }
    )
    # Configured tatilleri (2026-10-29) içermeyen aralık: B2 takvim guard'ı değil, kapsam guard'ı test edilir.
    idx = pd.bdate_range("2026-03-02", periods=80)
    nav = pd.DataFrame({c: 10.0 for c in fm["fund_code"]}, index=idx)
    out = tmp_path / "bes_features.parquet"
    legacy = pd.DataFrame(
        {
            "feature_asof": [idx[0], idx[0]],
            "fund_code": ["E00", "E05"],
            "label_ready": [False, False],
            "eligible_at_decision": [True, True],
        }
    )
    legacy.to_parquet(out, index=False)
    before = pd.read_parquet(out)
    before_bytes = out.read_bytes()
    assert len(before) == 2

    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: pd.DataFrame(index=cal))
    monkeypatch.setattr(feat_market, "market_features", lambda *a, **k: pd.DataFrame(index=nav.index))

    with pytest.raises(RuntimeError, match="fail-closed"):
        _features_build_impl(nav, fm, cfg, Store(tmp_path / "legacy.duckdb"), out, incremental=True, leg="bes")

    # Fail-closed invariantı: kanonik dosya değişmedi, kapsam dışı E00 budanmadı.
    assert out.read_bytes() == before_bytes, "fail-closed sonrası kanonik dosya değişti"
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=True)
    assert set(pd.read_parquet(out)["fund_code"]) == {"E00", "E05"}


# --- B2/ADR-0029/5: panel takvimi fail-closed (RED→GREEN) ---------------------------------------


@pytest.mark.parametrize("leg", ["tefas", "bes"])
def test_b2_non_business_day_panel_fails_closed(tmp_path, leg):
    """RED→GREEN: configured takvim-dışı/hafta sonu panel günü → guard, yazım yok, dosya değişmez."""
    cfg = load_config()
    holiday = pd.Timestamp(cfg["calendar"]["holidays"][0])
    dates = pd.DatetimeIndex([pd.Timestamp("2026-10-28"), holiday, pd.Timestamp("2026-10-31")])
    if leg == "tefas":
        fm = _make_fm(["F1"])
    else:
        fm = pd.DataFrame(
            {
                "fund_code": ["F1"],
                "fund_class": ["EMK"],
                "umbrella_type": ["Değişken"],
                "category": ["Değişken"],
                "withholding_rate": [0.0],
                "founder": ["Kurucu"],
                "name": ["Fon"],
                "manager": [""],
            }
        )
    nav = pd.DataFrame({"F1": [10.0, 10.0, 10.0]}, index=dates)
    out = tmp_path / "features.parquet"
    pd.DataFrame({"feature_asof": [dates[0]], "fund_code": ["F1"]}).to_parquet(out, index=False)
    before_bytes = out.read_bytes()

    with pytest.raises(RuntimeError, match="iş günü olmayan"):
        _features_build_impl(nav, fm, cfg, Store(tmp_path / "x.duckdb"), out, incremental=False, leg=leg)

    assert out.read_bytes() == before_bytes, "takvim ihlalinde dosya değişti"


def test_b2_business_day_guard_green_valid_panel_and_terminal(tmp_path):
    """GREEN: geçerli iş günü paneli + terminal next_business_day(cal[-1]) kabul; predictions yazılır."""
    cfg = load_config()
    nav = make_panel(n_funds=6, days=430, seed=11)
    cal = pd.DatetimeIndex(nav.index)
    fm = _make_fm(list(nav.columns))
    cash = pd.Series(0.0002, index=cal)
    feats = build_features(nav, fm, cash, _make_macro(cal), _make_market(cal), cfg=CFG, leg="tefas")
    out = tmp_path / "predictions.parquet"

    summary = _predictions_build_impl(feats, out, model_kwargs=MK, cfg=cfg)

    assert out.exists()
    preds = pd.read_parquet(out)
    terminal = next_business_day(cal[-1], cfg)
    assert pd.to_datetime(preds["decision_at"]).max() == terminal
    assert summary["train_max_t_ok"] is True
