"""S3b-4 suite testleri: sentetik panel + elle kalibrasyon; 5 satır, kullanılabilirlik, rapor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from _panel import make_panel
from janus.backtest.costs import meta_for_synthetic
from janus.backtest.select_suite import build_strategies, run_select_suite

CFG = {
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
CFG["legs"]["tefas"]["rebalance"] = {
    "schedule": "monthly_first_business_day",
    "drift_threshold": 0.03,
    "drift_threshold_taxable_sale": 0.06,
}
CFG["legs"]["tefas"]["universe"] = {"max_stale_days": 2}
CFG["legs"]["tefas"]["risk"] = {
    "dd_trigger_medium": 0.12,
    "dd_exposure": 0.65,
    "dd_release": 0.06,
    "exposure_levels": {"low": 0.30, "medium": 0.65, "full": 1.00},
    "suspension_haircut": 0.30,
}


def make_calibrated(nav: pd.DataFrame, positive_from: int = 0) -> pd.DataFrame:
    """Elle kalibrasyon: aylık ilk işlem günü decision_at; F0–F3 lower > 0, F4–F5 lower <= 0."""
    idx = nav.index
    firsts = pd.Series(np.arange(len(idx)), index=idx).groupby(idx.to_period("M")).min()
    dec_days = idx[firsts.to_numpy()]
    dec_days = dec_days[dec_days >= idx[260]]
    rows = []
    for d in dec_days:
        for i, c in enumerate(nav.columns):
            lower = 0.02 - 0.006 * i if i >= positive_from else -0.01
            rows.append(
                {
                    "decision_at": d,
                    "fund_code": c,
                    "lower": lower,
                    "upper": lower + 0.05,
                    "y": 0.01 if i < 4 else -0.01,
                    "alpha_D": 0.2,
                    "n_calib": 300,
                    "quality_flag": "ok",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def env():
    nav = make_panel(n_funds=6, days=500, seed=0)
    cash_r = pd.Series(0.0003, index=nav.index)
    fm = pd.DataFrame({"fund_code": list(nav.columns), "founder": "K"})
    d = {
        "nav": nav,
        "fund_master": fm,
        "meta": meta_for_synthetic(list(nav.columns)),
        "eligible": pd.Series(True, index=nav.columns),
        "cash_codes": [],
        "cash_returns": cash_r,
        "cash_nav": (1 + cash_r).cumprod(),
        "cash_category": None,
        "cash_rate": 0.175,
        "equity_index": nav.iloc[:, 0],
        "snapshot_asof": pd.Timestamp("2026-09-24 10:00"),
    }
    return d, make_calibrated(nav)


def test_build_strategies_five_rows(env):
    d, cal = env
    strats = build_strategies(d, CFG, top_n=3, calibrated=cal, kill_switch=False)
    names = [s.name for s in strats]
    assert names == ["B0_cash", "B2b_hrp_q_smooth", "B2c_q_conformal", "B2c_m_conformal", "B2c_m_conformal_taxpen"]
    assert strats[2].refresh_every == 3 and strats[3].refresh_every == 1
    assert strats[4].tax_penalty and not strats[3].tax_penalty


def test_build_strategies_with_fdr_row(env):
    """S3b-5-2: predictions/features verildiğinde B2c-fdr satırı eklenir (6 satır)."""
    d, cal = env
    nav = d["nav"]
    idx = nav.index
    preds, feats = [], []
    for i in range(len(idx) - 30):
        for c in nav.columns:
            preds.append(
                {
                    "decision_at": idx[i + 1],
                    "fund_code": c,
                    "q10": -0.01,
                    "q50": 0.0,
                    "q90": 0.01,
                    "model_id": idx[0],
                    "train_max_t": idx[0],
                }
            )
            feats.append(
                {
                    "feature_asof": idx[i],
                    "decision_at": idx[i + 1],
                    "fund_code": c,
                    "y": 0.01 if c == "F0" else -0.02,
                    "label_available_at": idx[i + 23],
                    "feature_ready": True,
                    "eligible_at_decision": True,
                }
            )
    strats = build_strategies(
        d,
        CFG,
        top_n=3,
        calibrated=cal,
        predictions=pd.DataFrame(preds),
        features=pd.DataFrame(feats),
        kill_switch=False,
    )
    names = [s.name for s in strats]
    assert names[-1] == "B2c_fdr_hrp" and len(names) == 6
    assert strats[-1].fdr_q == 0.20 and not strats[-1].kill_switch


def test_suite_runs_and_reports(tmp_path, env):
    d, cal = env
    d["cash_codes"] = ["F0", "F1"]
    profile_index = pd.MultiIndex.from_product([d["nav"].index, d["nav"].columns], names=["decision_date", "fund_code"])
    d["execution_meta_by_date"] = pd.DataFrame(
        {
            "buy_valor": 1,
            "sell_valor": 2,
            "entry_fee": 0.0,
            "exit_fee": 0.0,
            "tax_category": "diger",
            "can_buy": True,
            "can_sell": True,
            "buy_reason": "ok",
            "sell_reason": "ok",
            "status": "A3 varsayımlı, PIT kanıtı değil",
            "execution_source": "A3_CURRENT_PROFILE",
        },
        index=profile_index,
    )
    md, path = run_select_suite(
        d,
        CFG,
        top_n=3,
        select_end=str(d["nav"].index[400].date()),
        out_dir=tmp_path,
        quick=True,
        calibrated=cal,
        kill_switch=False,  # sentetik ortamda predictions/features yok — kill-switch testi test_conformal_select'te
    )
    assert path is not None and path.exists()
    text = path.read_text()
    assert "select_suite_" in text and "Deney kimliği" in text
    assert "veri as-of: 2026-09-24" in text  # raporda veri as-of satırı
    assert "Kapsama" in text and "12 başlangıç" in text and "PBO" in text
    assert "B2c_m_conformal" in text and "B2c_q_conformal" in text
    assert "B0: endeks vekili ve fon-lot satış valörü stresi (ADR-0023)" in text
    assert 'eski stres "üretilmedi, pp hesaplanamaz"' in text
    assert "icra edilebilir fon sepeti — satış valörü +1" in text
    assert "fon − proxy (pp)" in text and "tahsilat settle_idx" in text
    assert "A4 (valör yönü) kapalıdır" in text and "A5 (zarar mahsubu) açık kabul" in text
    assert "A4 (valör yönü) ve A5 (zarar mahsubu) açık kabul engelleridir" not in text
    assert "ok; A3 varsayımlı, PIT kanıtı değil" in text
    assert "sentetik dönem sonu tasfiye; öneri olayı değil" in text
    assert "değerlendirilemedi: dönem sonu bağımsız tasfiye/yeniden giriş yok" in text
    assert "fon-lot dönem getirisi değildir" in text
    b0_section = text.split("## B0: endeks vekili")[1].split("### Ufuk-içi sentetik")[0]
    for period in ("select", "external"):
        proxy_row = next(
            line for line in b0_section.splitlines() if line.startswith(f"| {period} | endeks vekili (referans)")
        )
        assert "endeks vekili (referans)" in proxy_row
        fund_rows = [line for line in b0_section.splitlines() if line.startswith(f"| {period} | B0 sabit terminal")]
        assert len(fund_rows) == 2
        assert all(
            "| — | — | değerlendirilemedi: dönem sonu bağımsız tasfiye/yeniden giriş yok;" in line for line in fund_rows
        )
    full_rows = [line for line in b0_section.splitlines() if line.startswith("| full | B0 sabit terminal")]
    assert len(full_rows) == 2 and all("| — |" not in line for line in full_rows)
    # (2) excess_cagr_*/cash_cagr_* yüzde biçiminde
    assert "%7.9" in text or "%0.0" in text  # _pct biçimi ana tabloda
    main_sec = text.split("## Kullanılabilirlik")[0]
    main_rows = [ln for ln in main_sec.splitlines() if ln.startswith("| B2c_m_conformal |")]
    assert main_rows and any(c.strip().startswith("%") for c in main_rows[0].split("|"))
    # (1) kullanılabilirlik tablosu satırları dolu; fallback_days görünür
    avail = text.split("## Kullanılabilirlik")[1]
    hdr = next(ln for ln in avail.splitlines() if ln.startswith("| strategy") or "aday_min_seç" in ln)
    for col in ("aday_min_seç", "aday_med_dış", "nakit_kalma", "ort_risky", "fallback_gün"):
        assert col in hdr
    arows = [ln for ln in avail.splitlines() if ln.startswith("| B2c")]
    assert len(arows) == 3
    for ln in arows:
        cells = [c.strip() for c in ln.split("|")]
        assert cells[8].startswith("%"), ln  # nakit_kalma yüzde biçiminde
        assert int(cells[10]) >= 0  # fallback_gün
    # (3) seçilen-fon kapsaması aylık tablo
    assert "### Seçilen-fon kapsama — B2c_m_conformal" in text
    cov_sec = text.split("### Seçilen-fon kapsama — B2c_m_conformal")[1].split("###")[0]
    assert "| month" in cov_sec or "month" in cov_sec
    assert cov_sec.count("\n") >= 3  # en az bir ay satırı
    # (4) aday sayısı dönem bazında sütunları
    assert "aday_min_seç" in hdr and "aday_min_dış" in hdr


def test_suite_reports_synthetic_b0_reinvestment_separately(tmp_path, env):
    d, cal = env
    d["cash_codes"] = ["F0", "F1"]
    profile_index = pd.MultiIndex.from_product([d["nav"].index, d["nav"].columns], names=["decision_date", "fund_code"])
    d["execution_meta_by_date"] = pd.DataFrame(
        {
            "buy_valor": 0,
            "sell_valor": 1,
            "entry_fee": 0.0,
            "exit_fee": 0.0,
            "tax_category": "diger",
            "can_buy": True,
            "can_sell": True,
            "buy_reason": "ok",
            "sell_reason": "ok",
            "status": "İşlem Görüyor",
            "execution_source": "A3_CURRENT_PROFILE",
        },
        index=profile_index,
    )
    _, path = run_select_suite(
        d,
        CFG,
        top_n=3,
        select_end=str(d["nav"].index[400].date()),
        out_dir=tmp_path,
        quick=True,
        calibrated=cal,
        kill_switch=False,
    )
    text = path.read_text()
    assert "B0 sabit terminal" in text
    assert "ufuk-içi sentetik yeniden yatırım duyarlılığı" in text
    assert "gerçek tarihsel PPF getirisi veya alfa kanıtı değildir" in text
    assert "vergili nakit yaklaşımı" in text
    assert "gerçek fon-lot nakit getirisi değildir" in text
    assert "kağıt defter/canlı üretim sonucu değildir" in text
    assert "A3 varsayımlı, PIT kanıtı değil" in text


def test_suite_keeps_b0_proxy_row_when_terminal_tax_category_is_missing(tmp_path, env):
    d, cal = env
    d["cash_codes"] = ["F0", "F1"]
    profile_index = pd.MultiIndex.from_product([d["nav"].index, d["nav"].columns], names=["decision_date", "fund_code"])
    profiles = pd.DataFrame(
        {
            "buy_valor": 0,
            "sell_valor": 1,
            "entry_fee": 0.0,
            "exit_fee": 0.0,
            "tax_category": "diger",
            "can_buy": True,
            "can_sell": True,
            "buy_reason": "ok",
            "sell_reason": "ok",
            "status": "İşlem Görüyor",
            "execution_source": "A3_CURRENT_PROFILE",
        },
        index=profile_index,
    )
    profiles.loc[(d["nav"].index[-1], slice(None)), "tax_category"] = "  "
    d["execution_meta_by_date"] = profiles

    _, path = run_select_suite(
        d,
        CFG,
        top_n=3,
        select_end=str(d["nav"].index[400].date()),
        out_dir=tmp_path,
        quick=True,
        calibrated=cal,
        kill_switch=False,
    )

    text = path.read_text()
    b0_section = text.split("## B0: endeks vekili")[1].split("### Ufuk-içi sentetik")[0]
    proxy_rows = [line for line in b0_section.splitlines() if "endeks vekili (referans)" in line]
    fund_rows = [line for line in b0_section.splitlines() if line.startswith("|") and "B0 sabit terminal" in line]
    assert len(proxy_rows) == 3
    assert len(fund_rows) == 6
    assert all("değerlendirilemedi:" in line for line in fund_rows)


def test_suite_cash_stay_when_no_evidence(tmp_path, env):
    d, cal = env
    cal0 = cal.copy()
    cal0["lower"] = -0.01  # kanıt yok → tüm B2c satırları nakitte
    md, _ = run_select_suite(
        d,
        CFG,
        top_n=3,
        select_end=str(d["nav"].index[400].date()),
        out_dir=None,
        quick=True,
        calibrated=cal0,
        kill_switch=False,
    )
    avail = md.split("## Kullanılabilirlik")[1]
    rows = [ln for ln in avail.splitlines() if ln.startswith("| B2c")]
    assert rows
    for ln in rows:
        cells = [c.strip() for c in ln.split("|")]
        assert cells[8] == "%100.0", ln  # nakit_kalma sütunu
        assert float(cells[9]) == 0.0  # ort_risky
        assert int(cells[10]) == 0  # fallback_gün
    # dış test döneminde aday yok (tüm lower <= 0 zaten); seçim döneminde de 0
    assert all(c.strip() in ("0", "0.0") for ln in rows for c in ln.split("|")[2:8])
