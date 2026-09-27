"""S3b-5-1 testleri: kapsama açığı ayrıştırımı tabloları, maske ayrımı, dönem özeti, yorum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from janus.models.coverage_diag import _selection_masks, build_report, coverage_diag_tables


def make_calibrated(days=8, funds=30, seed=0) -> pd.DataFrame:
    """Sentetik: her gün 30 fon; ilk 20 fon lower > 0 (aday havuzu), ilk 10 seçilenler y > upper (kapsama 0)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=days + 1)
    rows = []
    for i in range(days):
        d = idx[i + 1]
        for j in range(funds):
            lower = 0.01 + 0.001 * (20 - j) if j < 20 else -0.01
            y = lower + 0.06 if j < 10 else rng.normal(0, 0.02)  # seçilenlerde y > upper → kapsama 0
            rows.append(
                {
                    "decision_at": d,
                    "fund_code": f"F{j:02d}",
                    "lower": lower,
                    "upper": lower + 0.05,
                    "y": y,
                    "alpha_D": 0.2,
                    "n_calib": 300,
                    "quality_flag": "ok",
                }
            )
    return pd.DataFrame(rows)


def test_mask_nesting_and_sizes():
    cal = make_calibrated()
    top, rand = _selection_masks(cal, top_n=10, seed=0)
    pos = cal["lower"].notna() & (cal["lower"] > 0)
    assert (top & ~pos).sum() == 0 and (rand & ~pos).sum() == 0  # top10 ⊆ pos, rand10 ⊆ pos
    for _, g in cal.groupby(pd.to_datetime(cal["decision_at"])):
        assert top.loc[g.index].sum() == 10 and rand.loc[g.index].sum() == 10
    # tohum sabit → rand10 tekrarlanabilir
    top2, rand2 = _selection_masks(cal, top_n=10, seed=0)
    assert top.equals(top2) and rand.equals(rand2)
    # farklı tohum → farklı küme (30 adaydan 10 seçim; çakışma olasılığı düşük)
    _, rand3 = _selection_masks(cal, top_n=10, seed=1)
    assert not rand.equals(rand3)


def test_tables_and_period_split():
    cal = make_calibrated(days=8)
    sel_end = pd.Timestamp(cal["decision_at"].unique()[3])
    tables, summary = coverage_diag_tables(cal, sel_end, top_n=10, seed=0)
    assert set(tables) == {"all", "pos", "top10", "rand10"}
    for t in tables.values():
        assert {"month", "coverage", "width", "n_rows", "period"} <= set(t.columns)
        assert ((t["coverage"] >= 0) & (t["coverage"] <= 1)).all()
        assert (t["width"] > 0).all()  # genişlik = upper − lower
    # dönem ayrımı: seçim/dış satırları
    assert set(summary["period"]) == {"seçim", "dış"}
    assert set(summary["series"]) == {"all", "pos", "top10", "rand10"}
    # seçilenlerde y > lower → kapsama 0; tüm satırlarda > 0
    sel = summary.set_index(["series", "period"])
    assert sel.loc[("top10", "seçim"), "coverage_mean"] == pytest.approx(0.0)
    assert sel.loc[("all", "seçim"), "coverage_mean"] > 0.3  # havuz karışımı
    # seçim etkisi: top10 kapsama < rand10 (sentetik kurgu: seçilenlerde y üstte)
    assert sel.loc[("top10", "seçim"), "coverage_mean"] < sel.loc[("rand10", "seçim"), "coverage_mean"]


def test_build_report_comment_and_sections():
    cal = make_calibrated(days=8)
    sel_end = pd.Timestamp(cal["decision_at"].unique()[3])
    t_a, s_a = coverage_diag_tables(cal, sel_end)
    t_f, s_f = coverage_diag_tables(cal, sel_end)  # sabit-α frame'i aynı alınabilir (yapı testi)
    md = build_report(t_a, s_a, t_f, s_f, "coverage_diag_test", "veri as-of: test", sel_end)
    for key in ("ACI (gerçek α yolu)", "Sabit α = hedef (ACI kapalı)", "Rastgele 10", "Dönem özeti", "## Yorum"):
        assert key in md
    assert "Baskın kaynak" in md and "Havuz etkisi" in md and "Seçim etkisi" in md
    assert "|havuz|" in md and "|seçim|" in md and "|ACI|" in md  # üçlü karşılaştırma
    # sentetik kurguda havuz koşullanması baskın (top10 kapsama 0, havuz karışımı ~0.3)
    assert "lower > 0 havuz koşullanması" in md.split("Baskın kaynak")[1]
