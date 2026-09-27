"""S3b-5-1: kapsama açığı ayrıştırımı (tanı; γ/α tuning DEĞİL).

Aylık kapsama + ortalama aralık genişliği (upper − lower) üç seri üzerinden:
(a) tüm aralıklı satırlar, (b) lower > 0 satırlar (aday havuzu), (c) seçilen top-10
(gün başına lower sırası; B2c-m iskeleti, HRP tavanları hariç) ve (c2) rastgele 10
(lower > 0 arasından, tohum sabit) — seçim etkisini (sıralama) aday havuzu etkisinden ayırır.
Karşı-olgusal: α sabit = hedef (ACI kapalı; calibrate_predictions gamma=0.0) ile aynı dörtlü.
Dönem ayrımı: seçim (decision_at <= select_end) / dış test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SERIES = ("all", "pos", "top10", "rand10")
SERIES_LABEL = {
    "all": "(a) Tüm aralıklı satırlar",
    "pos": "(b) lower > 0 (aday havuzu)",
    "top10": "(c) Seçilen top-10",
    "rand10": "(c2) Rastgele 10 (aday havuzundan)",
}


def _selection_masks(calibrated: pd.DataFrame, top_n: int, seed: int) -> tuple[pd.Series, pd.Series]:
    top = pd.Series(False, index=calibrated.index)
    rand = pd.Series(False, index=calibrated.index)
    rng = np.random.default_rng(seed)
    for _, g in calibrated.groupby(pd.to_datetime(calibrated["decision_at"]), sort=True):
        cand = g.index[g["lower"].notna() & (g["lower"] > 0)]
        if len(cand) == 0:
            continue
        top.loc[g.loc[cand, "lower"].sort_values(ascending=False).index[:top_n]] = True
        rand.loc[rng.choice(cand, size=min(top_n, len(cand)), replace=False)] = True
    return top, rand


def _monthly(calibrated: pd.DataFrame, mask: pd.Series, sel_end: pd.Timestamp) -> pd.DataFrame:
    df = calibrated[mask & calibrated["y"].notna() & calibrated["lower"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["month", "coverage", "width", "n_rows", "period"])
    y = df["y"].to_numpy(float)
    inside = (df["lower"].to_numpy(float) <= y) & (y <= df["upper"].to_numpy(float))
    width = df["upper"].to_numpy(float) - df["lower"].to_numpy(float)
    g = pd.DataFrame(
        {
            "month": pd.to_datetime(df["decision_at"]).dt.to_period("M").astype(str),
            "coverage": inside.astype(float),
            "width": width,
            "period": np.where(pd.to_datetime(df["decision_at"]) <= sel_end, "seçim", "dış"),
        }
    )
    return (
        g.groupby(["period", "month"], sort=True)
        .agg(coverage=("coverage", "mean"), width=("width", "mean"), n_rows=("coverage", "size"))
        .reset_index()
    )


def coverage_diag_tables(
    calibrated: pd.DataFrame, sel_end: pd.Timestamp, top_n: int = 10, seed: int = 0
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Dört serinin aylık tabloları + dönem özeti (series, period, coverage_mean, width_mean)."""
    top, rand = _selection_masks(calibrated, top_n, seed)
    masks = {
        "all": pd.Series(True, index=calibrated.index),
        "pos": calibrated["lower"].notna() & (calibrated["lower"] > 0),
        "top10": top,
        "rand10": rand,
    }
    tables = {k: _monthly(calibrated, m, sel_end) for k, m in masks.items()}
    rows = []
    for k, t in tables.items():
        for period, g in t.groupby("period"):
            rows.append(
                {
                    "series": k,
                    "period": period,
                    "coverage_mean": float(g["coverage"].mean()),
                    "width_mean": float(g["width"].mean()),
                    "n_months": int(len(g)),
                }
            )
    return tables, pd.DataFrame(rows)


def _fmt(x: float, pct: bool = True) -> str:
    if pd.isna(x):
        return "—"
    return f"%{x * 100:.1f}" if pct else f"{x:.4f}"


def build_report(
    tables_aci: dict[str, pd.DataFrame],
    summary_aci: pd.DataFrame,
    tables_fixed: dict[str, pd.DataFrame],
    summary_fixed: pd.DataFrame,
    exp_id: str,
    asof_line: str,
    sel_end: pd.Timestamp,
) -> str:
    """Markdown rapor: ACI ve sabit-α koşularında dört serinin aylık tabloları + dönem özeti + 3 satır yorum."""
    md = [
        f"# Kapsama açığı ayrıştırımı — {exp_id}",
        "",
        f"Seçim dönemi: decision_at <= {sel_end.date()} · dış test: sonrası · γ/α tuning YOK (tanı amaçlı).",
        "",
        f"- {asof_line}",
        "",
    ]
    for tag, tables, summary in (
        ("ACI (gerçek α yolu)", tables_aci, summary_aci),
        ("Sabit α = hedef (ACI kapalı)", tables_fixed, summary_fixed),
    ):
        md += [f"## {tag}", ""]
        for k in SERIES:
            t = tables[k]
            md += [f"### {SERIES_LABEL[k]}", ""]
            if t.empty:
                md += ["(satır yok)", ""]
                continue
            tbl = t.copy()
            tbl["coverage"] = tbl["coverage"].map(_fmt)
            tbl["width"] = tbl["width"].map(lambda v: _fmt(v, pct=False))
            md += [tbl.to_markdown(index=False), ""]
        if len(summary):
            sm = summary.copy()
            sm["coverage_mean"] = sm["coverage_mean"].map(_fmt)
            sm["width_mean"] = sm["width_mean"].map(lambda v: _fmt(v, pct=False))
            md += [f"**Dönem özeti ({tag})**", "", sm.to_markdown(index=False), ""]

    # 3 satırlık yorum — sayıya dayalı
    sa = summary_aci.set_index(["series", "period"])
    sf = summary_fixed.set_index(["series", "period"])
    n_fix_raw = summary_fixed

    def get(df: pd.DataFrame, series: str, period: str, col: str) -> float:
        try:
            return float(df.loc[(series, period), col])
        except KeyError:
            return float("nan")

    sel_gap = get(sa, "top10", "dış", "coverage_mean") - get(sa, "rand10", "dış", "coverage_mean")
    pool_gap = get(sa, "rand10", "dış", "coverage_mean") - get(sa, "all", "dış", "coverage_mean")
    aci_gap = get(sa, "top10", "dış", "coverage_mean") - get(sf, "top10", "dış", "coverage_mean")
    w_aci = get(sa, "top10", "dış", "width_mean")
    w_fix = get(sf, "top10", "dış", "width_mean")
    # sabit-α karşılaştırması yalnız iki koşuda da yeterli ay varsa sonuç verir
    n_fix = int(n_fix_raw.loc[n_fix_raw["series"] == "top10", "n_months"].sum()) if len(n_fix_raw) else 0
    aci_note = "" if n_fix >= 6 else f" (sabit-α havuzu {n_fix} ay — karşılaştırma sınırlı sonuç verir; ikincil)"
    gaps = {
        "lower > 0 havuz koşullanması": abs(pool_gap),
        "seçim (sıralama)": abs(sel_gap),
        "ACI daralması": abs(aci_gap),
    }
    dominant = max(gaps, key=gaps.get)
    md += [
        "## Yorum (sayıya dayalı)",
        "",
        f"1. Havuz etkisi (dış test, rand10 − tüm kapsama): {_fmt(pool_gap)}; Seçim etkisi (top10 − rand10): {_fmt(sel_gap)}.",
        f"2. ACI etkisi (dış test, top10: ACI − sabit α): {_fmt(aci_gap)}; ortalama genişlik ACI {_fmt(w_aci, pct=False)} vs sabit {_fmt(w_fix, pct=False)}{aci_note}.",
        f"3. Baskın kaynak: **{dominant}** (|havuz|={gaps['lower > 0 havuz koşullanması']:.3f} vs |seçim|={gaps['seçim (sıralama)']:.3f} vs |ACI|={gaps['ACI daralması']:.3f}, dış test kapsaması üzerinden).",
        "",
    ]
    return "\n".join(md)
