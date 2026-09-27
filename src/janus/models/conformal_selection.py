"""S3b-5-2: B2c-fdr — seçim-ayarlı conformal selection (Jin & Candès 2023).

H0_j: y_j ≤ 0 (fon j'nin 21g nakit-fazlası pozitif değil). Skor V(x, y) = y − q50(x) (y'de artan).
Test birimi j (karar günü D): V̂_j = V(x_j, c=0) = −q50_j (yalnız tahmin; y_j bilinmez).
Kalibrasyon birimleri i (TEMPORAL §6 olgunlaşma kuralı: label_available_at ≤ D,
t ≥ D − calib_window − 23, n_min 200): V_i = y_i − q50_i; "clipped" varyant (daha güçlü):
y_i > 0 olan kalibrasyon birimlerinde V_i = +∞.
p_j = (1 + #{i: V_i ≤ V̂_j}) / (n + 1) — büyük q50_j → küçük V̂_j → az sayıda V_i altında
kalır → küçük p → H0 reddedilir → aday. Bağlar: muhafazakâr "≤" (varsayılan) veya
U ~ U(0,1) rastgeleleştirme (tohum sabit). Sonra Benjamini–Hochberg FDR q (config conformal.fdr_q).

Varsayım notu (rapora yazılır, iddia EDİLMEZ): aynı gün fonlar bağımlı (piyasa faktörü);
BH'nin PRDS koşulu altındaki geçerliliği bu panelde garanti edilmez — ampirik FDR raporlanır.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from janus.models.conformal import CALIB_WINDOW, LABEL_LAG, N_MIN, _decision_position
from janus.strategies.conformal_select import ConformalHRP


def conformal_pvalues(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    calib_window: int = CALIB_WINDOW,
    n_min: int = N_MIN,
    clipped: bool = False,
    ties: str = "conservative",  # "conservative" (≤) | "randomized" (U(0,1), tohum sabit)
    seed: int = 0,
) -> pd.DataFrame:
    """Günlük conformal p-değerleri (H0: y ≤ 0). Çıktı: decision_at, fund_code, q50, p_value, n_calib, quality_flag."""
    if ties not in ("conservative", "randomized"):
        raise ValueError(f"bilinmeyen ties: {ties}")
    f = predictions.merge(
        features[["feature_asof", "fund_code", "decision_at", "y", "label_available_at"]],
        on=["decision_at", "fund_code"],
        how="left",
        validate="one_to_one",
    )
    nav_cal = pd.DatetimeIndex(np.sort(pd.to_datetime(features["feature_asof"].dropna().unique())))
    pos_of = pd.Series(np.arange(len(nav_cal)), index=nav_cal)
    t_pos = pd.to_datetime(f["feature_asof"]).map(pos_of).to_numpy()
    y = f["y"].to_numpy(float)
    q50 = f["q50"].to_numpy(float)
    laa = pd.to_datetime(f["label_available_at"]).to_numpy()
    rng = np.random.default_rng(seed)
    out = f[["decision_at", "fund_code", "q50"]].copy()
    out["p_value"] = np.nan
    out["n_calib"] = 0
    out["quality_flag"] = "ok"
    # kalibrasyon birimleri D'nin KENDİ karar satırları DEĞİLDİR: label_available_at ≤ D olan
    # (⇔ t ≤ D−23) satırlar kendi geçmiş karar günlerine aittir (MB-031/MB-032 ortak kural).
    # Satırlar t_pos'a göre sıralanır → pencere dilimi searchsorted ile.
    order = np.argsort(t_pos, kind="stable")
    t_sorted = t_pos[order]
    for d, idx in f.groupby("decision_at").indices.items():
        d = pd.Timestamp(d)
        pos_d = _decision_position(f, d, nav_cal, pos_of)
        if pos_d is None:
            continue
        lo, hi = np.searchsorted(t_sorted, [pos_d - calib_window - LABEL_LAG, pos_d - LABEL_LAG + 1])
        m_cal = order[lo:hi]
        m_cal = m_cal[~np.isnan(y[m_cal]) & ~np.isnan(q50[m_cal]) & (laa[m_cal] <= d)]
        n = len(m_cal)
        rows = np.where(f["decision_at"].to_numpy() == np.datetime64(d))[0]
        out.loc[f.index[rows], "n_calib"] = n
        if n < n_min:
            out.loc[f.index[rows], "quality_flag"] = "insufficient_calibration"
            continue
        v_cal = y[m_cal] - q50[m_cal]
        if clipped:
            v_cal = np.where(y[m_cal] > 0, np.inf, v_cal)
        v_hat = -q50[idx]
        nan_hat = np.isnan(v_hat)
        out.loc[f.index[rows[nan_hat]], "quality_flag"] = "missing_q50"
        if not nan_hat.all():
            ok = ~nan_hat
            v_hat_ok = v_hat[ok]
            if ties == "conservative":
                cnt = np.array([(v_cal <= vj).sum() for vj in v_hat_ok])
            else:  # rastgeleleştirme: p = (1 + #{<} + U·#{==}) / (n+1), U ~ U(0,1) tohum sabit
                cnt = np.array([(v_cal < vj).sum() + float(rng.random()) * (v_cal == vj).sum() for vj in v_hat_ok])
            p = (1.0 + cnt) / (n + 1.0)
            out.loc[f.index[rows[ok]], "p_value"] = np.clip(p, 0.0, 1.0)
    return out


def bh_select(pvals: pd.Series, q: float) -> list[str]:
    """Benjamini–Hochberg adım-up: en büyük k öyle ki p_(k) ≤ k·q/m; boş küme → []."""
    pv = pvals.dropna()
    m = len(pv)
    if m == 0 or q <= 0:
        return []
    order = pv.sort_values(kind="stable")
    thresh = q * np.arange(1, m + 1) / m
    ok = order.to_numpy() <= thresh
    if not ok.any():
        return []
    k = int(np.max(np.nonzero(ok)[0])) + 1
    return [str(c) for c in order.index[:k]]


def selected_positive_rate(
    calibrated: pd.DataFrame, history: dict, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> tuple[float, float]:
    """Seçilen kümede gerçekleşen y > 0 oranı (ampirik 1−FDR karşılığı) ve satır sayısı; dönem penceresi opsiyonel."""
    dec = pd.to_datetime(calibrated["decision_at"])
    ys = []
    for d, chosen in history.items():
        if not chosen:
            continue
        d = pd.Timestamp(d)
        if start is not None and d <= start:
            continue
        if end is not None and d > end:
            continue
        m = (dec == d) & calibrated["fund_code"].isin(chosen)
        y = calibrated.loc[m, "y"].dropna()
        if len(y):
            ys.append(y)
    if not ys:
        return float("nan"), 0.0
    y = pd.concat(ys)
    return float((y > 0).mean()), float(len(y))


@dataclass
class FdrHRP(ConformalHRP):
    """B2c-fdr: adaylık = BH kümesi (p-değeri + FDR q); HRP sıralama skoru = lower."""

    fdr_q: float = 0.20
    clipped: bool = False
    ties: str = "conservative"
    pvalues: pd.DataFrame | None = None  # önceden hesaplanmış p-değerleri (suite q ızgarasında bir kez)
    name: str = "B2c_fdr_hrp"
    _bh: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.predictions is None or self.features is None:
            raise ValueError("FdrHRP → predictions ve features gerekli (p-değeri)")
        pv = (
            self.pvalues
            if self.pvalues is not None
            else conformal_pvalues(self.predictions, self.features, clipped=self.clipped, ties=self.ties)
        )
        self._bh = {
            pd.Timestamp(d): set(bh_select(g.set_index("fund_code")["p_value"], self.fdr_q))
            for d, g in pv.groupby(pd.to_datetime(pv["decision_at"]))
        }
        super().__post_init__()

    def scores(self, ctx) -> pd.Series:  # noqa: ANN001
        d = pd.Timestamp(ctx.date)
        bh = self._bh.get(d, set())
        if not bh:
            return pd.Series(dtype=float)  # küme boş → nakit (ADR-18)
        s = self._lower_by_day.get(d, pd.Series(dtype=float)).reindex(sorted(bh)).dropna()
        if self.tax_penalty:
            tax = pd.Series(ctx.meta.tax_rate, index=ctx.meta.index).reindex(s.index).fillna(0.175)
            s = s * (1.0 - tax)
        return s
