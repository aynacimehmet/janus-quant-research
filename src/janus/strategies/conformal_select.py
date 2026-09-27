"""S3b-4: ConformalHRP — skor = conformal alt sınır (lower), kill-switch (TEMPORAL §7).

- Skor (karar günü D): S3b-3 kalibrasyonundan (target 0.20) `decision_at == D` satırlarının
  `lower` değeri. `lower <= 0` veya NaN → aday değil (ADR-18); aday yoksa portföy nakit
  sepetinde kalır (motorun defter içi slotu, P02).
- `tax_penalty`: yalnızca sıralama cezası `lower × (1 − stopaj)`; "vergi sonrası getiri"
  DENMEZ (H10) — gerçek vergi defterde, lot bazında.
- Kill-switch (TEMPORAL §7): son `ks_window` işlem gününün OOS satırlarında
  (`label_available_at <= D`) pinball(model) >= pinball(tarihsel-kuantil baseline, aynı
  satır kümesi ve α) ise seçim skoru momentum'a düşer (B2b iskeletiyle aynı skorlar).
  Karşılaştırma yapılamıyorsa (baseline/model NaN) bayrak False — sessiz fallback yok.
- Fon sırası invariance: skor fon başına skaler; seçim/HRP fon sırasından bağımsız.
- Seçim dönemi / dış test ayrımı suite katmanındadır (VALIDATION §1); bu modül tarih
  bilgisi dışında dönem bilgisi gerektirmez.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from janus.strategies.portfolio import MomentumHRP


def kill_switch_flags(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    window: int = 126,
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> pd.Series:
    """TEMPORAL §7: karar günü D için True/False; yalnız `label_available_at <= D` OOS satırları.

    Baseline: S3b-2 tarihsel-kuantil baseline (empirical_quantile_baseline), aynı satır
    kümesi ve α'lar; model = WF q10/q50/q90. Skor: α'lar üzerinden ortalama pinball.
    Karşılaştırma olanaksızsa (model veya baseline NaN) → False.
    """
    from janus.models.baselines import empirical_quantile_baseline  # noqa: PLC0415
    from janus.models.metrics import pinball_loss  # noqa: PLC0415

    f = predictions.merge(
        features[["feature_asof", "fund_code", "decision_at", "y", "label_available_at"]],
        on=["decision_at", "fund_code"],
        how="left",
        validate="one_to_one",
    )
    min_model = pd.Timestamp(predictions["model_id"].min())
    mask = (
        features["feature_ready"].fillna(False).astype(bool)
        & features["eligible_at_decision"].fillna(False).astype(bool)
        & features["y"].notna()
        & pd.to_datetime(features["label_available_at"]).notna()
        & (pd.to_datetime(features["decision_at"]) >= min_model)  # modelle aynı satır kümesi (S3b-2)
    )
    base = empirical_quantile_baseline(features, mask, alphas=alphas)
    f = f.merge(base, on=["decision_at", "fund_code"], how="left", suffixes=("", "_b"))
    nav_cal = pd.DatetimeIndex(np.sort(pd.to_datetime(features["feature_asof"].dropna().unique())))
    pos_of = pd.Series(np.arange(len(nav_cal)), index=nav_cal)
    t_pos = pd.to_datetime(f["feature_asof"]).map(pos_of).to_numpy()
    y = f["y"].to_numpy(float)
    qm = np.column_stack([f[f"q{int(round(a * 100))}"].to_numpy(float) for a in alphas])
    qb = np.column_stack([f[f"q{int(round(a * 100))}_b"].to_numpy(float) for a in alphas])
    # satırları t_pos'a göre sırala → her karar günü için olgunlaşmış pencere dilimi searchsorted ile
    order = np.argsort(t_pos, kind="stable")
    t_sorted = t_pos[order]
    flags: dict[pd.Timestamp, bool] = {}
    for d in pd.DatetimeIndex(np.sort(pd.to_datetime(f["decision_at"].dropna().unique()))):
        if d not in pos_of.index:
            continue
        pos_d = int(pos_of.loc[d])
        # OOS satırları: label_available_at <= D (⇔ t <= D-23) ve olgunlaşma son `window` işlem günü içinde
        lo, hi = np.searchsorted(t_sorted, [pos_d - window - 23, pos_d - 23 + 1])
        idx = order[lo:hi]
        idx = idx[~np.isnan(y[idx])]
        if len(idx) == 0:
            flags[d] = False
            continue
        y_m = y[idx]
        pm = np.nanmean([pinball_loss(y_m, qm[idx, j], a) for j, a in enumerate(alphas)])
        pb = np.nanmean([pinball_loss(y_m, qb[idx, j], a) for j, a in enumerate(alphas)])
        flags[d] = bool(np.isfinite(pm) and np.isfinite(pb) and pm >= pb)
    return pd.Series(flags, name="kill_switch")


@dataclass
class ConformalHRP(MomentumHRP):
    """B2c: conformal alt sınır skoru + HRP + tavanlar; kanıt yoksa nakit (ADR-18)."""

    calibrated: pd.DataFrame | None = None  # decision_at, fund_code, lower (target 0.20; y/upper kapsama için)
    predictions: pd.DataFrame | None = None  # kill-switch için: q10/q50/q90, model_id
    features: pd.DataFrame | None = None  # kill-switch için: y, label_available_at, feature_asof
    kill_switch: bool = True
    ks_window: int = 126
    tax_penalty: bool = False  # sezgisel sıralama cezası: lower × (1 − stopaj)
    name: str = "B2c_conformal_hrp"
    _lower_by_day: dict = field(default_factory=dict, repr=False)
    _ks: pd.Series | None = field(default=None, repr=False)
    _fallback_count: int = field(default=0, repr=False)
    _history: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        cal = self.calibrated
        if cal is None or cal.empty:
            raise ValueError("calibrated boş — S3b-3 calibrated_target_020.parquet gerekli")
        missing = {"decision_at", "fund_code", "lower"} - set(cal.columns)
        if missing:
            raise ValueError(f"calibrated kolonları eksik: {sorted(missing)}")
        # F-06: stratejiye yalnızca beyaz listeli kolonlar görünür; y/upper sızıntı riski kapatılır
        cal = cal[["decision_at", "fund_code", "lower"]].copy()
        self._lower_by_day = {
            pd.Timestamp(d): g.set_index("fund_code")["lower"].astype(float)
            for d, g in cal.groupby(pd.to_datetime(cal["decision_at"]))
        }
        if self.kill_switch:
            if self.predictions is None or self.features is None:
                raise ValueError("kill_switch=True → predictions ve features gerekli (TEMPORAL §7)")
            self._ks = kill_switch_flags(self.predictions, self.features, window=self.ks_window)

    @property
    def fallback_days(self) -> int:
        """Kill-switch'in momentum fallback'ine düşürdüğü rebalance gün sayısı."""
        return self._fallback_count

    def scores(self, ctx) -> pd.Series:  # noqa: ANN001 — StrategyContext (döngüsel import yok)
        d = pd.Timestamp(ctx.date)
        if self._ks is not None and bool(self._ks.get(d, False)):
            self._fallback_count += 1
            return super().scores(ctx)  # TEMPORAL §7: skor momentum'a düşer (B2b iskeleti)
        s = self._lower_by_day.get(d)
        if s is None:
            return pd.Series(dtype=float)
        s = s.dropna()
        s = s[s > 0]  # lower <= 0 → aday değil (ADR-18)
        if self.tax_penalty:
            tax = pd.Series(ctx.meta.tax_rate, index=ctx.meta.index).reindex(s.index).fillna(0.175)
            s = s * (1.0 - tax)  # yalnız sıralama cezası; "vergi sonrası getiri" denmez (H10)
        return s

    def _compute(self, ctx) -> pd.Series:  # noqa: ANN001
        w = super()._compute(ctx)
        chosen = list(self._last.get("chosen", [])) if self._last.get("date") == ctx.date else []
        self._history[pd.Timestamp(ctx.date)] = chosen
        return w


def candidate_counts(calibrated: pd.DataFrame) -> pd.Series:
    """Gün başına aday sayısı (lower > 0); nakitte kalma oranı ve kullanılabilirlik için."""
    s = calibrated.dropna(subset=["lower"])
    s = s[s["lower"] > 0]
    return s.groupby(pd.to_datetime(s["decision_at"])).size()


def selected_mask_from_history(calibrated: pd.DataFrame, history: dict) -> pd.Series:
    """Seçilen-fon kapsaması için maske (calibrated indeksiyle hizalı bool)."""
    mask = pd.Series(False, index=calibrated.index)
    dec = pd.to_datetime(calibrated["decision_at"])
    for d, chosen in history.items():
        if chosen:
            mask |= (dec == pd.Timestamp(d)) & calibrated["fund_code"].isin(chosen)
    return mask
