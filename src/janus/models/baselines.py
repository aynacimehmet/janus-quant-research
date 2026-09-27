"""Baseline tahminler (S3b-2): modelle aynı OOS satır kümesinde, aynı ufuk (21g) ve aynı α'larda.

- 0-tahmin: q_α = 0 (nakit-fazlası uzayı).
- 63g ortalama günlük nakit-fazlası × 21: günlük nakit-fazlası = cash_excess_21.diff()
  (log uzayında; beyaz liste kolonlarından türetilir, as-of güvenli).
- Tarihsel kuantil: olgunlaşmış y'lerin (label_available_at <= D ⇔ t <= D-23) son 252 işlem
  günü empirik α-kuantilleri; kaydırma 22 (t-22 konumundaki değer karar D = cal[t+1]'de olgun).
"""

from __future__ import annotations

import pandas as pd


def _pivot(features: pd.DataFrame, col: str) -> pd.DataFrame:
    return features.pivot_table(index="feature_asof", columns="fund_code", values=col, aggfunc="last")


def _stack_long(wide: dict[float, pd.DataFrame], mask: pd.Series, features: pd.DataFrame) -> pd.DataFrame:
    """Geniş α-kuantil çerçevelerini maskeye bağlı uzun kayıtlara çevirir (modelle aynı satırlar)."""
    keys = features.index[mask]
    idx = pd.MultiIndex.from_frame(features.loc[keys, ["feature_asof", "fund_code"]])
    out = pd.DataFrame(index=idx)
    for a, w in wide.items():
        long = w.stack()
        out[f"q{int(round(a * 100))}"] = long.reindex(idx).to_numpy()
    out = out.reset_index()
    dec = features.loc[keys, ["feature_asof", "decision_at"]].drop_duplicates()
    out = out.merge(dec, on="feature_asof", how="left")
    return out[["decision_at", "fund_code", *(f"q{int(round(a * 100))}" for a in wide)]]


def zero_baseline(features: pd.DataFrame, mask: pd.Series, alphas: tuple[float, ...] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
    wide = {
        a: pd.DataFrame(0.0, index=features["feature_asof"].unique(), columns=features["fund_code"].unique())
        for a in alphas
    }
    return _stack_long(wide, mask, features)


def cash_excess_baseline(features: pd.DataFrame, mask: pd.Series, window: int = 63, horizon: int = 21) -> pd.DataFrame:
    """63g ortalama günlük nakit-fazlası × 21 (nokta tahmin; tüm α'larda aynı)."""
    w = _pivot(features, "cash_excess_21").sort_index()
    daily = w.diff()  # cash_excess_21_t - cash_excess_21_{t-1} = log_ret_1_t - log1p(cash_t)
    base = daily.rolling(window, min_periods=window).mean() * horizon
    wide = {a: base for a in (0.1, 0.5, 0.9)}
    return _stack_long(wide, mask, features)


def empirical_quantile_baseline(
    features: pd.DataFrame,
    mask: pd.Series,
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
    window: int = 252,
    min_obs: int = 63,
) -> pd.DataFrame:
    """Olgunlaşmış y'lerin son 252g empirik kuantilleri; yeterli gözlem yoksa tahmin yok (NaN)."""
    yw = _pivot(features, "y").sort_index()
    wide = {a: yw.rolling(window, min_periods=min_obs).quantile(a).shift(22) for a in alphas}
    return _stack_long(wide, mask, features)


def baseline_predictions(
    features: pd.DataFrame,
    mask: pd.Series,
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
    min_decision_at: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Üç baseline'ı modelle aynı OOS satır kümesinde üretir; anahtarlar: zero, cash_excess, empirical.

    min_decision_at: modelin ilk refit'i (eğitim verisi olan ilk ay) — önceki aylar modelde yok,
    baseline'dan da atılır ki satır kümeleri birebir aynı kalsın.
    """
    if min_decision_at is not None:
        mask = mask & (features["decision_at"] >= min_decision_at)
    return {
        "zero": zero_baseline(features, mask, alphas),
        "cash_excess": cash_excess_baseline(features, mask),
        "empirical": empirical_quantile_baseline(features, mask, alphas),
    }
