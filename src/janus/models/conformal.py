"""CQR + ACI-tarzı günlük global α (S3b-3) — TEMPORAL §6 birebir.

- Kalibrasyon seti (karar günü D): OOS tahmin satırları, `label_available_at <= D`,
  `t >= D - calib_window - 23` (işlem-günü indeksi üzerinden; takvim günü değil).
- Skor: `max(q_lo - y, y - q_hi)` (q_lo=q10, q_hi=q90). Sıralı istatistik
  `k = ceil((n+1)(1-alpha_D))`; `n < n_min` → tahmin yok; `k > n` → tahmin yok (aday olamaz).
- `lower = q_lo - s_(k)`, `upper = q_hi + s_(k)`.
- Crossing yalnız `q10 > q90`: uçlar sıralanır, quality_flag'e "crossing" yazılır;
  `q10 > q50` / `q50 > q90` tek başına crossing DEĞİLDİR; q50 dokunulmaz.
- ACI-tarzı günlük global α: `alpha_{D+1} = clip(alpha_D + gamma*(miscoverage_target - err_D),
  alpha_min, alpha_max)`; err_D = o gün olgunlaşan (t = D-23) satırlarda 1[y aralık dışı] ortalaması.
  Fon sırasından bağımsızdır; bu panel uyarlaması özgün ACI garantisini İDDİA ETMEZ.
- gamma=0.05 tasarım varsayılanıdır (optimize edilmiş/kanıtlanmış değer DEĞİL); alpha_init="target".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ALPHA_MIN = 0.02
ALPHA_MAX = 0.5
N_MIN = 200
CALIB_WINDOW = 126
LABEL_LAG = 23  # label_available_at = cal[t+23]
GAMMA_DEFAULT = 0.05  # tasarım varsayılanı (kanıtlanmış değer değil)


def _apply_crossing(lo: np.ndarray, hi: np.ndarray, crossing: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crossing durumunda uçları sırala (F-09: üretim ve tanı aynı yardımcıyı paylaşır)."""
    lo_f = np.where(crossing, np.minimum(lo, hi), lo)
    hi_f = np.where(crossing, np.maximum(lo, hi), hi)
    return lo_f, hi_f


def _flag(states: list[str]) -> str:
    """quality_flag: crossing bilgisi başka durumlar tarafından EZİLMEZ (PM kararı 24.09).

    Birden fazla durum varsa sözleşme adları sabit sırayla '+' ile birleşir
    (örn. "crossing+insufficient_calibration"); normal satır "ok".
    """
    order = ["crossing", "insufficient_calibration", "k_gt_n"]
    present = [s for s in order if s in states]
    return "+".join(present) if present else "ok"


def conformal_interval(scores: np.ndarray, alpha_d: float, n_min: int = N_MIN) -> tuple[float, int, bool]:
    """Sıralı istatistik: k = ceil((n+1)(1-alpha_D)); dönen: (s_k, n, k_gt_n).

    n < n_min → (nan, n, False); k > n → (nan, n, True). Sessiz fallback yok.
    """
    n = int(len(scores))
    if n < n_min:
        return float("nan"), n, False
    k = int(np.ceil((n + 1) * (1.0 - alpha_d)))
    if k > n:
        return float("nan"), n, True
    return float(np.sort(np.asarray(scores, dtype=float))[k - 1]), n, False


def _decision_position(
    features: pd.DataFrame, decision_at: pd.Timestamp, nav_cal: pd.DatetimeIndex, pos_of: pd.Series
) -> int | None:
    """Feature panel terminal satırının sonraki-session kararını takvim indeksinde `len(nav_cal)` yap."""
    d = pd.Timestamp(decision_at).normalize()
    if d in pos_of.index:
        return int(pos_of.loc[d])
    if not len(nav_cal) or d <= nav_cal[-1]:
        return None
    terminal_rows = pd.to_datetime(features["feature_asof"]).dt.normalize().eq(nav_cal[-1])
    terminal_decisions = pd.to_datetime(features.loc[terminal_rows, "decision_at"]).dropna().dt.normalize()
    if terminal_decisions.eq(d).any():
        return len(nav_cal)
    return None


def calibration_mask(features: pd.DataFrame, decision_at: pd.Timestamp, calib_window: int = CALIB_WINDOW) -> pd.Series:
    """Karar günü D için kalibrasyon satır maskesi (features çerçevesi üzerinde).

    Koşullar: y gözlenmiş (olgun), label_available_at <= D, t >= D - calib_window - 23
    (işlem-günü indeksi; NAV takvimi pozisyonları üzerinden).
    """
    cal = pd.DatetimeIndex(np.sort(pd.to_datetime(features["feature_asof"].dropna().unique())))
    pos_of = pd.Series(np.arange(len(cal)), index=cal)
    d = pd.Timestamp(decision_at)
    pos_d = _decision_position(features, d, cal, pos_of)
    if pos_d is None:
        raise ValueError(f"decision_at işlem günü takviminde değil: {d.date()}")
    t_pos = pd.to_datetime(features["feature_asof"]).map(pos_of)
    return (
        features["y"].notna()
        & (pd.to_datetime(features["label_available_at"]) <= d)
        & (t_pos >= pos_d - calib_window - LABEL_LAG)
    )


def aci_alpha_path(
    err_by_day: pd.Series,
    miscoverage_target: float,
    gamma: float = GAMMA_DEFAULT,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
) -> pd.Series:
    """Günde bir global α güncellemesi: alpha_{D+1} = clip(alpha_D + gamma*(target - err_D), ...).

    err_by_day: decision_at sıralı; err_D (NaN = o gün aralıklı olgun satır yok → α değişmez).
    Fon sırasından bağımsız; özgün ACI garantisi iddia edilmez.
    """
    alphas: dict[pd.Timestamp, float] = {}
    a = float(miscoverage_target)  # alpha_init = target (PM kararı 24.09)
    for d, err in err_by_day.sort_index().items():
        alphas[d] = a
        if pd.notna(err):
            a = float(np.clip(a + gamma * (miscoverage_target - float(err)), alpha_min, alpha_max))
    return pd.Series(alphas, name="alpha_D")


def calibrate_predictions(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    miscoverage_target: float,
    gamma: float = GAMMA_DEFAULT,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
    n_min: int = N_MIN,
    calib_window: int = CALIB_WINDOW,
) -> pd.DataFrame:
    """S3b-2 tahmin kayıtlarına CQR aralığı + günlük global alpha ekler (kayıt şeması korunur).

    Çıktı: decision_at, fund_code, model_id, q10, q50, q90, lower, upper, alpha_D,
    n_calib, quality_flag. Tahmin verilmeyen satırlar (n<200, k>n) lower/upper=NaN ile
    korunur (izlenebilirlik; TEMPORAL §6 "tahmin verilmez").
    """
    f = predictions.merge(
        features[["feature_asof", "fund_code", "decision_at", "y", "label_available_at"]],
        on=["decision_at", "fund_code"],
        how="left",
        validate="one_to_one",
    )
    nav_cal = pd.DatetimeIndex(np.sort(pd.to_datetime(features["feature_asof"].dropna().unique())))
    pos_of = pd.Series(np.arange(len(nav_cal)), index=nav_cal)
    dec_cal = pd.DatetimeIndex(np.sort(pd.to_datetime(f["decision_at"].dropna().unique())))
    yv = f["y"].to_numpy(float)
    q10 = f["q10"].to_numpy(float)
    q90 = f["q90"].to_numpy(float)
    score = np.maximum(q10 - yv, yv - q90)
    f["_score"] = np.where(f["y"].notna() & f["q10"].notna() & f["q90"].notna(), score, np.nan)
    laa = pd.to_datetime(f["label_available_at"])
    t_pos = pd.to_datetime(f["feature_asof"]).map(pos_of)

    out = f.copy()
    out["lower"] = np.nan
    out["upper"] = np.nan
    out["alpha_D"] = np.nan
    out["n_calib"] = 0
    out["quality_flag"] = "ok"
    alpha = float(miscoverage_target)  # alpha_init = target
    for d in dec_cal:
        pos_d = _decision_position(f, d, nav_cal, pos_of)
        if pos_d is None:
            rows = f["decision_at"] == d
            out.loc[rows, "quality_flag"] = "no_calendar"
            continue
        alpha_d = alpha
        m = (laa <= d) & (t_pos >= pos_d - calib_window - LABEL_LAG) & f["_score"].notna()
        s_k, n, k_gt_n = conformal_interval(f.loc[m, "_score"].to_numpy(float), alpha_d=alpha_d, n_min=n_min)
        rows = f["decision_at"] == d
        out.loc[rows, "alpha_D"] = alpha_d
        out.loc[rows, "n_calib"] = n
        rows_arr = rows.to_numpy()
        crossing = q10[rows_arr] > q90[rows_arr]
        if np.isnan(s_k):  # tahmin yok: n < n_min veya k > n — satırlar NaN aralıkla korunur
            states = ["k_gt_n"] if k_gt_n else ["insufficient_calibration"]
            out.loc[rows, "quality_flag"] = [_flag((["crossing"] if c else []) + states) for c in crossing]
        else:
            lo = q10[rows.to_numpy()] - s_k
            hi = q90[rows.to_numpy()] + s_k
            lo_f, hi_f = _apply_crossing(lo, hi, crossing)
            out.loc[rows, "lower"] = lo_f
            out.loc[rows, "upper"] = hi_f
            out.loc[rows, "quality_flag"] = np.where(crossing, "crossing", "ok")
        # err_D: o gün olgunlaşan satırlar (t = D-23 işlem günü) — aralığı olanlar üzerinden
        t_mature = pos_d - LABEL_LAG
        err = float("nan")
        if 0 <= t_mature < len(nav_cal):
            # D günü olgunlaşan satırlar: feature_asof = nav_cal[t_mature] (decision_at = D-22 işlem günü).
            # DÜZELTME (S3b-3-fix): eski `rows & ...` D günü karar satırlarıyla kesişiyordu → her zaman
            # boş → err_D NaN → alpha zinciri donardı (gerçek veri QA: alpha_unique=1).
            m_mature = (t_pos == t_mature) & out["lower"].notna() & f["y"].notna()
            if m_mature.any():
                y_m = f.loc[m_mature, "y"].to_numpy(float)
                lo_m = out.loc[m_mature, "lower"].to_numpy(float)
                hi_m = out.loc[m_mature, "upper"].to_numpy(float)
                err = float(np.mean((y_m < lo_m) | (y_m > hi_m)))
        if not np.isnan(err):
            alpha = float(np.clip(alpha + gamma * (miscoverage_target - err), alpha_min, alpha_max))
    return out.drop(columns=["_score", "label_available_at", "feature_asof"], errors="ignore")


def calibration_diagnostics(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    miscoverage_target: float,
    gamma: float = GAMMA_DEFAULT,
    alpha_min: float = ALPHA_MIN,
    alpha_max: float = ALPHA_MAX,
    n_min: int = N_MIN,
    calib_window: int = CALIB_WINDOW,
) -> dict[str, Any]:
    """Salt-okunur agregat diagnostik (S3b-3-fix): veri satırı/fon kodu/tutar raporlamaz.

    Raporlar: decision gün sayısı, err_D hesaplanan/NaN gün sayısı, olgunlaşan satır bulunan
    gün sayısı, günlük olgunlaşan satır min/medyan/max, interval'li olgunlaşmış satır sayısı,
    alpha değişen gün sayısı, ilk alpha değişim tarihi, alpha min/max.
    """
    f = predictions.merge(
        features[["feature_asof", "fund_code", "decision_at", "y", "label_available_at"]],
        on=["decision_at", "fund_code"],
        how="left",
        validate="one_to_one",
    )
    nav_cal = pd.DatetimeIndex(np.sort(pd.to_datetime(features["feature_asof"].dropna().unique())))
    pos_of = pd.Series(np.arange(len(nav_cal)), index=nav_cal)
    dec_cal = pd.DatetimeIndex(np.sort(pd.to_datetime(f["decision_at"].dropna().unique())))
    yv = f["y"].to_numpy(float)
    score = np.maximum(f["q10"].to_numpy(float) - yv, yv - f["q90"].to_numpy(float))
    f["_score"] = np.where(f["y"].notna() & f["q10"].notna() & f["q90"].notna(), score, np.nan)
    laa = pd.to_datetime(f["label_available_at"])
    t_pos = pd.to_datetime(f["feature_asof"]).map(pos_of)
    lower = np.full(len(f), np.nan)
    upper = np.full(len(f), np.nan)

    n_err_days = n_err_nan_days = n_mature_days = n_alpha_changes = 0
    mature_counts: list[int] = []
    n_mature_with_interval = 0
    alpha = float(miscoverage_target)
    alpha_first_change: pd.Timestamp | None = None
    prev_alpha = alpha
    for d in dec_cal:
        pos_d = _decision_position(f, d, nav_cal, pos_of)
        if pos_d is None:
            continue
        m = (laa <= d) & (t_pos >= pos_d - calib_window - LABEL_LAG) & f["_score"].notna()
        s_k, _, _ = conformal_interval(f.loc[m, "_score"].to_numpy(float), alpha_d=alpha, n_min=n_min)
        if not np.isnan(s_k):
            rows = (f["decision_at"] == d).to_numpy()
            lo = f.loc[rows, "q10"].to_numpy(float) - s_k
            hi = f.loc[rows, "q90"].to_numpy(float) + s_k
            crossing = f.loc[rows, "q10"].to_numpy(float) > f.loc[rows, "q90"].to_numpy(float)
            lower[rows], upper[rows] = _apply_crossing(lo, hi, crossing)
        t_mature = pos_d - LABEL_LAG
        n_mature = 0
        err = float("nan")
        if 0 <= t_mature < len(nav_cal):
            m_mature = (t_pos == t_mature) & pd.Series(lower, index=f.index).notna() & f["y"].notna()
            n_mature = int(m_mature.sum())
            n_mature_with_interval += n_mature
            if n_mature:
                n_mature_days += 1
                mature_counts.append(n_mature)
                y_m = f.loc[m_mature, "y"].to_numpy(float)
                lo_m = lower[m_mature.to_numpy()]
                hi_m = upper[m_mature.to_numpy()]
                err = float(np.mean((y_m < lo_m) | (y_m > hi_m)))
        if not np.isnan(err):
            n_err_days += 1
            alpha = float(np.clip(alpha + gamma * (miscoverage_target - err), alpha_min, alpha_max))
        else:
            n_err_nan_days += 1
        if alpha != prev_alpha:
            n_alpha_changes += 1
            alpha_first_change = alpha_first_change or d
            prev_alpha = alpha
    return {
        "n_decision_days": int(len(dec_cal)),
        "n_err_days": n_err_days,
        "n_err_nan_days": n_err_nan_days,
        "n_mature_days": n_mature_days,
        "mature_rows_min": int(min(mature_counts)) if mature_counts else 0,
        "mature_rows_median": float(np.median(mature_counts)) if mature_counts else 0.0,
        "mature_rows_max": int(max(mature_counts)) if mature_counts else 0,
        "n_mature_with_interval": n_mature_with_interval,
        "n_alpha_changes": n_alpha_changes,
        "alpha_first_change": str(alpha_first_change.date()) if alpha_first_change is not None else None,
        "alpha_min": float(min(prev_alpha, miscoverage_target)),
        "alpha_max": float(max(prev_alpha, miscoverage_target)),
    }


def coverage_report(calibrated: pd.DataFrame, selected_mask: pd.Series | None = None) -> pd.DataFrame:
    """Ay bazında kapsama: aralığı olan satırlarda 1[lower <= y <= upper] ortalaması.

    selected_mask: S3b-4 seçim maskesi (calibrated indeksiyle hizalı bool) — S3b-3'te yoksa
    generic OOS kapsaması raporlanır; seçili-fon kapsaması bu maskeyi kabul eder (S3b-4).
    """
    df = calibrated[calibrated["lower"].notna() & calibrated["y"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["month", "coverage", "n_rows"])
    if selected_mask is not None:
        df = df[selected_mask.reindex(df.index).fillna(False).to_numpy()]
    y = df["y"].to_numpy(float)
    inside = ((df["lower"].to_numpy(float) <= y) & (y <= df["upper"].to_numpy(float))).astype(float)
    month = pd.to_datetime(df["decision_at"]).dt.to_period("M")
    g = pd.DataFrame({"month": month, "cov": inside}).groupby("month")["cov"].agg(["mean", "size"])
    g.columns = ["coverage", "n_rows"]
    return g.reset_index()
