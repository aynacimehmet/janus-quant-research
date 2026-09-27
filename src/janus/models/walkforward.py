"""Walk-forward OOS tahmin (S3b-2): aylık refit R = ayın ilk işlem günü; TEMPORAL §5.

Maske birleşimi (PM onayı 24.09, S3b-2 planı madde 5):
- eğitim: feature_ready & eligible_at_decision & y.notna() & (label_available_at <= R)
- tahmin/OOS: feature_ready & eligible_at_decision
- label_ready WF filtresi DEĞİLDİR; policy_today_excluded S3b-2'de kullanılmaz.
Her kayıt: [decision_at, fund_code, q10, q50, q90, model_id, train_max_t];
train_max_t <= decision_at - 23 (işlem günü) zorunlu.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from janus.features.fund_features import FEATURE_COLUMNS, validate_columns
from janus.models.quantile_gbdt import QuantileGBDT

PURGE_DAYS = 21  # VALIDATION walk_forward.purge_days
EMBARGO_DAYS = 5


def refit_dates(decision_calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ayın ilk işlem günleri (işlem günü takviminden; TEMPORAL §5)."""
    cal = pd.DatetimeIndex(decision_calendar).sort_values()
    months = cal.to_period("M")
    first = np.r_[True, months[1:] != months[:-1]]
    return cal[first]


def train_mask(features: pd.DataFrame, r: pd.Timestamp) -> pd.Series:
    """Eğitim maskesi: label_available_at <= R (purge tanımı; TEMPORAL §4/§5)."""
    return (
        features["feature_ready"].fillna(False).astype(bool)
        & features["eligible_at_decision"].fillna(False).astype(bool)
        & features["y"].notna()
        & (pd.to_datetime(features["label_available_at"]) <= r)
    )


def oos_mask(features: pd.DataFrame) -> pd.Series:
    """Tahmin/OOS maskesi: feature_ready & eligible_at_decision (label_ready filtre değildir)."""
    return features["feature_ready"].fillna(False).astype(bool) & features["eligible_at_decision"].fillna(False).astype(
        bool
    )


def run_walkforward(features: pd.DataFrame, model_kwargs: dict[str, Any] | None = None) -> pd.DataFrame:
    """Aylık refit; R ile sonraki refit arasındaki karar günlerinde aynı M_R kullanılır.

    Çıktı: [decision_at, fund_code, q10, q50, q90, model_id, train_max_t].
    """
    mk = dict(model_kwargs or {})
    f = features
    allowed = {
        *FEATURE_COLUMNS,
        "feature_asof",
        "fund_code",
        "decision_at",
        "label_end",
        "label_available_at",
        "feature_ready",
        "label_ready",
        "eligible_at_decision",
        "policy_today_excluded",
        "y",
    }
    validate_columns([c for c in f.columns if c not in allowed])  # beyaz liste dışı kolon → ValueError
    dec_cal = pd.DatetimeIndex(np.sort(pd.to_datetime(f["decision_at"]).dropna().unique()))
    r_dates = refit_dates(dec_cal)
    base_tr = (
        f["feature_ready"].fillna(False).astype(bool)
        & f["eligible_at_decision"].fillna(False).astype(bool)
        & f["y"].notna()
    )
    base_pr = oos_mask(f)
    laa = pd.to_datetime(f["label_available_at"])
    out: list[pd.DataFrame] = []
    for i, r in enumerate(r_dates):
        nxt = r_dates[i + 1] if i + 1 < len(r_dates) else pd.Timestamp.max
        tr = base_tr & (laa <= r)
        if not tr.any():
            continue
        pr = base_pr & (f["decision_at"] >= r) & (f["decision_at"] < nxt)
        if not pr.any():
            continue
        model = QuantileGBDT(**mk).fit(f.loc[tr, FEATURE_COLUMNS], f.loc[tr, "y"])
        train_max_t = pd.Timestamp(f.loc[tr, "feature_asof"].max())
        pos = dec_cal.get_loc(r)
        if pos >= 23 and train_max_t > dec_cal[pos - 23]:
            raise AssertionError(f"sızıntı: train_max_t {train_max_t.date()} > decision_at-23 (R={r.date()})")
        preds = model.predict(f.loc[pr, FEATURE_COLUMNS])
        rec = pd.DataFrame(
            {
                "decision_at": f.loc[pr, "decision_at"].to_numpy(),
                "fund_code": f.loc[pr, "fund_code"].to_numpy(),
                **{c: preds[c].to_numpy() for c in preds.columns},
                "model_id": r,
                "train_max_t": train_max_t,
            }
        )
        out.append(rec)
    if not out:
        return pd.DataFrame(columns=["decision_at", "fund_code", "q10", "q50", "q90", "model_id", "train_max_t"])
    return (
        pd.concat(out, ignore_index=True)
        .sort_values(["decision_at", "fund_code"], kind="stable")
        .reset_index(drop=True)
    )


def cpcv_select(
    features: pd.DataFrame,
    grid: list[dict[str, Any]],
    n_blocks: int = 6,
    n_test: int = 2,
    purge: int = PURGE_DAYS,
    embargo: int = EMBARGO_DAYS,
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
    model_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """CPCV (6 blok / 2 test) — yalnızca hiperparametre seçimi (VALIDATION §1); strateji kanıtı değildir.

    Purge/embargo: eğitim satırı dışlanır, etiket penceresi [t+1, t+22] test bloğuyla kesişiyorsa
    VEYA t test bloğu sonundan embargo kadar içindeyse. Skor: test satırlarında α'lar ortalaması pinball.
    """
    from janus.models.metrics import pinball_loss

    f = features
    cal = pd.DatetimeIndex(np.sort(pd.to_datetime(f["feature_asof"]).dropna().unique()))
    pos_of = {d: i for i, d in enumerate(cal)}
    edges = np.linspace(0, len(cal), n_blocks + 1).astype(int)
    blocks = [(edges[b], edges[b + 1]) for b in range(n_blocks)]
    base = oos_mask(f) & f["y"].notna()
    t_pos = f["feature_asof"].map(pos_of)
    label_end_pos = t_pos + 22  # label_end = cal[t+22]
    rows = []
    for combo in combinations(range(n_blocks), n_test):
        test_lo = blocks[combo[0]][0]
        test_hi = blocks[max(combo)][1]
        test_pos = np.zeros(len(cal), dtype=bool)
        for b in combo:
            test_pos[blocks[b][0] : blocks[b][1]] = True
        in_test = pd.Series(test_pos[t_pos.to_numpy()], index=f.index)
        # purge/embargo: eğitim = test dışı VE (etiket penceresi testten önce biter VEYA embargo sonrası)
        tr = base & ~in_test & ((label_end_pos < test_lo) | (t_pos > test_hi + embargo))
        te = base & in_test
        if not tr.any() or not te.any():
            continue
        Xtr, ytr = f.loc[tr, FEATURE_COLUMNS], f.loc[tr, "y"]
        Xte, yte = f.loc[te, FEATURE_COLUMNS], f.loc[te, "y"]
        for g in grid:
            mk = dict(model_kwargs or {})
            mk.update(g)
            model = QuantileGBDT(alphas=alphas, **mk).fit(Xtr, ytr)
            preds = model.predict(Xte)
            losses = [
                pinball_loss(yte.to_numpy(float), preds[f"q{int(round(a * 100))}"].to_numpy(float), a) for a in alphas
            ]
            rows.append({**g, "combo": combo, "pinball_mean": float(np.mean(losses)), "n_test_rows": int(te.sum())})
    if not rows:
        return pd.DataFrame(columns=["pinball_mean"])
    return pd.DataFrame(rows)
