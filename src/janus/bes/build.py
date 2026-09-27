"""BES boru hattı: EMK evreni için feature / prediction / calibration (S5-5).

Çıktılar:
- data/features/bes_features.parquet
- data/predictions/bes_predictions.parquet
- data/predictions/bes_calibrated_target_020.parquet

BES'e özgü kurallar:
- fund_class = EMK
- devlet katkısı fonları dışarı (config legs.bes.universe.exclude_state_contribution_patterns)
- stopaj yok (tax_rate = 0)
- nakit vekili EMK para piyasası fonları sepeti; yoksa hata (fallback yok)
- valör A8 varsayımı (config legs.bes.execution)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def bes_build(
    st,
    cfg: dict[str, Any],
    root: Path,
    model_kwargs: dict[str, Any] | None = None,
    incremental: bool = False,
) -> dict[str, Any]:
    """BES boru hattını çalıştır: features → predictions → calibration (target 0.20).

    `st`: Store (okuma)
    `cfg`: yüklenmiş janus.yaml
    `root`: proje kök dizini
    `model_kwargs`: LightGBM hiperparametreleri; None → cpcv_summary.json'dan
    `incremental`: mevcut BES parquet'lerinden devam et
    """
    # döngüsel import önlemi: CLI'nin helper'ları burada içe aktarılır
    from janus.cli import (
        _features_build_impl,
        _model_kwargs_from_cpcv,
        _predictions_build_impl,
        _predictions_calibrate_impl,
    )

    nav = st.nav_wide()
    fm = st.latest_fund_master()
    if nav.empty or fm.empty:
        raise ValueError("BES build için veri yok — önce `janus ingest tefas`")

    feat_path = root / "data" / "features" / "bes_features.parquet"
    pred_path = root / "data" / "predictions" / "bes_predictions.parquet"
    cal_path = root / "data" / "predictions" / "bes_calibrated_target_020.parquet"

    mk = model_kwargs if model_kwargs is not None else _model_kwargs_from_cpcv()

    feat_summary = _features_build_impl(nav, fm, cfg, st, feat_path, incremental=incremental, leg="bes")
    bes_features = pd.read_parquet(feat_path)
    # ADR-0029/4: kapsam pruning'i guard öncesinden kaldırıldı; impl'ler canonical'ı kendileri korur.
    pred_summary = _predictions_build_impl(bes_features, pred_path, model_kwargs=mk, incremental=incremental, cfg=cfg)
    bes_predictions = pd.read_parquet(pred_path)
    cal_summary = _predictions_calibrate_impl(
        bes_predictions,
        bes_features,
        0.20,
        cal_path,
        cfg.get("conformal", {}),
        incremental=incremental,
    )
    return {
        "status": "ok",
        "features": feat_summary,
        "predictions": pred_summary,
        "calibrate": cal_summary,
    }
