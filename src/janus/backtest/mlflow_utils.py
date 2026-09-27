"""Opsiyonel MLflow kaydı (yerel SQLite: mlruns/mlflow.db). mlflow yoksa ya da hata verirse sessizce atlanır."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from loguru import logger


def _git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:  # noqa: BLE001
        return "nogit"


def tracking_uri(root: Path) -> str:
    (root / "mlruns").mkdir(exist_ok=True)
    return f"sqlite:///{root / 'mlruns' / 'mlflow.db'}"


def log_rows(rows: list[dict], cfg: dict, root: Path, experiment: str = "janus-backtest", data_asof: str = "") -> bool:
    try:
        import mlflow  # noqa: PLC0415

        mlflow.set_tracking_uri(tracking_uri(root))
        mlflow.set_experiment(experiment)
        cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
        sha = _git_sha(root)
        for r in rows:
            with mlflow.start_run(run_name=str(r.get("strategy", "run"))):
                mlflow.set_tags({"git_sha": sha, "config_hash": cfg_hash, "data_asof": data_asof})
                for k, v in r.items():
                    if isinstance(v, int | float) and v == v:
                        mlflow.log_metric(k, float(v))
                    elif isinstance(v, str) and k != "strategy":
                        mlflow.log_param(k, v[:200])
        logger.info("MLflow: {} koşu kaydedildi ({})", len(rows), tracking_uri(root))
        return True
    except Exception as e:  # noqa: BLE001 — kayıt hiçbir zaman raporu engellemez
        logger.warning("MLflow kaydı atlandı: {}", type(e).__name__)
        return False
