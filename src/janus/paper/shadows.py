"""S5-3: gölge portföyler (B0, B2b, B2c-m, B2c-fdr q20/q10/q30)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from janus.backtest.data import prepare
from janus.backtest.engine import BacktestConfig, run_backtest
from janus.models.conformal_selection import FdrHRP, conformal_pvalues
from janus.strategies.baselines import CashOnly
from janus.strategies.conformal_select import ConformalHRP
from janus.strategies.portfolio import MomentumHRP

SHADOWS = {
    "B0_cash": lambda _p, _c, _f: CashOnly(),
    "B2b_momentum_hrp": lambda p, c, f: _make_b2b(p, c, f),
    "B2c_m_conformal_hrp": lambda p, c, f: _make_b2c_m(p, c, f),
    "B2c_fdr_q20": lambda p, c, f: _make_fdr(p, c, f, 0.20),
    "B2c_fdr_q10": lambda p, c, f: _make_fdr(p, c, f, 0.10),
    "B2c_fdr_q30": lambda p, c, f: _make_fdr(p, c, f, 0.30),
}


def _make_b2b(prepared, cfg, features):
    con = cfg["legs"]["tefas"]["constraints"]
    hrp = cfg.get("hrp", {})
    founders = prepared["fund_master"].drop_duplicates("fund_code").set_index("fund_code")["founder"]
    return MomentumHRP(
        n=10,
        eligible=prepared["eligible"],
        founders=founders,
        max_fund=con["max_weight_per_fund"],
        max_founder=con["max_weight_per_founder"],
        max_per_founder=con["max_funds_per_founder"],
        max_per_cluster=con["max_funds_per_hrp_cluster"],
        cov_days=hrp.get("cov_days", 126),
        cluster_distance=hrp.get("cluster_distance", 0.4),
        linkage=hrp.get("linkage", "single"),
        buffer_mult=hrp.get("buffer_mult", 2.0),
        tax_aware=hrp.get("tax_aware", True),
        refresh_every=hrp.get("refresh_every", 3),
        smooth_lambda=hrp.get("smooth_lambda", 0.5),
        name="B2b_momentum_hrp",
    )


def _make_b2c_m(prepared, cfg, features):
    con = cfg["legs"]["tefas"]["constraints"]
    hrp = cfg.get("hrp", {})
    founders = prepared["fund_master"].drop_duplicates("fund_code").set_index("fund_code")["founder"]
    preds_path = Path("data") / "predictions" / "predictions.parquet"
    cal_path = Path("data") / "predictions" / "calibrated_target_020.parquet"
    if not preds_path.exists() or not cal_path.exists():
        raise FileNotFoundError("B2c-m için predictions/calibrated parquet gerekli")
    preds = pd.read_parquet(preds_path)
    cal = pd.read_parquet(cal_path)
    return ConformalHRP(
        calibrated=cal,
        predictions=preds,
        features=features,
        kill_switch=True,
        ks_window=126,
        eligible=prepared["eligible"],
        founders=founders,
        max_fund=con["max_weight_per_fund"],
        max_founder=con["max_weight_per_founder"],
        max_per_founder=con["max_funds_per_founder"],
        max_per_cluster=con["max_funds_per_hrp_cluster"],
        cov_days=hrp.get("cov_days", 126),
        cluster_distance=hrp.get("cluster_distance", 0.4),
        linkage=hrp.get("linkage", "single"),
        name="B2c_m_conformal_hrp",
    )


def _make_fdr(prepared, cfg, features, q: float):
    con = cfg["legs"]["tefas"]["constraints"]
    hrp = cfg.get("hrp", {})
    founders = prepared["fund_master"].drop_duplicates("fund_code").set_index("fund_code")["founder"]
    preds_path = Path("data") / "predictions" / "predictions.parquet"
    if not preds_path.exists():
        raise FileNotFoundError("B2c-fdr için predictions.parquet gerekli")
    preds = pd.read_parquet(preds_path)
    pvalues = conformal_pvalues(preds, features)
    return FdrHRP(
        calibrated=None,
        predictions=preds,
        features=features,
        fdr_q=q,
        pvalues=pvalues,
        eligible=prepared["eligible"],
        founders=founders,
        max_fund=con["max_weight_per_fund"],
        max_founder=con["max_weight_per_founder"],
        max_per_founder=con["max_funds_per_founder"],
        max_per_cluster=con["max_funds_per_hrp_cluster"],
        cov_days=hrp.get("cov_days", 126),
        cluster_distance=hrp.get("cluster_distance", 0.4),
        linkage=hrp.get("linkage", "single"),
        name=f"B2c_fdr_q{int(round(q * 100)):02d}",
    )


def run_shadows(store, cfg: dict, names: list[str] | None = None) -> dict[str, dict]:
    """Portföy başına durum ve equity döndür; bir stratejinin hatası diğerlerini durdurmaz."""
    selected = list(SHADOWS.items()) if names is None else [(name, SHADOWS[name]) for name in names]
    try:
        prepared = prepare(store, cfg)
        features_path = Path("data") / "features" / "fund_features.parquet"
        features = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
        bcfg = BacktestConfig.from_cfg(cfg)
    except Exception as exc:  # noqa: BLE001
        return {
            name: {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "equity": pd.Series(dtype=float)}
            for name, _ in selected
        }

    results: dict[str, dict] = {}
    for name, factory in selected:
        try:
            strat = factory(prepared, cfg, features)
            res = run_backtest(
                prepared["nav"],
                prepared["meta"],
                strat,
                bcfg,
                cash_returns=prepared["cash_returns"],
                cash_nav=prepared["cash_nav"],
                cash_category=prepared["cash_category"],
                cash_rate=prepared["cash_rate"],
                execution_meta_by_date=prepared.get("execution_meta_by_date"),
            )
            equity = res.equity
            if not isinstance(equity, pd.Series) or equity.empty or not equity.notna().any():
                raise ValueError("equity serisi boş")
            results[name] = {"status": "ok", "error": None, "equity": equity}
        except Exception as exc:  # noqa: BLE001 — diğer gölge portföyler devam etsin
            results[name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "equity": pd.Series(dtype=float),
            }
    return results


def snapshot_shadow_equity(store, results: dict[str, dict], nav_asof: pd.Timestamp) -> dict:
    """Durum tablosuna her portföyü yazar; bileşenleri hesaplanmadıysa NULL bırakır."""
    from janus.paper.core import _ensure_paper_tables

    _ensure_paper_tables(store)
    run_date = pd.Timestamp(nav_asof).normalize()
    n_rows = 0
    n_failed = 0
    for name, result in results.items():
        status = result.get("status", "failed")
        equity = result.get("equity")
        error = result.get("error")
        if status != "ok" or not isinstance(equity, pd.Series) or equity.empty:
            status = "failed"
            error = error or "equity serisi yok"
            n_failed += 1
            n_equity_rows = 0
        else:
            valid = equity.dropna()
            if valid.empty or not np.isfinite(valid.to_numpy(dtype=float)).all():
                status = "failed"
                error = "equity serisi boş veya sonlu değil"
                n_failed += 1
                n_equity_rows = 0
            else:
                equity = valid
                n_equity_rows = len(equity)

        store.con.execute(
            "DELETE FROM paper_shadow_runs WHERE portfolio_name = ? AND date = ?", [name, run_date.date()]
        )
        store.con.execute(
            "INSERT INTO paper_shadow_runs VALUES (?, ?, ?, ?, ?, ?)",
            [name, run_date.date(), status, error, n_equity_rows, run_date.date()],
        )
        if status != "ok":
            store.con.execute(
                "DELETE FROM paper_equity WHERE portfolio_name = ? AND nav_asof = ?",
                [name, run_date.date()],
            )
            continue
        last_date = pd.Timestamp(equity.index[-1]).normalize()
        value = float(equity.iloc[-1])
        store.con.execute(
            "DELETE FROM paper_equity WHERE portfolio_name = ? AND date = ?",
            [name, last_date.date()],
        )
        store.con.execute(
            """
            INSERT INTO paper_equity (portfolio_name, date, equity, cash, receivables, risky_value, slot_value, nav_asof)
            VALUES (?, ?, ?, NULL, NULL, NULL, NULL, ?)
            """,
            [name, last_date.date(), value, run_date.date()],
        )
        n_rows += 1
    return {
        "status": "failed" if n_failed == len(results) and n_failed else "partial" if n_failed else "ok",
        "n_rows": n_rows,
        "failed": n_failed,
    }
