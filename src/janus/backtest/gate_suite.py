"""S3a koşucusu: B2b iskeleti üstünde dört kapı (yok / R0 kural / R1 jump / R1 HMM / R2 makas), 12 başlangıç; B2b taraması + PBO/DSR."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from janus.backtest.data import prepare
from janus.backtest.engine import BacktestConfig, run_backtest, stale_flags
from janus.backtest.report import result_row, save_report
from janus.backtest.robustness import parameter_sweep, start_date_sweep
from janus.data.store import Store
from janus.features.macro import macro_features
from janus.features.market import market_features, regime_feature_matrix
from janus.strategies.baselines import RuleGate
from janus.strategies.gates import RegimeGate, SpreadGate
from janus.strategies.portfolio import MomentumHRP


def build_regime_features(store: Store, d: dict, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    nav = d["nav"]
    fm = d["fund_master"].drop_duplicates("fund_code").set_index("fund_code")
    eq_cols = [c for c in nav.columns if str(fm["umbrella_type"].get(c, "")).lower().startswith("hisse")]
    stale = stale_flags(nav, int(cfg["legs"]["tefas"]["universe"].get("max_stale_days", 2)))
    mk = market_features(nav, d["equity_index"], eq_cols, stale)
    mc = macro_features(store, nav.index)
    cols = cfg.get("regime", {}).get("features")
    X = regime_feature_matrix(mk, mc, cols)
    cash_idx = (1 + d["cash_returns"].fillna(0.0)).cumprod()
    excess = np.log(d["equity_index"]).diff() - np.log(cash_idx).diff()
    return X, excess.fillna(0.0)


def run_gate_suite(
    store: Store, cfg: dict, top_n: int = 10, out_dir: Path | None = None, quick: bool = False
) -> tuple[str, Path | None]:
    d = prepare(store, cfg)
    bcfg = BacktestConfig.from_cfg(cfg)
    lv = bcfg.exposure_levels
    con = cfg["legs"]["tefas"]["constraints"]
    hrp = cfg.get("hrp", {})
    rg = cfg.get("regime", {})
    founders = d["fund_master"].drop_duplicates("fund_code").set_index("fund_code")["founder"]
    X, excess = build_regime_features(store, d, cfg)
    cash_index = (1 + d["cash_returns"].fillna(0.0)).cumprod()

    def core(**k):
        params = dict(
            n=top_n,
            eligible=d["eligible"],
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
        )
        params.update(k)
        return MomentumHRP(**params)

    gates = {
        "B2b_no_gate": lambda: core(name="B2b_no_gate"),
        "B3_R0_rule": lambda: RuleGate(index_nav=d["equity_index"], inner=core(), levels=lv, name="B3_R0_rule"),
        "B3_R1_jump": lambda: RegimeGate(
            inner=core(),
            features=X,
            excess_returns=excess,
            levels=lv,
            model="jump",
            k=rg.get("k", 3),
            jump_penalty=rg.get("jump_penalty", 50.0),
            refit_every=rg.get("refit_every", 1),
            name="B3_R1_jump",
        ),
        "B3_R2_spread": lambda: SpreadGate(
            inner=core(),
            equity_index=d["equity_index"],
            cash_index=cash_index,
            levels=lv,
            window=rg.get("spread_window", 63),
            margin=rg.get("spread_margin", 0.02),
            name="B3_R2_spread",
        ),
    }
    if rg.get("hmm", True) and not quick:
        gates["B3_R1_hmm"] = lambda: RegimeGate(
            inner=core(), features=X, excess_returns=excess, levels=lv, model="hmm", k=rg.get("k", 3), name="B3_R1_hmm"
        )

    rows, dist_rows = [], []
    n_starts = 4 if quick else int(cfg.get("validation", {}).get("start_dates", 12))
    for name, factory in gates.items():
        try:
            res = run_backtest(
                d["nav"],
                d["meta"],
                factory(),
                bcfg,
                cash_returns=d["cash_returns"],
                cash_nav=d["cash_nav"],
                cash_category=d["cash_category"],
                cash_rate=d["cash_rate"],
                execution_meta_by_date=d.get("execution_meta_by_date"),
            )
            rows.append(result_row(res))
            sd = start_date_sweep(
                d["nav"],
                d["meta"],
                factory,
                bcfg,
                d["cash_returns"],
                n_starts=n_starts,
                cash_nav=d["cash_nav"],
                cash_category=d["cash_category"],
                cash_rate=d["cash_rate"],
                execution_meta_by_date=d.get("execution_meta_by_date"),
            )
            q = sd[["excess_cagr", "mdd", "cagr"]].quantile([0.0, 0.5, 1.0])
            dist_rows.append(
                {
                    "gate": name,
                    "excess_min": q.loc[0.0, "excess_cagr"],
                    "excess_med": q.loc[0.5, "excess_cagr"],
                    "excess_max": q.loc[1.0, "excess_cagr"],
                    "mdd_med": q.loc[0.5, "mdd"],
                    "mdd_max": q.loc[1.0, "mdd"],
                    "cagr_med": q.loc[0.5, "cagr"],
                    "n_starts": len(sd),
                }
            )
        except Exception as e:  # noqa: BLE001 — bir kapı düşerse diğerleri raporlansın
            logger.warning("{} atlandı: {}", name, type(e).__name__)

    md = [
        f"# S3a — kapı karşılaştırması (B2b iskeleti) — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Tüm metrikler vergi ve maliyet sonrası; tutar yok.",
    ]
    if d.get("snapshot_asof") is not None:
        md += ["", f"veri as-of: {d['snapshot_asof']:%Y-%m-%d %H:%M} (parquet anlık görüntüsü)"]
    md += ["", "## Tam dönem", ""]
    tbl = pd.DataFrame(rows).set_index("strategy")[
        ["cagr", "excess_cagr", "sharpe", "mdd", "turnover_per_year", "taxes_pct_of_final", "dd_triggers"]
    ]
    md.append(
        (tbl.assign(**{c: tbl[c] * 100 for c in ["cagr", "excess_cagr", "mdd", "taxes_pct_of_final"]}))
        .round(2)
        .to_markdown()
    )
    md += ["", f"## {n_starts} başlangıç tarihi (yüzde puan)", ""]
    dd = pd.DataFrame(dist_rows).set_index("gate")
    md.append(
        (
            dd.assign(
                **{c: dd[c] * 100 for c in ["excess_min", "excess_med", "excess_max", "mdd_med", "mdd_max", "cagr_med"]}
            )
        )
        .round(1)
        .to_markdown()
    )
    md += ["", "Kabul kriteri: excess_med ≥ 0 ve mdd_max ≤ B3_R0_rule'ınki.", ""]

    # B2b taraması
    grid = dict(cfg.get("sweep_hrp", {"n": [5, 10, 15], "smooth_lambda": [0.3, 0.5, 0.7], "refresh_every": [3, 6]}))
    if quick:
        grid = {k: v[:2] for k, v in grid.items()}
    sweep_df, info = parameter_sweep(
        d["nav"],
        d["meta"],
        lambda **p: core(**p),
        grid,
        bcfg,
        d["cash_returns"],
        cash_nav=d["cash_nav"],
        cash_category=d["cash_category"],
        cash_rate=d["cash_rate"],
        execution_meta_by_date=d.get("execution_meta_by_date"),
    )
    md += [
        "## B2b taraması — PBO ve deflated Sharpe",
        "",
        sweep_df.round(3).to_markdown(index=False),
        "",
        f"- PBO (CSCV, {info['n_combos']} kombinasyon): **{info['pbo']:.2f}**",
        f"- En iyi konfig deflated Sharpe olasılığı: **{info['deflated_sharpe_prob']:.2f}**",
        f"- En iyi: {info['best']}",
        "",
        "## Rejim özellikleri",
        "",
        f"Kolonlar: {', '.join(X.columns)}; satır: {X.dropna(how='all').shape[0]} gün",
        "",
    ]
    text = "\n".join(md)
    path = save_report(text, out_dir, f"gate_suite_{datetime.now():%Y%m%d_%H%M}") if out_dir else None
    return text, path
