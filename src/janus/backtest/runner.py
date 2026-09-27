"""S2 koşucuları: baseline zinciri (B0/B1 varyantları/B2/B3) ve tam S2b paketi (teşhis + sağlamlık + tarama + MLflow)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd

from janus.backtest.data import prepare
from janus.backtest.diagnostics import friction_decomposition, holdings_by_type, yearly_table
from janus.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from janus.backtest.report import result_row, save_report, to_markdown
from janus.backtest.robustness import (
    parameter_sweep,
    start_date_sweep,
    stressed_execution_meta,
    stressed_meta,
    suspension_scenario,
)
from janus.data.store import Store
from janus.strategies.baselines import CashOnly, RuleGate, TopNMomentum
from janus.strategies.portfolio import MomentumHRP


def _pct_table(df: pd.DataFrame) -> str:
    return (df * 100).round(1).astype(str).radd("%").to_markdown()


def build_chain(d: dict, cfg: dict, top_n: int) -> list[tuple[object, BacktestConfig]]:
    """Strateji, konfig çiftleri. Sıra = zincir."""
    bcfg = BacktestConfig.from_cfg(cfg)
    lv = bcfg.exposure_levels
    con = cfg["legs"]["tefas"]["constraints"]
    hrp = cfg.get("hrp", {})
    founders = d["fund_master"].drop_duplicates("fund_code").set_index("fund_code")["founder"]

    def mk_b1(**k):
        return TopNMomentum(n=top_n, eligible=d["eligible"], max_weight=con["max_weight_per_fund"], **k)

    def mk_b2(**k):
        return MomentumHRP(
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
            **k,
        )

    b2 = mk_b2()
    b2b = mk_b2(
        name="B2b_hrp_q_smooth", refresh_every=hrp.get("refresh_every", 3), smooth_lambda=hrp.get("smooth_lambda", 0.5)
    )
    b3 = RuleGate(
        index_nav=d["equity_index"],
        levels=lv,
        name="B3_hrp_gate",
        inner=mk_b2(refresh_every=hrp.get("refresh_every", 3), smooth_lambda=hrp.get("smooth_lambda", 0.5)),
    )
    return [
        (CashOnly(), bcfg),
        (mk_b1(name="B1_topN_momentum"), bcfg),
        (mk_b1(name="B1b_buffer2x", buffer_mult=2.0), bcfg),
        (mk_b1(name="B1c_quarterly", buffer_mult=2.0), replace(bcfg, schedule="quarterly_first_business_day")),
        (mk_b1(name="B1d_tax_aware", buffer_mult=2.0, tax_aware=True), bcfg),
        (b2, bcfg),
        (b2b, bcfg),
        (b3, bcfg),
    ]


def run_chain(d: dict, cfg: dict, top_n: int, start=None, end=None) -> list[BacktestResult]:
    out = []
    for strat, bcfg in build_chain(d, cfg, top_n):
        out.append(
            run_backtest(
                d["nav"],
                d["meta"],
                strat,
                bcfg,
                cash_returns=d["cash_returns"],
                start=start,
                end=end,
                cash_nav=d["cash_nav"],
                cash_category=d["cash_category"],
                cash_rate=d["cash_rate"],
                execution_meta_by_date=d.get("execution_meta_by_date"),
            )
        )
    return out


def run_baselines(
    store: Store, cfg: dict, start=None, end=None, top_n: int = 10, out_dir: Path | None = None
) -> tuple[list[dict], Path | None]:
    d = prepare(store, cfg, start)
    results = run_chain(d, cfg, top_n, start, end)
    rows = [result_row(r) for r in results]
    path = None
    if out_dir is not None:
        notes = [
            f"Evren: Faz-1 uygun {int(d['eligible'].sum())} fon; NAV paneli {d['nav'].shape[0]} gün × {d['nav'].shape[1]} fon",
            "Nakit vekili: en büyük 3 para piyasası fonu (stopaj sonrası)",
            "Sapma eşiği %3 / vergi doğuran satışta %6; DD > %12 → orta maruziyet",
            "Yürütme profili karar gününe göre PIT'tir; 2026-09-22 öncesi sonuçlar A3 varsayımlı, PIT kanıtı değildir.",
            "Bugünkü PYŞ kara listesi tarihsel icra profilinden ayrıdır.",
        ]
        proxy_rows = (
            int(d["execution_meta_by_date"]["last_success_source"].eq("proxy_ingest_time").sum())
            if "last_success_source" in d["execution_meta_by_date"]
            else 0
        )
        if proxy_rows:
            notes.append(
                f"Profil zamanında {proxy_rows} karar-fon gözlemi proxy_ingest_time kullandı: vekil zaman, PIT kanıtı değil."
            )
        flag_mismatches = int(d["execution_meta_by_date"]["status_flag_mismatch"].fillna(False).sum())
        if flag_mismatches:
            notes.append(
                f"Durum metni ile kayıtlı can_buy/can_sell bayrakları {flag_mismatches} gözlemde uyuşmadı; "
                "durum metni tek işlem kaynağıdır."
            )
        if d.get("snapshot_asof") is not None:
            notes.append(f"veri as-of: {d['snapshot_asof']:%Y-%m-%d %H:%M} (parquet anlık görüntüsü)")
        md = to_markdown(
            rows,
            f"Baseline zinciri — {datetime.now():%Y-%m-%d %H:%M}",
            notes=notes,
        )
        md += "\n\n## Yıllık getiriler\n\n" + _pct_table(yearly_table(results))
        path = save_report(md, out_dir, f"backtest_baselines_{datetime.now():%Y%m%d_%H%M}")
    return rows, path


def run_suite(
    store: Store, cfg: dict, top_n: int = 10, out_dir: Path | None = None, root: Path | None = None, quick: bool = False
) -> tuple[str, Path | None]:
    """S2b tam paket. quick=True: tarama ve askı senaryosu küçültülür."""
    d = prepare(store, cfg)
    bcfg = BacktestConfig.from_cfg(cfg)
    chain = build_chain(d, cfg, top_n)
    results = [
        run_backtest(
            d["nav"],
            d["meta"],
            s,
            c,
            cash_returns=d["cash_returns"],
            cash_nav=d["cash_nav"],
            cash_category=d["cash_category"],
            cash_rate=d["cash_rate"],
            execution_meta_by_date=d.get("execution_meta_by_date"),
        )
        for s, c in chain
    ]
    rows = [result_row(r) for r in results]
    md = to_markdown(rows, f"S2b — baseline zinciri ve sağlamlık — {datetime.now():%Y-%m-%d %H:%M}")
    md += "\n\n## Yıllık getiriler\n\n" + _pct_table(yearly_table(results))

    # teşhis: B1 ve B2 vergisiz koşu → sürtünme
    md += "\n\n## Sürtünme ayrışımı (vergi; TEFAS/BES işlem komisyonu ADR-27 ile 0, CAGR puanı)\n\n"
    fr = []
    gross_meta = replace(
        d["meta"], tax_rate=d["meta"].tax_rate * 0, entry_fee=d["meta"].entry_fee * 0, exit_fee=d["meta"].exit_fee * 0
    )
    for r, (s, c) in zip(results, chain, strict=True):
        if r.name.startswith(("B1_", "B2")):
            g = run_backtest(
                d["nav"],
                gross_meta,
                s,
                replace(c, cash_tax_rate=0.0),
                cash_returns=d["cash_returns"],
                cash_nav=d["cash_nav"],
                cash_category=d["cash_category"],
                cash_rate=d["cash_rate"],
                execution_meta_by_date=(
                    d.get("execution_meta_by_date").assign(entry_fee=0.0, exit_fee=0.0)
                    if d.get("execution_meta_by_date") is not None
                    else None
                ),
            )
            fr.append({"strategy": r.name, **friction_decomposition(r, g)})
    if fr:
        md += (pd.DataFrame(fr).set_index("strategy") * 100).round(
            1
        ).to_markdown() + "\n(değerler yüzde; friction_pts = brüt − net CAGR)"

    md += "\n\n## Tutulan fon türleri (B2 ortalama hedef ağırlık)\n\n"
    b2 = next(r for r in results if r.name.startswith("B2"))
    md += (holdings_by_type(b2, d["fund_master"]) * 100).round(1).to_frame("%").to_markdown()

    # sağlamlık: B3 üzerinde
    def b3_factory():
        return chain[-1][0]

    md += "\n\n## Sağlamlık — B3\n\n"
    n_starts = int(cfg.get("validation", {}).get("start_dates", 12))
    sd = start_date_sweep(
        d["nav"],
        d["meta"],
        b3_factory,
        bcfg,
        d["cash_returns"],
        n_starts=4 if quick else n_starts,
        cash_nav=d["cash_nav"],
        cash_category=d["cash_category"],
        cash_rate=d["cash_rate"],
        execution_meta_by_date=d.get("execution_meta_by_date"),
    )
    md += (
        "### 12 başlangıç tarihi\n\n"
        + (sd[["cagr", "excess_cagr", "mdd"]].describe().loc[["min", "50%", "max"]] * 100).round(1).to_markdown()
        + "\n\n"
    )
    st_cfg = cfg.get("validation", {}).get("stress", {})
    stress_rows = []
    for label, meta_s, execution_s in [
        ("baz", d["meta"], d.get("execution_meta_by_date")),
        (
            "+10bp/işlem",
            stressed_meta(d["meta"], extra_fee=st_cfg.get("extra_fee_bps", 10) / 1e4),
            stressed_execution_meta(d.get("execution_meta_by_date"), extra_fee=st_cfg.get("extra_fee_bps", 10) / 1e4),
        ),
        (
            "satış valörü +1",
            stressed_meta(d["meta"], valor_plus=int(st_cfg.get("valor_plus_days", 1))),
            stressed_execution_meta(d.get("execution_meta_by_date"), valor_plus=int(st_cfg.get("valor_plus_days", 1))),
        ),
    ]:
        r = run_backtest(
            d["nav"],
            meta_s,
            b3_factory(),
            bcfg,
            cash_returns=d["cash_returns"],
            cash_nav=d["cash_nav"],
            cash_category=d["cash_category"],
            cash_rate=d["cash_rate"],
            execution_meta_by_date=execution_s,
            allow_fee_stress=label == "+10bp/işlem",
        )
        m = result_row(r)
        stress_rows.append(
            {
                "senaryo": label,
                "cagr": m["cagr"],
                "excess_cagr": m["excess_cagr"],
                "mdd": m["mdd"],
                "sharpe": m["sharpe"],
            }
        )
    md += (
        "### Maliyet / valör stresi\n\n"
        + pd.DataFrame(stress_rows).set_index("senaryo").round(3).to_markdown()
        + "\n\n"
    )
    founders = d["fund_master"].drop_duplicates("fund_code").set_index("fund_code")["founder"]
    susp = suspension_scenario(
        d["nav"],
        d["meta"],
        b3_factory,
        bcfg,
        d["cash_returns"],
        founders,
        months=int(st_cfg.get("suspension", {}).get("months", 6)),
        seeds=2 if quick else int(cfg.get("validation", {}).get("seeds", 5)),
        cash_nav=d["cash_nav"],
        cash_category=d["cash_category"],
        cash_rate=d["cash_rate"],
        execution_meta_by_date=d.get("execution_meta_by_date"),
    )
    if not susp.empty:
        md += (
            "### Askı senaryosu (rastgele kurucu, haircut %30)\n\n"
            + susp.set_index("scenario")[["cagr", "excess_cagr", "mdd"]].round(3).to_markdown()
            + "\n\n"
        )

    # parametre taraması + PBO + deflated Sharpe (B1 ailesi)
    grid = dict(cfg.get("sweep", {"n": [5, 10, 15], "lookback": [126, 252], "buffer_mult": [1.0, 2.0]}))
    grid = {("cfg:schedule" if k == "schedule" else k): v for k, v in grid.items()}
    if quick:
        grid = {k: v[:2] for k, v in grid.items()}
    con = cfg["legs"]["tefas"]["constraints"]

    def factory(**p):
        return TopNMomentum(eligible=d["eligible"], max_weight=con["max_weight_per_fund"], tax_aware=True, **p)

    sweep_df, info = parameter_sweep(
        d["nav"],
        d["meta"],
        factory,
        grid,
        bcfg,
        d["cash_returns"],
        cash_nav=d["cash_nav"],
        cash_category=d["cash_category"],
        cash_rate=d["cash_rate"],
        execution_meta_by_date=d.get("execution_meta_by_date"),
    )
    md += "## Parametre taraması (B1 vergi-farkındalıklı, tampon 2N) — PBO ve deflated Sharpe\n\n"
    md += sweep_df.round(3).to_markdown(index=False) + "\n\n"
    md += f"- PBO (CSCV, {info['n_combos']} kombinasyon): **{info['pbo']:.2f}** (0,5 üstü = aşırı uyum baskın)\n"
    md += f"- En iyi konfig deflated Sharpe olasılığı: **{info['deflated_sharpe_prob']:.2f}** (0,95 üstü = seçim etkisi düşüldükten sonra anlamlı)\n"
    md += f"- En iyi: {info['best']}\n"
    md += "\n## Notlar\n- Tüm metrikler vergi ve maliyet sonrası; evren bugünün Faz-1 sağ kalanları (ADR-07: hafif iyimser).\n- Valör yönü config execution.valor_mapping.\n"
    if d.get("snapshot_asof") is not None:
        md += f"\n- veri as-of: {d['snapshot_asof']:%Y-%m-%d %H:%M} (parquet anlık görüntüsü)\n"

    path = None
    if out_dir is not None:
        path = save_report(md, out_dir, f"backtest_s2b_{datetime.now():%Y%m%d_%H%M}")
    if root is not None:
        from janus.backtest.mlflow_utils import log_rows  # noqa: PLC0415

        log_rows(rows, cfg, root, data_asof=str(d["nav"].index.max().date()))
    return md, path
