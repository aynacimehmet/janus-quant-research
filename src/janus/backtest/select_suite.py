"""S3b-4 suite: B0 nakit sepeti (referans) · B2b momentum · B2c conformal (q/m, +ceza); kapı yok (ADR-18).

Satırlar: B0 (nakit sepeti), B2b (MomentumHRP, çeyreklik), B2c-q (conformal, çeyreklik — B2b iskeleti),
B2c-m (conformal, aylık — "lower ≤ 0 → çık" ay bazında), B2c-m+ceza (tax_penalty yalnız B2c-m'de).

VALIDATION §1: seçim dönemi (panel başı → select_end) ve dış test (select_end → panel sonu) ayrı
metriklerle raporlanır; tek tam-dönem koşu, equity/cash_index pencere dilimleri üzerinden (warmup
artefaktını önler; vergi/turnover tam dönem birikimlidir — notta belirtilir). 12 başlangıç
duyarlılığı start_date_sweep ile; kapsama (genel + seçilen fonlar), aday sayısı, nakitte kalma
oranı, ortalama risky payı, fallback_days raporlanır. PBO/DSR 4 riskli satır üzerinden keşif amaçlıdır.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from janus.backtest.data import prepare
from janus.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from janus.backtest.metrics import deflated_sharpe, summary
from janus.backtest.report import save_report
from janus.backtest.robustness import b0_fund_valor_stress, pbo_cscv, start_date_sweep
from janus.data.store import Store
from janus.models.conformal import coverage_report
from janus.models.conformal_selection import FdrHRP, conformal_pvalues, selected_positive_rate
from janus.strategies.baselines import CashOnly
from janus.strategies.conformal_select import ConformalHRP, candidate_counts, selected_mask_from_history
from janus.strategies.portfolio import MomentumHRP

METRIC_COLS = ("cagr", "cash_cagr", "excess_cagr", "vol", "sharpe", "mdd")


def load_s3b_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """S3b-2/3 çıktıları: predictions, calibrated (target 0.20), features (ajana içerik gösterilmez)."""
    preds = pd.read_parquet(root / "data" / "predictions" / "predictions.parquet")
    cal = pd.read_parquet(root / "data" / "predictions" / "calibrated_target_020.parquet")
    feats = pd.read_parquet(root / "data" / "features" / "fund_features.parquet")
    return preds, cal, feats


def build_strategies(
    d: dict,
    cfg: dict,
    top_n: int,
    calibrated: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    kill_switch: bool = True,
    fdr_q: float = 0.20,
    fdr_q_grid: tuple[float, ...] = (),
    features_pit: pd.DataFrame | None = None,
) -> list[object]:
    con = cfg["legs"]["tefas"]["constraints"]
    hrp = cfg.get("hrp", {})
    founders = d["fund_master"].drop_duplicates("fund_code").set_index("fund_code")["founder"]
    common = dict(
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
    )
    smooth = hrp.get("smooth_lambda", 0.5)
    refresh = hrp.get("refresh_every", 3)

    def mk_conf(name: str, **k) -> ConformalHRP:
        return ConformalHRP(
            calibrated=calibrated,
            predictions=predictions,
            features=features,
            kill_switch=kill_switch,
            name=name,
            **common,
            **k,
        )

    rows: list[object] = [
        CashOnly(),
        MomentumHRP(name="B2b_hrp_q_smooth", refresh_every=refresh, smooth_lambda=smooth, **common),
        mk_conf("B2c_q_conformal", refresh_every=refresh, smooth_lambda=smooth),
        mk_conf("B2c_m_conformal", refresh_every=1, smooth_lambda=smooth),
        mk_conf("B2c_m_conformal_taxpen", refresh_every=1, smooth_lambda=smooth, tax_penalty=True),
    ]
    if features_pit is not None:  # kill-switch v2 (S3b-5-3): referans nakit
        rows += [
            MomentumHRP(
                name="B2b_ks2",
                ks2=True,
                ks2_features=features_pit,
                refresh_every=refresh,
                smooth_lambda=smooth,
                **common,
            ),
            mk_conf("B2c_m_ks2", refresh_every=1, smooth_lambda=smooth, ks2=True, ks2_features=features_pit),
        ]
    if predictions is not None and features is not None:  # S3b-5-2/5-3: B2c-fdr + q ızgarası
        pv = conformal_pvalues(predictions, features)  # p-değerleri bir kez; BH q başına ucuz
        rows.append(
            FdrHRP(
                calibrated=calibrated,
                predictions=predictions,
                features=features,
                pvalues=pv,
                kill_switch=False,
                fdr_q=fdr_q,
                name="B2c_fdr_hrp",
                refresh_every=1,
                smooth_lambda=smooth,
                **common,
            )
        )
        for qg in fdr_q_grid:
            if abs(qg - fdr_q) < 1e-9:
                continue  # varsayılan q satırı zaten var
            rows.append(
                FdrHRP(
                    calibrated=calibrated,
                    predictions=predictions,
                    features=features,
                    pvalues=pv,
                    kill_switch=False,
                    fdr_q=qg,
                    name=f"B2c_fdr_hrp_q{int(round(qg * 100)):02d}",
                    refresh_every=1,
                    smooth_lambda=smooth,
                    **common,
                )
            )
    return rows


def _period_metrics(res: BacktestResult, start=None, end=None) -> dict:
    eq, ci = res.equity, res.cash_index
    if start is not None:
        eq, ci = eq[eq.index >= pd.Timestamp(start)], ci[ci.index >= pd.Timestamp(start)]
    if end is not None:
        eq, ci = eq[eq.index <= pd.Timestamp(end)], ci[ci.index <= pd.Timestamp(end)]
    m = summary(eq, ci, res.taxes_paid, res.fees_paid, res.turnover)
    return {k: m[k] for k in METRIC_COLS}


AVAIL_COLS = [
    "strategy",
    "aday_min_seç",
    "aday_med_seç",
    "aday_max_seç",
    "aday_min_dış",
    "aday_med_dış",
    "aday_max_dış",
    "nakit_kalma",
    "ort_risky",
    "fallback_gün",
]


def _availability_row(res: BacktestResult, strat: object, cands: pd.Series, sel_end: pd.Timestamp) -> list:
    """Kullanılabilirlik satırı; aday sayısı dönem bazında (seçim / dış test).

    Not: aday sayıları yalnızca rebalance günleri için sayılır; kanıt günü (decision_at,
    lower/p-değerinin hazır olduğu gün) ile işlem günü aynı takvim günü olsa da kavramsal
    ayrım korunur (TEMPORAL §3/§6).
    """
    w = res.weights_realized if not res.weights_realized.empty else res.weights
    if "CASH_PROXY" in w.columns:
        w = w.drop(columns=["CASH_PROXY"])  # risky pay = nakit slotu hariç (Ekle-2)
    if w.empty:
        cash_stay, avg_risky = 1.0, 0.0
    else:
        sums = w.sum(axis=1)
        cash_stay = float((sums < 1e-9).mean())
        avg_risky = float(sums.mean())
    # FdrHRP satırlarında aday sayısı = BH küme büyüklüğü; yalnızca rebalance günleri
    if hasattr(strat, "_bh") and strat._bh:
        reb_days = set(res.weights.index)
        bh_counts = pd.Series({d: len(s) for d, s in strat._bh.items() if pd.Timestamp(d) in reb_days}).sort_index()
        cands = bh_counts
    else:
        # aday sayısını yalnızca rebalance günlerine indir
        cands = cands[cands.index.isin(res.weights.index)]
    c_sel = cands[cands.index <= sel_end]
    c_ext = cands[cands.index > sel_end]

    def stats(c: pd.Series) -> list:
        return [int(c.min()) if len(c) else 0, float(c.median()) if len(c) else 0.0, int(c.max()) if len(c) else 0]

    return [
        getattr(strat, "name", str(type(strat).__name__)),
        *stats(c_sel),
        *stats(c_ext),
        f"%{cash_stay * 100:.1f}",
        f"{avg_risky:.2f}",
        int(getattr(strat, "fallback_days", 0)),
    ]


def _selected_coverage(calibrated: pd.DataFrame, strat: object) -> pd.DataFrame:
    history = getattr(strat, "_history", {})
    if not history:
        return pd.DataFrame(columns=["month", "coverage", "n_rows"])
    return coverage_report(calibrated, selected_mask_from_history(calibrated, history))


def _pct(x: float) -> str:
    return f"%{x * 100:.1f}" if pd.notna(x) else "—"


def run_select_suite(
    d: dict,
    cfg: dict,
    top_n: int = 10,
    select_end: str = "2025-06-30",
    out_dir: Path | None = None,
    quick: bool = False,
    predictions: pd.DataFrame | None = None,
    calibrated: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    kill_switch: bool = True,
) -> tuple[str, Path | None]:
    """Suite koşusu. d = prepare(store, cfg) çıktısı; S3b-2/3 çerçeveleri opsiyonel enjekte edilir."""
    if calibrated is None:
        predictions, calibrated, features = load_s3b_inputs(Path(__file__).resolve().parents[3])
    bcfg = BacktestConfig.from_cfg(cfg)
    fdr_q = float(cfg.get("conformal", {}).get("fdr_q", 0.20))
    fdr_q_grid = tuple(float(q) for q in cfg.get("conformal", {}).get("fdr_q_grid", []) or [])
    strats = build_strategies(
        d, cfg, top_n, calibrated, predictions, features, kill_switch, fdr_q, fdr_q_grid, features
    )
    results = [
        run_backtest(
            d["nav"],
            d["meta"],
            s,
            bcfg,
            cash_returns=d["cash_returns"],
            cash_nav=d["cash_nav"],
            cash_category=d["cash_category"],
            cash_rate=d["cash_rate"],
            execution_meta_by_date=d.get("execution_meta_by_date"),
        )
        for s in strats
    ]
    exp_id = f"select_suite_{datetime.now():%Y%m%d_%H%M}"
    sel_end = pd.Timestamp(select_end)

    rows = []
    for res in results:
        row = {"strategy": res.name}
        for tag, (s_, e_) in {
            "full": (None, None),
            "select": (None, sel_end),
            "external": (sel_end, None),
        }.items():
            for k, v in _period_metrics(res, s_, e_).items():
                row[f"{k}_{tag}"] = v
        rows.append(row)
    md = [
        f"# S3b-4 select-suite — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Deney kimliği: `{exp_id}` · seçim dönemi: panel başı → {sel_end.date()} · dış test: {sel_end.date()} → panel sonu",
        "",
        "Tüm metrikler vergi ve maliyet sonrası; excess = B0 sepet defteri karşısında (pp için fark sütunları pp).",
        "",
        "| strategy | " + " | ".join(f"{c}" for c in rows[0] if c != "strategy") + " |",
        "|---|" + "---|" * (len(rows[0]) - 1),
    ]
    for r in rows:
        vals = [
            (_pct(r[c]) if ("cagr" in c or c.startswith(("vol", "mdd"))) else f"{r[c]:.2f}")
            if isinstance(r[c], float)
            else str(r[c])
            for c in r
            if c != "strategy"
        ]
        md.append("| " + r["strategy"] + " | " + " | ".join(vals) + " |")

    # ADR-0023 requires B0's own real-fund execution stress; B3/other-strategy stress is not a substitute.
    b0_proxy = next(row for row in rows if row["strategy"] == "B0_cash")
    cal = pd.DatetimeIndex(d["nav"].index)
    cutoff_indices = np.flatnonzero(cal <= sel_end)
    synthetic_liquidation_idx = int(cutoff_indices[-1]) if len(cutoff_indices) else None
    if synthetic_liquidation_idx is not None and not 0 < synthetic_liquidation_idx < len(cal) - 1:
        synthetic_liquidation_idx = None
    b0_fund_rows = b0_fund_valor_stress(
        d["nav"],
        list(d.get("cash_codes", [])),
        d["meta"],
        d.get("execution_meta_by_date"),
        select_end=sel_end,
        initial_capital=bcfg.initial_capital,
        synthetic_liquidation_idx=synthetic_liquidation_idx,
        cash_returns=d["cash_returns"] if synthetic_liquidation_idx is not None else None,
    )
    stress_by_period = {
        (str(row.scenario), str(row.period)): row
        for row in b0_fund_rows.query("mode == 'fixed_terminal'").itertuples(index=False)
    }
    sensitivity_by_scenario = {
        str(row.scenario): row for row in b0_fund_rows.query("mode == 'synthetic_reinvestment'").itertuples(index=False)
    }
    b0_report_rows = []
    for period in ("full", "select", "external"):
        proxy_cagr = b0_proxy.get(f"cagr_{period}", np.nan)
        b0_report_rows.append(
            {
                "period": period,
                "B0 yolu": "endeks vekili (referans)",
                "CAGR": _pct(proxy_cagr),
                "fon − proxy (pp)": "—",
                "durum / varsayım": "mevcut B0 endeks vekili; fon lotu defteri değildir",
                "tahsilat": "—",
            }
        )
        for scenario, label in (
            ("fon-baz", "B0 sabit terminal — icra edilebilir fon sepeti — baz"),
            ("satış valörü +1", "B0 sabit terminal — icra edilebilir fon sepeti — satış valörü +1"),
        ):
            row = stress_by_period[(scenario, period)]
            fund_cagr = float(row.cagr) if pd.notna(row.cagr) else np.nan
            diff_pp = (fund_cagr - proxy_cagr) * 100 if pd.notna(fund_cagr) and pd.notna(proxy_cagr) else np.nan
            b0_report_rows.append(
                {
                    "period": period,
                    "B0 yolu": label,
                    "CAGR": _pct(fund_cagr),
                    "fon − proxy (pp)": f"{diff_pp:+.2f} pp" if pd.notna(diff_pp) else "—",
                    "durum / varsayım": f"{row.status}; {row.assumption_label}; {row.event_note}",
                    "tahsilat": str(row.settle_idx) if pd.notna(row.settle_idx) else "—",
                }
            )
    sensitivity_report_rows = []
    for scenario, label in (
        ("fon-baz", "ufuk-içi sentetik yeniden yatırım duyarlılığı — baz"),
        ("satış valörü +1", "ufuk-içi sentetik yeniden yatırım duyarlılığı — satış valörü +1"),
    ):
        row = sensitivity_by_scenario.get(scenario)
        sensitivity_report_rows.append(
            {
                "B0 yolu": label,
                "CAGR": _pct(float(row.cagr)) if row is not None and pd.notna(row.cagr) else "—",
                "tahsilat settle_idx": str(row.settle_idx) if row is not None and pd.notna(row.settle_idx) else "—",
                "durum / varsayım": (
                    f"{row.status}; {row.assumption_label}; {row.event_note}"
                    if row is not None
                    else "değerlendirilemedi: önceden belirlenen seçim kesiminde ufuk-içi NAV yok"
                ),
            }
        )
    md += [
        "",
        "## B0: endeks vekili ve fon-lot satış valörü stresi (ADR-0023) — B0 sabit terminal",
        "",
        f"Deney kimliği: `{exp_id}`. Dönem sonu tasfiye sentetik muhasebe olayıdır, öneri/aylık satış değildir. "
        "Fon baz / +1 satırları Ledger FIFO, stopaj ve ADR-27 uyarınca sıfır TEFAS komisyonunu içerir; "
        "alacak equity içinde kalır. "
        "Yalnız full satırı ortak ilk alım ve panel-sonu terminal tasfiyeyi muhasebeleştirir; select/external fon-lot "
        "satırları dönem sınırında bağımsız tasfiye/yeniden giriş olmadığından değerlendirilmez, CAGR NaN ve fon−proxy —. "
        "Bunlar eski lot/gölge dönem net testi değildir; select/external B0 endeks vekili satırları korunur. "
        "Fark = fon CAGR − proxy CAGR, yüzde değil pp; alfa iddiası değildir.",
        'Önceki ec70e5b satırı bulunmadığından eski stres "üretilmedi, pp hesaplanamaz"; aşağıdaki satırlar bu koşuya aittir.',
        "",
        "| dönem | B0 yolu | CAGR | fon − proxy (pp) | durum / varsayım | tahsilat settle_idx |",
        "|---|---|---:|---:|---|---:|",
        *[
            "| "
            + " | ".join(
                [
                    str(row[key])
                    for key in ("period", "B0 yolu", "CAGR", "fon − proxy (pp)", "durum / varsayım", "tahsilat")
                ]
            )
            + " |"
            for row in b0_report_rows
        ],
        "",
        "Sabit üyeler `d['cash_codes']` girdisidir; helper üye seçmez/değiştirmez. Profil, NAV veya eşit ilk alım/tasfiye "
        "koşulları sağlanmazsa fon satırı `değerlendirilemedi` kalır; endeks vekili satırı yine raporlanır.",
        "",
        "### Ufuk-içi sentetik yeniden yatırım duyarlılığı",
        "",
        f"Deney kimliği: `{exp_id}`. Tasfiye indeksi={synthetic_liquidation_idx} "
        f"(tarih={cal[synthetic_liquidation_idx].date() if synthetic_liquidation_idx is not None else '—'}), "
        f"seçim kesimi={sel_end.date()} ile sabitlenir; "
        "gelecek NAV değerleriyle seçilmez. Alacak yalnız `Ledger.settle` sonrası varsayımsal B0 getiri serisine alınır; "
        "alacak settle öncesi getiri tahakkuk etmez. `Ledger.accrue_cash` günlük vergili nakit yaklaşımı kullanır; "
        "bu gerçek fon-lot nakit getirisi veya kağıt defter/canlı üretim sonucu değildir ve aynı gün zincirleme üretim "
        "akışı değildir. Bu satırlar sentetiktir/varsayımsaldır; gerçek tarihsel PPF getirisi veya alfa kanıtı değildir. "
        "CAGR yalnız bu duyarlılığın baz/+1 senaryoları arasında yorumlanır; sabit terminal satırlarıyla birleştirilmez.",
        "",
        "| B0 yolu | CAGR | tahsilat settle_idx | durum / varsayım |",
        "|---|---:|---:|---|",
        *[
            "| "
            + " | ".join(str(row[key]) for key in ("B0 yolu", "CAGR", "tahsilat settle_idx", "durum / varsayım"))
            + " |"
            for row in sensitivity_report_rows
        ],
        "",
    ]

    # kullanılabilirlik: aday sayısı (dönem bazında), nakitte kalma oranı, ortalama risky payı, fallback_days
    cands = candidate_counts(calibrated)
    avail_rows = [_availability_row(res, s, cands, sel_end) for res, s in zip(results, strats, strict=True)]
    md += [
        "",
        "## Kullanılabilirlik",
        "",
        "| " + " | ".join(AVAIL_COLS) + " |",
        "|---|" + "---|" * (len(AVAIL_COLS) - 1),
        *["| " + " | ".join(str(v) for v in row) + " |" for row in avail_rows],
        "",
        "aday_* = lower > 0 fon sayısı (seç = seçim dönemi, dış = dış test); nakit_kalma = risky payı 0 olan rebalance günü oranı; "
        "ort_risky = ortalama risky payı (nakit slotu hariç); fallback_gün = kill-switch'in momentum'a düşürdüğü gün sayısı.",
        "",
    ]

    # kapsama: genel + seçilen fonlar (B2c satırları)
    cov_gen = coverage_report(calibrated)
    md += [
        "## Kapsama (aylık)",
        "",
        f"- Genel OOS kapsama (tüm aralıklı satırlar): son ay {_pct(cov_gen['coverage'].iloc[-1]) if len(cov_gen) else '—'}",
    ]
    for res, s in zip(results, strats, strict=True):
        if not hasattr(s, "_history"):
            continue
        cov_sel = _selected_coverage(calibrated, s)
        if len(cov_sel):
            tbl = cov_sel.copy()
            tbl["coverage"] = tbl["coverage"].map(_pct)
            md += [
                f"### Seçilen-fon kapsama — {res.name}",
                "",
                tbl.to_markdown(index=False),
                "",
            ]
    md.append("")

    # seçim kalitesi: seçilen kümede gerçekleşen y > 0 oranı (ampirik 1−FDR karşılığı)
    md += ["## Seçim kalitesi (seçilen kümede y > 0 oranı; ampirik 1−FDR karşılığı)", ""]
    for res, s_ in zip(results, strats, strict=True):
        if not hasattr(s_, "_history"):
            continue
        r_sel, n_sel = selected_positive_rate(calibrated, s_._history, end=sel_end)
        r_ext, n_ext = selected_positive_rate(calibrated, s_._history, start=sel_end)
        md.append(f"- {res.name}: seçim {_pct(r_sel)} ({int(n_sel)} satır) · dış {_pct(r_ext)} ({int(n_ext)} satır)")
    md.append("")

    # 12 başlangıç duyarlılığı (B2b ve B2c-m)
    n_starts = 4 if quick else int(cfg.get("validation", {}).get("start_dates", 12))
    md += ["## 12 başlangıç duyarlılığı (excess_cagr, B0 sepet karşısında)", ""]
    for res, s in zip(results, strats, strict=True):
        if res.name not in ("B2b_hrp_q_smooth", "B2c_m_conformal", "B2c_fdr_hrp"):
            continue
        params = {k: v for k, v in s.__dict__.items() if not k.startswith("_")}

        def factory(st=s, p=params) -> object:
            return st.__class__(**p)

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
        md += [
            f"### {res.name}",
            (sd[["cagr", "excess_cagr", "mdd"]].describe().loc[["min", "50%", "max"]] * 100).round(1).to_markdown(),
            "",
        ]

    # PBO / deflated Sharpe (4 riskli satır; keşif amaçlı — tarama boyutu küçük)
    rets = []
    for res in results[1:]:
        r = np.log(res.equity).diff().dropna()
        rc = np.log(res.cash_index).diff().reindex(r.index).fillna(0.0)
        rets.append((r - rc).to_numpy())
    R = np.column_stack(rets)
    info = pbo_cscv(R)
    per_obs_sr = R.mean(0) / (R.std(0, ddof=1) + 1e-12)
    best = int(np.argmax(per_obs_sr))
    dsr = deflated_sharpe(
        float(per_obs_sr[best]),
        n_trials=R.shape[1],
        sr_var=float(per_obs_sr.var(ddof=1)),
        n_obs=R.shape[0],
        skew_=float(pd.Series(R[:, best]).skew()),
        kurt_=float(pd.Series(R[:, best]).kurt() + 3),
    )
    n_risky = len(results) - 1
    md += [
        f"## Tarama istatistikleri ({n_risky} riskli satır; keşif amaçlı)",
        "",
        f"- PBO (CSCV, {info['n_combos']} kombinasyon): **{info['pbo']:.2f}**",
        f"- En iyi satırın deflated Sharpe olasılığı: **{dsr:.2f}** (n_trials = {R.shape[1]})",
        "",
        "## Notlar",
        f"- veri as-of: {d['snapshot_asof']:%Y-%m-%d %H:%M} (parquet anlık görüntüsü)"
        if d.get("snapshot_asof")
        else "- veri as-of: DuckDB (anlık görüntü yok)",
        "- tax_penalty = lower × (1 − stopaj): yalnızca sıralama cezası; 'vergi sonrası getiri' denmez (H10).",
        "- Kill-switch: TEMPORAL §7; fallback_days kullanılabilirlik tablosunda.",
        "- Kabul ölçütü (ADR-18): dış testte excess_cagr_pp ≥ +1 ve 12 başlangıçta medyan ≥ 0; geçmezse üretim = B0.",
        "- A4 (valör yönü) kapalıdır (kullanıcı doğrulaması, 24.09.2026); A5 (zarar mahsubu) açık kabul "
        "engelidir ve sonuçlar bu varsayıma bağlıdır.",
        "- Dönem metriklerinde vergi/turnover tam dönem birikimlidir; pencere dilimi yalnız getiri/Sharpe/MDD için güvenlidir.",
        "- Kurucu göstergesi (H11) tarihsel deneyde KULLANILMAZ (PIT tarihçe yok).",
    ]
    md_txt = "\n".join(md)
    path = None
    if out_dir is not None:
        path = save_report(md_txt, out_dir, exp_id)
    return md_txt, path


def run_select_suite_from_store(
    store: Store,
    cfg: dict,
    top_n: int = 10,
    select_end: str = "2025-06-30",
    out_dir: Path | None = None,
    quick: bool = False,
) -> tuple[str, Path | None]:
    d = prepare(store, cfg)
    return run_select_suite(d, cfg, top_n=top_n, select_end=select_end, out_dir=out_dir, quick=quick)
