"""Backtest raporu (Markdown; yalnızca yüzde/oran, mutlak tutar yok)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from janus.backtest.engine import BacktestResult
from janus.backtest.metrics import summary


def result_row(res: BacktestResult) -> dict:
    m = summary(res.equity, res.cash_index, res.taxes_paid, res.fees_paid, res.turnover)
    m["strategy"] = res.name
    m["n_rebalances"] = res.n_rebalances
    m["dd_triggers"] = int((res.events["event"] == "dd_trigger").sum()) if not res.events.empty else 0
    return m


def to_markdown(rows: list[dict], title: str, notes: list[str] | None = None) -> str:
    df = pd.DataFrame(rows).set_index("strategy")
    cols = [
        "cagr",
        "cash_cagr",
        "excess_cagr",
        "vol",
        "sharpe",
        "sortino",
        "mdd",
        "turnover_per_year",
        "taxes_pct_of_final",
        "n_rebalances",
        "dd_triggers",
    ]
    df = df[[c for c in cols if c in df.columns]]
    pct = ["cagr", "cash_cagr", "excess_cagr", "vol", "mdd", "taxes_pct_of_final"]
    fmt = df.copy()
    for c in pct:
        if c in fmt:
            fmt[c] = (fmt[c] * 100).map(lambda x: f"%{x:.1f}")
    for c in ("sharpe", "sortino", "turnover_per_year"):
        if c in fmt:
            fmt[c] = fmt[c].map(lambda x: f"{x:.2f}")
    lines = [
        f"# {title}",
        "",
        "Tüm metrikler vergi ve maliyet sonrası; getiriler yıllık; tutar yok (yüzde).",
        "",
        fmt.to_markdown(),
    ]
    if notes:
        lines += ["", "## Notlar", *[f"- {n}" for n in notes]]
    return "\n".join(lines)


def save_report(md: str, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{stem}.md"
    p.write_text(md, encoding="utf-8")
    return p
