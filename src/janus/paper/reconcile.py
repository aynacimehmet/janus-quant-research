"""S5-2: günlük mutabakat raporu."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from janus.paper.core import SLOT_CODE, PaperLedger


def _proposal_targets(store, date: pd.Timestamp) -> pd.Series:
    row = store.con.execute(
        "SELECT target_weights_json FROM paper_proposals WHERE date <= ? ORDER BY date DESC LIMIT 1",
        [date.date()],
    ).fetchone()
    if not row or not row[0]:
        return pd.Series(dtype=float)
    return pd.Series(json.loads(row[0]), dtype=float)


def paper_reconcile(store, cfg: dict, date: str, out_dir: Path | None = None) -> dict:
    """Tarihli snapshot ile bağımsız defter değerini ve tam hedef ağırlıkları mutabık kıl."""
    d = pd.Timestamp(date).normalize()
    try:
        paper = PaperLedger(store, cfg, asof=d)
    except ValueError as exc:
        return {
            "date": date,
            "equity": float("nan"),
            "cash": float("nan"),
            "receivables": float("nan"),
            "risky_value": float("nan"),
            "slot_value": float("nan"),
            "identity_ok": False,
            "liquidation_value": float("nan"),
            "max_weight_diff": float("nan"),
            "n_expired": 0,
            "pnl_since_previous": None,
            "md_path": "",
            "error": str(exc),
        }
    nav = paper.latest_nav()
    arr = nav.reindex(paper.meta.index).to_numpy(float).copy()
    arr[paper._slot_i] = paper._slot_price() or 1.0
    eq = float(paper.ledger.equity(arr, None, 0.0))
    cash = float(paper.ledger.cash)
    rec = float(paper.ledger.receivable_total())
    _, risky_value, slot_value = paper.position_components(nav)
    component_equity = cash + rec + risky_value + slot_value
    saved = store.con.execute(
        "SELECT equity, cash, receivables, risky_value, slot_value FROM paper_equity "
        "WHERE portfolio_name='live' AND date=?",
        [d.date()],
    ).fetchone()
    tolerance = float(cfg.get("paper", {}).get("kpi", {}).get("identity_tol", 1e-6))
    identity_ok = abs(component_equity - eq) <= tolerance
    if saved is None:
        identity_ok = False
    else:
        saved_equity, *saved_parts = map(float, saved)
        identity_ok = identity_ok and abs(saved_equity - component_equity) <= tolerance
        identity_ok = identity_ok and all(
            abs(a - b) <= tolerance for a, b in zip(saved_parts, [cash, rec, risky_value, slot_value], strict=True)
        )

    current_weights = pd.Series(paper.ledger.weights(arr, None, 0.0), index=paper.meta.index)
    b0_codes = set(paper._b0_codes) | set(paper._b0_history_codes)
    b0_weight = float(current_weights.reindex(list(b0_codes)).fillna(0.0).sum())
    current_weights = current_weights.drop(index=list(b0_codes), errors="ignore")
    current_weights[SLOT_CODE] = b0_weight
    target = _proposal_targets(store, d).reindex(current_weights.index)
    if target.empty or target.isna().all():
        diff = pd.Series(dtype=float)
    else:
        target = target.fillna(0.0)
        diff = (target - current_weights).dropna()
        diff = diff[diff.abs() > 1e-6]

    expired = store.con.execute(
        "SELECT proposal_id, date FROM paper_proposals WHERE status = 'expired' AND date <= ?", [d.date()]
    ).df()
    liquidation_value = float(eq - paper.ledger.unrealized_tax(arr))
    previous = store.con.execute(
        "SELECT equity FROM paper_equity WHERE portfolio_name='live' AND date < ? ORDER BY date DESC LIMIT 1",
        [d.date()],
    ).fetchone()
    pnl_since_previous = eq - float(previous[0]) if previous else None

    lines = [
        f"# Mutabakat — {date}",
        f"**equity:** {eq:.4f} | **cash:** {cash:.4f} | **receivables:** {rec:.4f}",
        f"**risky:** {risky_value:.4f} | **slot:** {slot_value:.4f}",
        f"**özdeşlik:** {'OK' if identity_ok else 'HATA'}",
        f"**liquidation_value:** {liquidation_value:.4f}",
        f"**önceki snapshot'tan net PnL:** {pnl_since_previous if pnl_since_previous is not None else 'veri yok'}",
        "",
        "## Tam hedef vs gerçek ağırlık farkı (sepet dahil)",
        "| varlık | hedef % | gerçek % | fark % |",
        "|-----|---------|----------|--------|",
    ]
    for code, delta in diff.items():
        lines.append(
            f"| {code} | {target.get(code, 0.0) * 100:.1f} | {current_weights.get(code, 0.0) * 100:.1f} | {delta * 100:+.1f} |"
        )
    lines.extend(["", "## Süresi dolmuş öneriler", f"{len(expired)} adet"])
    for _, row in expired.iterrows():
        lines.append(f"- {row['proposal_id']} ({row['date']})")

    store.con.execute(
        """INSERT OR REPLACE INTO paper_reconcile
        (date, equity, cash, receivables, risky_value, slot_value, identity_ok,
         liquidation_value, max_weight_diff, n_expired) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            d.date(),
            eq,
            cash,
            rec,
            risky_value,
            slot_value,
            identity_ok,
            liquidation_value,
            float(diff.abs().max()) if not diff.empty else 0.0,
            len(expired),
        ],
    )
    out = Path(out_dir) if out_dir else Path(cfg.get("reporting", {}).get("orders_dir", "reports"))
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"reconcile_{date}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "date": date,
        "equity": eq,
        "cash": cash,
        "receivables": rec,
        "risky_value": risky_value,
        "slot_value": slot_value,
        "identity_ok": identity_ok,
        "liquidation_value": liquidation_value,
        "max_weight_diff": float(diff.abs().max()) if not diff.empty else 0.0,
        "n_expired": len(expired),
        "pnl_since_previous": pnl_since_previous,
        "md_path": str(md_path),
    }
