"""S5-2: kağıt gerçekleşme (fill) — kullanıcı bildirimi veya otomatik simülasyon."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from janus.data.quality import flag_suspended_status, is_business_day
from janus.paper.core import PaperLedger, _profile_execution_reasons
from janus.strategies.hrp import constrain_founder_targets


def _next_business_day(d: pd.Timestamp, cfg: dict) -> pd.Timestamp:
    """D'nin bir sonraki iş günü (hafta sonu ve config tatillerini atlayarak)."""
    nxt = d + pd.Timedelta(days=1)
    holidays = set(cfg.get("calendar", {}).get("holidays", []))
    while nxt.weekday() >= 5 or nxt.strftime("%Y-%m-%d") in holidays:
        nxt += pd.Timedelta(days=1)
    return nxt


def _date_idx(dates: pd.DatetimeIndex, d: pd.Timestamp) -> int | None:
    ts = pd.Timestamp(d).normalize()
    if ts in dates:
        return int(dates.get_loc(ts))
    return None


def _valor(value: object) -> int | None:
    """Valör bilinmiyorsa T+0 varsayma."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _load_proposal(store, proposal_id: str):
    row = store.con.execute(
        "SELECT proposal_id, date, created_at, expires_at, status, orders_json FROM paper_proposals WHERE proposal_id = ?",
        [proposal_id],
    ).fetchone()
    if not row:
        return None
    return {
        "proposal_id": row[0],
        "date": pd.Timestamp(row[1]),
        "created_at": row[2],
        "expires_at": row[3],
        "status": row[4],
        "orders": pd.DataFrame(json.loads(row[5]) if row[5] else []),
    }


def _load_fills(store, proposal_id: str) -> pd.DataFrame:
    df = store.con.execute("SELECT * FROM paper_fills WHERE proposal_id = ?", [proposal_id]).df()
    return df


def paper_fill(
    store,
    cfg: dict,
    proposal_id: str,
    pct: float = 1.0,
    price_date: str | None = None,
    skip: bool = False,
    auto: bool = False,
) -> dict:
    """Bir öneriyi deftere işle; 12:00 sonrası/önceden işlenmişse reddet; idempotent."""
    proposal = _load_proposal(store, proposal_id)
    if proposal is None:
        return {"status": "error", "proposal_id": proposal_id, "message": "proposal bulunamadı"}

    if proposal["status"] in ("filled", "skipped", "rejected"):
        return {
            "status": proposal["status"],
            "proposal_id": proposal_id,
            "fills": _load_fills(store, proposal_id),
            "n_fills": len(_load_fills(store, proposal_id)),
            "message": f"proposal zaten {proposal['status']}",
        }
    if proposal["status"] == "pending":
        pending_receivable = store.con.execute(
            "SELECT count(*) FROM paper_order_results WHERE proposal_id=? AND outcome='pending' "
            "AND reason='B0 satış alacağı aynı gün kullanılamaz'",
            [proposal_id],
        ).fetchone()[0]
        if pending_receivable:
            prior_fills = _load_fills(store, proposal_id)
            return {
                "status": "pending",
                "proposal_id": proposal_id,
                "fills": prior_fills,
                "n_fills": len(prior_fills),
                "message": "T+1 satış alacağı bekliyor; yeniden dolum fiyatı kullanıcı kararı olmadan belirlenemez",
            }

    tz_name = cfg.get("project", {}).get("timezone", "Europe/Istanbul")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:  # noqa: BLE001
        tz = None
    now = datetime.now(tz)
    if not auto and not skip and proposal["expires_at"]:
        exp = pd.Timestamp(proposal["expires_at"])
        if exp.tzinfo is None:
            exp = exp.tz_localize(tz or "UTC")
        if now > exp.to_pydatetime():
            store.con.execute("UPDATE paper_proposals SET status = 'expired' WHERE proposal_id = ?", [proposal_id])
            return {"status": "expired", "proposal_id": proposal_id, "message": "öneri süresi doldu"}

    if skip:
        store.con.execute("UPDATE paper_proposals SET status = 'skipped' WHERE proposal_id = ?", [proposal_id])
        return {"status": "skipped", "proposal_id": proposal_id, "fills": pd.DataFrame()}

    d = proposal["date"].normalize()
    expected_fill_d = _next_business_day(d, cfg)
    fill_d = pd.Timestamp(price_date).normalize() if price_date else expected_fill_d
    if fill_d != expected_fill_d:
        return {
            "status": "error",
            "proposal_id": proposal_id,
            "message": f"fill tarihi {fill_d.date()} karar D+1 ({expected_fill_d.date()}) değil",
        }
    nav = store.nav_wide()
    if nav.empty or fill_d not in nav.index:
        return {
            "status": "error",
            "proposal_id": proposal_id,
            "message": f"fill tarihi {fill_d.date()} NAV'da yok",
        }

    try:
        paper = PaperLedger(store, cfg, asof=fill_d)
    except ValueError as exc:
        return {"status": "error", "proposal_id": proposal_id, "message": str(exc)}
    dates = pd.DatetimeIndex(paper.ledger.dates)
    fill_idx = _date_idx(dates, fill_d)
    if fill_idx is None:
        return {"status": "error", "proposal_id": proposal_id, "message": "takvim indeksi bulunamadı"}
    decision_idx = _date_idx(dates, d)
    if decision_idx is None:
        decision_idx = max(0, int(dates.searchsorted(d, side="right") - 1))
    # Budget is frozen at decision D. Claims settling on D+1 and proceeds from
    # proposal SELLs may not finance BUY orders from this proposal.
    remaining_buy_budget = float(paper.ledger.cash) + sum(
        amount for settle_idx, amount in paper.ledger.receivables.items() if settle_idx <= decision_idx
    )
    paper.ledger.settle(fill_idx)
    trade_nav_arr = paper._nav.loc[fill_d].reindex(paper.meta.index).to_numpy(float).copy()
    mark_nav_arr = paper.latest_nav().reindex(paper.meta.index).to_numpy(float).copy()
    eq = paper.ledger.equity(mark_nav_arr, None, 0.0)
    position_values = paper.ledger.position_value(mark_nav_arr, None, 0.0)
    current_weights = pd.Series(
        position_values / eq if eq > 0 else np.zeros_like(position_values), index=paper.meta.index
    )
    fill_targets = current_weights.copy()
    for _, order in proposal["orders"].iterrows():
        if str(order.get("action", "")) == "BUY" and order.get("fund_code") in fill_targets.index:
            code = str(order["fund_code"])
            fill_targets.loc[code] += abs(float(order.get("delta_w", 0.0))) * pct
    founder_codes = paper._fund_master.drop_duplicates("fund_code").set_index("fund_code").get("founder_code")
    effective_fill_targets = constrain_founder_targets(
        fill_targets,
        current_weights,
        founder_codes if founder_codes is not None else pd.Series(dtype="string"),
        max_weight=float(cfg["legs"]["tefas"]["constraints"].get("max_weight_per_founder", 0.30)),
        max_funds=int(cfg["legs"]["tefas"]["constraints"].get("max_funds_per_founder", 3)),
    )
    b0_codes = set(paper._b0_codes) | set(paper._b0_history_codes)
    risky_codes = [code for code in current_weights.index if code not in b0_codes]
    risky_weight = float(current_weights.reindex(risky_codes).sum())
    constraints = cfg["legs"]["tefas"]["constraints"]
    risk = cfg["legs"]["tefas"]["risk"]
    risky_ceiling = float(constraints.get("pilot_risky_ceiling", 0.30))
    if paper.drawdown() > float(risk.get("dd_trigger_medium", 0.12)):
        risky_ceiling = min(risky_ceiling, float(risk.get("dd_exposure", 0.65)))
    risky_room = max(0.0, risky_ceiling - risky_weight)

    fills = []
    outcomes = []
    prior = store.con.execute(
        "SELECT order_index, outcome, reason, units, requested_value FROM paper_order_results WHERE proposal_id=?",
        [proposal_id],
    ).df()
    profiles = store.con.execute("SELECT * FROM fund_master").df()
    morning_cutoff = cfg.get("project", {}).get("runs", {}).get("morning", "09:15")
    proposal_codes = [str(code) for code in proposal["orders"].get("fund_code", pd.Series(dtype=str))]
    decision_blocks = _profile_execution_reasons(profiles, d, morning_cutoff, codes=proposal_codes)
    fill_blocks = _profile_execution_reasons(profiles, fill_d, morning_cutoff, codes=proposal_codes)
    prior_by_index = {int(row.order_index): row for row in prior.itertuples(index=False)}
    for order_idx, (_, order) in enumerate(proposal["orders"].iterrows()):
        previous = prior_by_index.get(order_idx)
        if previous is not None and previous.outcome == "filled":
            if str(order.get("action", "")) == "BUY":
                remaining_buy_budget = max(0.0, remaining_buy_budget - float(previous.requested_value))
            outcomes.append(
                (
                    order_idx,
                    order["fund_code"],
                    order["action"],
                    abs(float(order.get("delta_w", 0.0)) * eq * pct),
                    "filled",
                    "önceki denemede gerçekleşti",
                    float(previous.units),
                )
            )
            continue
        code = order["fund_code"]
        action = str(order.get("action", ""))
        requested = abs(float(order.get("delta_w", 0.0)) * eq * pct)
        outcome, reason, order_units = "rejected", "geçersiz fon/emir", 0.0
        if code not in paper.meta.index:
            outcomes.append((order_idx, code, action, requested, outcome, reason, order_units))
            continue
        i = int(paper.meta.index.get_loc(code))
        price = float(trade_nav_arr[i])
        suspended = bool(flag_suspended_status(paper._fund_master).get(code, False))
        d_block = decision_blocks.get(str(code), {}).get(action)
        fill_block = fill_blocks.get(str(code), {}).get(action)
        if not np.isfinite(price) or price <= 0:
            reason = "fill tarihinde geçerli NAV yok (data_stale)"
        elif suspended:
            reason = "fon askıda"
        elif d_block:
            reason = f"karar D PIT profili: {d_block}"
        elif fill_block:
            reason = f"D+1 PIT profili: {fill_block}"
        elif action == "BUY" and _valor(paper.meta.buy_valor[i]) is None:
            reason = "alış valörü bilinmiyor"
        elif action == "SELL" and _valor(paper.meta.sell_valor[i]) is None:
            reason = "satış valörü bilinmiyor"
        elif action == "BUY" and not paper.meta.can_buy[i]:
            reason = "can_buy=False"
        elif action == "SELL" and not paper.meta.can_sell[i]:
            reason = "can_sell=False"
        elif action == "BUY":
            requested_amount = float(order["delta_w"]) * eq * pct
            allowed_weight = max(0.0, float(effective_fill_targets.get(code, 0.0) - current_weights.get(code, 0.0)))
            if code not in b0_codes:
                fund_room = max(
                    0.0,
                    float(constraints.get("max_weight_per_fund", 0.25)) - float(current_weights.get(code, 0.0)),
                )
                allowed_weight = min(allowed_weight, fund_room, risky_room)
            amount = min(requested_amount, allowed_weight * eq)
            if amount <= 0:
                reason = "D+1 gerçekleşebilir ortak PYŞ kapasitesi yok"
            elif amount > remaining_buy_budget + 1e-9:
                reason = "karar D serbest nakdi yetersiz; yeni proposal gerekli"
            else:
                fill = paper.ledger.buy(i, amount, price, fill_idx)
                if fill:
                    fills.append(fill.__dict__)
                    remaining_buy_budget -= amount
                    if code not in b0_codes and eq > 0:
                        risky_room = max(0.0, risky_room - amount / eq)
                    if amount + 1e-9 < requested_amount:
                        outcome, reason = "pending", "D+1 ortak portföy PYŞ/risky kapasitesi nedeniyle kısmi"
                    else:
                        outcome, reason = "filled", ""
                    order_units = float(fill.units)
                else:
                    reason = "serbest nakit yetersiz veya Ledger alımı reddetti"
        elif action == "SELL":
            current_units = paper.ledger.units[i]
            target_units = float(order.get("target_w", 0.0)) * eq / price
            desired_units = abs(current_units - target_units) * pct
            sell_units = min(desired_units, paper.ledger.sellable_units(i, fill_idx))
            if sell_units <= 1e-12:
                outcome, reason = "pending", "lot valörü henüz dolmadı"
            else:
                fill = paper.ledger.sell(i, sell_units, price, fill_idx)
                if fill:
                    fills.append(fill.__dict__)
                    order_units = float(fill.units)
                    if fill.units + 1e-10 < desired_units:
                        outcome, reason = "pending", "satılabilir lot kısmi; kalan valör bekliyor"
                    else:
                        outcome, reason = "filled", ""
                else:
                    reason = "Ledger satışı reddetti"
        outcomes.append((order_idx, code, action, requested, outcome, reason, order_units))

    # Her emir sonucu, gerçekleşen fill'den bağımsız olarak kalıcıdır.
    now_ts = datetime.now()
    for order_idx, code, side, requested, outcome, reason, order_units in outcomes:
        store.con.execute(
            """INSERT OR REPLACE INTO paper_order_results
            (proposal_id, order_index, fund_code, side, requested_value, outcome, reason, units, fill_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [proposal_id, order_idx, str(code), side, requested, outcome, reason, order_units, fill_d.date()],
        )
    for fill_idx_in_batch, f in enumerate(fills):
        store.con.execute(
            """
            INSERT INTO paper_fills (fill_id, proposal_id, fund_code, side, units, price, gross, fee, tax, realized_gain, fill_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                f"{proposal_id}-{int(now_ts.timestamp() * 1000)}-{fill_idx_in_batch}",
                proposal_id,
                f["code"],
                f["side"],
                float(f["units"]),
                float(f["price"]),
                float(f["gross"]),
                float(f["fee"]),
                float(f["tax"]),
                float(f["realized_gain"]),
                fill_d.date(),
            ],
        )

    paper._persist()
    paper.snapshot_equity(fill_d, portfolio_name="live")
    result_states = [x[4] for x in outcomes]
    total_fills = int(
        store.con.execute("SELECT count(*) FROM paper_fills WHERE proposal_id=?", [proposal_id]).fetchone()[0]
    )
    if not result_states or all(state == "filled" for state in result_states):
        proposal_status = "filled"
    elif total_fills > 0 or any(x[6] > 0 for x in outcomes):
        proposal_status = "partially_filled"
    elif any(state == "pending" for state in result_states):
        proposal_status = "pending"
    else:
        proposal_status = "rejected"
    store.con.execute("UPDATE paper_proposals SET status = ? WHERE proposal_id = ?", [proposal_status, proposal_id])

    return {
        "status": proposal_status,
        "proposal_id": proposal_id,
        "fill_date": str(fill_d.date()),
        "n_fills": len(fills),
        "n_orders": len(outcomes),
        "order_results": pd.DataFrame(
            outcomes, columns=["order_index", "fund_code", "side", "requested_value", "outcome", "reason", "units"]
        ),
        "fills": pd.DataFrame(fills),
    }


def paper_fill_auto(store, cfg: dict, decision_date: str | None = None) -> dict:
    """Bir karar gününün önerisini D+1 fiyatıyla otomatik doldur.

    decision_date verilmezse Europe/Istanbul'daki bugünden önceki configured BIST iş günü seçilir.
    """
    if decision_date is None:
        tz_name = cfg.get("project", {}).get("timezone", "Europe/Istanbul")
        try:
            tz = ZoneInfo(tz_name)
        except Exception:  # noqa: BLE001
            tz = ZoneInfo("Europe/Istanbul")
        now = pd.Timestamp(datetime.now(tz))
        if now.tzinfo is not None:
            now = now.tz_localize(None)
        d = now.normalize() - pd.Timedelta(days=1)
        while not is_business_day(d, cfg):
            d -= pd.Timedelta(days=1)
    else:
        d = pd.Timestamp(decision_date).normalize()

    proposal_id = f"{d:%Y%m%d}-01"
    proposal = _load_proposal(store, proposal_id)
    if proposal is None:
        return {"status": "no_proposal", "proposal_id": proposal_id, "message": "öneri yok"}
    if proposal["status"] != "proposed":
        return {"status": proposal["status"], "proposal_id": proposal_id, "message": "öneri zaten işlenmiş"}

    return paper_fill(store, cfg, proposal_id, pct=1.0, auto=True)
