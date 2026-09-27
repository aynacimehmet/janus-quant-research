"""S5-3: haftalık gölge raporu ve ADR-20 sayacı."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from janus.paper.kpi import compute_kpis


def _load_equity(store, portfolio_name: str) -> pd.Series:
    df = store.con.execute(
        "SELECT date, equity FROM paper_equity WHERE portfolio_name = ? ORDER BY date",
        [portfolio_name],
    ).df()
    if df.empty:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["equity"]


def compute_b0_tracking(
    actual_equity: pd.Series,
    proxy_equity: pd.Series,
    *,
    epoch_start: pd.Timestamp | str | None,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> dict:
    """Same-date simple return comparison; never fills or estimates missing observations."""
    if epoch_start is None or pd.isna(epoch_start):
        return {"status": "değerlendirilemedi", "reason": "pilot epoch/init tarihi yok"}
    lo, hi = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    epoch = pd.Timestamp(epoch_start).normalize()
    if lo < epoch <= hi:
        return {"status": "değerlendirilemedi", "reason": "hafta reset/init sınırını kesiyor"}

    common = actual_equity.index.intersection(proxy_equity.index)
    common = common[(common >= max(lo, epoch)) & (common <= hi)]
    actual = pd.to_numeric(actual_equity.reindex(common), errors="coerce")
    proxy = pd.to_numeric(proxy_equity.reindex(common), errors="coerce")
    if len(common) < 2:
        return {"status": "değerlendirilemedi", "reason": "iki ortak geçerli tarih yok", "n_common": len(common)}
    valid = actual.notna() & proxy.notna() & np.isfinite(actual) & np.isfinite(proxy) & actual.gt(0) & proxy.gt(0)
    if not valid.all():
        return {
            "status": "değerlendirilemedi",
            "reason": "ortak tarihlerde eksik/geçersiz equity",
            "n_common": len(common),
        }
    proxy_returns = proxy.pct_change(fill_method=None).iloc[1:]
    if proxy_returns.isna().any() or not np.isfinite(proxy_returns.to_numpy(float)).all():
        return {
            "status": "değerlendirilemedi",
            "reason": "ortak tarihlerde proxy getirisi eksik",
            "n_common": len(actual),
        }
    real_return = float(actual.iloc[-1] / actual.iloc[0] - 1.0)
    proxy_return = float((1.0 + proxy_returns).prod() - 1.0)
    difference_pp = 100.0 * (real_return - proxy_return)
    return {
        "status": "değerlendirildi",
        "n_common": len(actual),
        "real_return": real_return,
        "proxy_return": proxy_return,
        "difference_pp": difference_pp,
        "gate_met": abs(difference_pp) <= 0.5,
        "first_date": str(actual.index[0].date()),
        "last_date": str(actual.index[-1].date()),
    }


def _load_real_b0_equity(store, end: pd.Timestamp) -> tuple[pd.Series, pd.Timestamp | None, str | None]:
    """Use live cash/receivables only when the entire epoch is demonstrably B0-only."""
    exists = store.con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='janus_pilot_epoch'"
    ).fetchone()[0]
    if not exists:
        return pd.Series(dtype=float), None, "pilot epoch kaydı yok"
    epoch_row = store.con.execute(
        "SELECT coalesce(initialized_at, started_at) FROM janus_pilot_epoch ORDER BY epoch_id DESC LIMIT 1"
    ).fetchone()
    if not epoch_row or epoch_row[0] is None:
        return pd.Series(dtype=float), None, "pilot init tarihi yok"
    epoch_start = pd.Timestamp(epoch_row[0]).normalize()
    snapshots = store.con.execute(
        "SELECT date, cash, receivables, risky_value, slot_value FROM paper_equity "
        "WHERE portfolio_name='live' AND date >= ? AND date <= ? ORDER BY date",
        [epoch_start.date(), end.date()],
    ).df()
    if snapshots.empty:
        return pd.Series(dtype=float), epoch_start, "live B0 equity snapshot yok"
    fields = ["cash", "receivables", "risky_value", "slot_value"]
    values = snapshots[fields].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(float)).all():
        return pd.Series(dtype=float), epoch_start, "B0 equity bileşenleri eksik"
    if values["risky_value"].abs().gt(1e-9).any():
        return pd.Series(dtype=float), epoch_start, "risky pozisyon varken ortak nakit B0'a ayrılamıyor"

    memberships = store.con.execute(
        "SELECT membership_date, fund_code FROM paper_b0_memberships WHERE membership_date <= ?",
        [end.date()],
    ).df()
    fills = store.con.execute(
        "SELECT fund_code, fill_date FROM paper_fills WHERE fill_date >= ? AND fill_date <= ?",
        [epoch_start.date(), end.date()],
    ).df()
    if not fills.empty:
        if memberships.empty:
            return pd.Series(dtype=float), epoch_start, "fill'ler için B0 üyelik kanıtı yok"
        memberships["membership_date"] = pd.to_datetime(memberships["membership_date"]).dt.normalize()
        fills["fill_date"] = pd.to_datetime(fills["fill_date"]).dt.normalize()
        for fill in fills.itertuples(index=False):
            eligible = memberships.loc[memberships["membership_date"] <= fill.fill_date, "fund_code"].astype(str)
            if str(fill.fund_code) not in set(eligible):
                return pd.Series(dtype=float), epoch_start, "risky fill nakit/alacak atfını belirsiz kılıyor"

    snapshots["date"] = pd.to_datetime(snapshots["date"]).dt.normalize()
    real_equity = values["slot_value"] + values["cash"] + values["receivables"]
    real_equity.index = pd.DatetimeIndex(snapshots["date"])
    return real_equity, epoch_start, None


def _load_live_proposals(store) -> pd.DataFrame:
    df = store.con.execute(
        "SELECT proposal_id, date, status, orders_json, evidence_count, evidence_codes "
        "FROM paper_proposals ORDER BY date"
    ).df()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df["n_orders"] = df["orders_json"].apply(lambda x: len(json.loads(x)) if x else 0)
    if "evidence_count" not in df:
        df["evidence_count"] = 0
    if "evidence_codes" not in df:
        df["evidence_codes"] = "[]"
    df["evidence_codes"] = df["evidence_codes"].fillna("[]").apply(json.loads)
    df["has_proof"] = pd.to_numeric(df["evidence_count"], errors="coerce").fillna(0).gt(0)
    return df


def adr20_status(
    store,
    features: pd.DataFrame | None = None,
    asof: pd.Timestamp | str | None = None,
) -> dict:
    """Bitişik BH kanıt günlerini bölümlendir; yalnız seçilmiş ve olgun y'leri say."""
    proposals = _load_live_proposals(store)
    if proposals.empty:
        return {"n_episodes": 0, "n_matured_episodes": 0, "matured_positive_rate": float("nan"), "adr21_ready": False}
    cutoff = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.now().normalize()
    proposals = proposals[proposals["date"] <= cutoff]
    proof = proposals[proposals["has_proof"]].sort_values("date")
    if proof.empty or features is None or features.empty:
        return {
            "n_episodes": 0,
            "n_matured_episodes": 0,
            "matured_positive_rate": float("nan"),
            "adr21_ready": False,
        }
    work = features.copy()
    work["decision_at"] = pd.to_datetime(work["decision_at"]).dt.normalize()
    work["label_available_at"] = pd.to_datetime(work["label_available_at"]).dt.normalize()
    work = work[work["label_available_at"] <= cutoff]
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(features["decision_at"]).dt.normalize().unique()))
    idx_by_day = {day: i for i, day in enumerate(calendar)}
    proof_rows = list(proof.itertuples(index=False))
    episodes: list[list[object]] = []
    for proposal in proof_rows:
        day = pd.Timestamp(proposal.date).normalize()
        if (
            episodes
            and idx_by_day.get(day, -2) == idx_by_day.get(pd.Timestamp(episodes[-1][-1].date).normalize(), -3) + 1
        ):
            episodes[-1].append(proposal)
        else:
            episodes.append([proposal])

    episode_rates: list[float] = []
    for episode in episodes:
        labels: list[float] = []
        for proposal in episode:
            day = pd.Timestamp(proposal.date).normalize()
            codes = set(proposal.evidence_codes or [])
            rows = work.loc[(work["decision_at"] == day) & work["fund_code"].astype(str).isin(codes), "y"].dropna()
            labels.extend(pd.to_numeric(rows, errors="coerce").dropna().astype(float).tolist())
        if labels:
            episode_rates.append(float(np.mean(np.asarray(labels) > 0)))
    n_matured = len(episode_rates)
    positive_rate = float(np.mean(episode_rates)) if episode_rates else float("nan")
    return {
        "n_episodes": len(episodes),
        "n_matured_episodes": n_matured,
        "matured_positive_rate": positive_rate,
        "adr21_ready": n_matured >= 3 and positive_rate >= 2.0 / 3.0,
    }


def paper_weekly(store, cfg: dict, date: str | None = None, out_dir: Path | None = None) -> dict:
    """Haftalık gölge raporu ve ADR-20 durumu."""
    d = pd.Timestamp(date).normalize() if date else pd.Timestamp.now().normalize()
    start = d - pd.Timedelta(days=7)

    b0 = _load_equity(store, "B0_cash")
    actual_b0, epoch_start, b0_reason = _load_real_b0_equity(store, d)
    tracking = (
        {"status": "değerlendirilemedi", "reason": b0_reason}
        if b0_reason
        else compute_b0_tracking(actual_b0, b0, epoch_start=epoch_start, start=start, end=d)
    )
    shadows = ["B2b_momentum_hrp", "B2c_m_conformal_hrp", "B2c_fdr_q20", "B2c_fdr_q10", "B2c_fdr_q30"]
    rows = []
    for name in shadows:
        eq = _load_equity(store, name)
        if eq.empty or b0.empty:
            continue
        common = eq.index.intersection(b0.index)
        common = common[common >= start]
        if len(common) < 2:
            continue
        ret = np.log(eq.loc[common].iloc[-1] / eq.loc[common].iloc[0])
        b0_ret = np.log(b0.loc[common].iloc[-1] / b0.loc[common].iloc[0])
        rows.append(
            {
                "portfolio": name,
                "week_return_pp": round(ret * 100, 2),
                "b0_excess_pp": round((ret - b0_ret) * 100, 2),
                "latest_equity": round(float(eq.iloc[-1]), 4),
            }
        )

    proposals = _load_live_proposals(store)
    proof_days = int(proposals["has_proof"].sum()) if not proposals.empty else 0

    features_path = Path("data") / "features" / "fund_features.parquet"
    features = pd.read_parquet(features_path) if features_path.exists() else None
    adr = adr20_status(store, features, asof=d)

    kpis = compute_kpis(store, cfg)
    gate_4w = kpis.get("gate_met")
    gate_text = "sağlandı" if gate_4w else ("sağlanmadı" if gate_4w is False else "veri yok")

    lines = [
        f"# Kağıt haftalık — {d.date()}",
        f"**kanıt günleri (canlı):** {proof_days}",
        f"**ADR-20:** {adr['n_episodes']} bölüm ({adr['n_matured_episodes']} olgun), "
        f"olgunlaşan pozitiflik %{adr.get('matured_positive_rate', 0) * 100:.1f}",
        f"**4 haftalık operasyon kapısı:** {gate_text}",
        "",
        "## B0 gerçek sepet ↔ vekil",
    ]
    if tracking["status"] == "değerlendirildi":
        lines.append("| ortak tarih aralığı | R_real % | R_proxy % | fark pp | ±0,5 pp kapısı |")
        lines.append("|---|---:|---:|---:|---|")
        lines.append(
            f"| {tracking['first_date']}–{tracking['last_date']} | "
            f"{tracking['real_return'] * 100:.2f} | {tracking['proxy_return'] * 100:.2f} | "
            f"{tracking['difference_pp']:+.2f} | {'sağlandı' if tracking['gate_met'] else 'sağlanmadı'} |"
        )
    else:
        lines.append(f"**değerlendirilemedi:** {tracking.get('reason', 'ortak B0 verisi yetersiz')}")
    lines.extend(
        [
            "",
            "## Gölge portföyler",
            "| portföy | haftalık getiri % | B0 fazlası pp | son equity |",
            "|---------|-------------------|---------------|------------|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['portfolio']} | {row['week_return_pp']:.2f} | {row['b0_excess_pp']:+.2f} | {row['latest_equity']:.2f} |"
        )

    warning = ""
    if adr["adr21_ready"]:
        warning = "\n\n**ADR-21 gündemi:** ≥3 bölümde ≥2/3 olgunlaşan seçim pozitif → gerçek sermaye kararı değerlendirilebilir."
        lines.append(warning)

    out = Path(out_dir) if out_dir else Path(cfg.get("reporting", {}).get("orders_dir", "reports"))
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"paper_weekly_{d:%Y-%m-%d}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    telegram = f"<b>JANUS haftalık — {d.date()}</b>\nkanıt günleri: {proof_days}\nADR-20: {adr['n_episodes']} bölüm, pozitiflik %{adr.get('matured_positive_rate', 0) * 100:.1f}\n4-haftalık kapı: {gate_text}"
    if tracking["status"] == "değerlendirildi":
        telegram += f"\nB0 gerçek/vekil farkı: {tracking['difference_pp']:+.2f}pp"
    else:
        telegram += "\nB0 gerçek/vekil farkı: değerlendirilemedi"
    if adr["adr21_ready"]:
        telegram += "\n<b>ADR-21 gündemi</b>"
    for row in rows:
        telegram += (
            f"\n• {html.escape(row['portfolio'])}: %{row['week_return_pp']:.2f} (B0 +{row['b0_excess_pp']:+.2f}pp)"
        )

    return {
        "date": str(d.date()),
        "rows": rows,
        "proof_days": proof_days,
        "adr20": adr,
        "b0_tracking": tracking,
        "md_path": str(md_path),
        "telegram": telegram,
    }
