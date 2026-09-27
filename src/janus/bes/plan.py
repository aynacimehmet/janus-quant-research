"""BES aylık plan: B2c-fdr seçimi + değişiklik sayacı + gölge rapor (S5-5).

- `janus bes plan --date D`: D günü için BES tahminlerinden seçim yapar.
- Yılda en fazla 12 değişiklik (10 planlı + 2 yedek); aynı ay tekrar koşu aynı change_id.
- Her değişiklikte ≤ 20 fon.
- Rapor: reports/bes_plan_<D>.md; emir/Telegram yok.
- B0 karşılaştırma: EMK PPF sepetinin 21/63/126g getirisi.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BES_PLAN_DDL = """
CREATE TABLE IF NOT EXISTS bes_plan (
    change_id VARCHAR,
    date DATE,
    year INTEGER,
    month INTEGER,
    fund_code VARCHAR,
    target_weight DOUBLE,
    p_value DOUBLE,
    lower DOUBLE,
    q50 DOUBLE,
    note VARCHAR
)
"""


def _ensure_bes_plan_table(store) -> None:
    store.con.execute(BES_PLAN_DDL)


def _next_change_id(store, year: int, planned: int, reserve: int) -> tuple[str, int] | None:
    """Yeni change_id ve yıl içindeki sıra numarasını döndürür; limit dolmuşsa None.

    F14: sıra `max(mevcut sıra) + 1`'dir; append-only geçmişte silinen/güncellenen satır
    varsayımı yapılmaz, boşluk olsa bile kimlik yeniden kullanılmaz.
    """
    legal_max = planned + reserve
    existing = store.con.execute("SELECT change_id FROM bes_plan WHERE year = ?", [year]).fetchall()
    seqs = []
    for (cid,) in existing:
        m = re.fullmatch(rf"{year}-(\d+)", str(cid))
        if m:
            seqs.append(int(m.group(1)))
    seq = (max(seqs) + 1) if seqs else 1
    if seq > legal_max:
        return None
    return f"{year}-{seq:02d}", seq


def _latest_change_id(store) -> str | None:
    """En son kayıtlı change_id (append-only geçmişte kronolojik son olay)."""
    row = store.con.execute(
        "SELECT change_id FROM bes_plan ORDER BY year DESC, change_id DESC, date DESC LIMIT 1"
    ).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def _change_targets(store, change_id: str) -> dict[str, float]:
    """Bir change_id'nin hedef fon→ağırlık vektörü (NULL fon satırları yok sayılır)."""
    rows = store.con.execute(
        "SELECT fund_code, target_weight FROM bes_plan WHERE change_id = ?", [change_id]
    ).fetchall()
    return {str(code): float(w) for code, w in rows if code is not None and w is not None}


def _targets_differ(new_w: dict[str, float], old_w: dict[str, float], tol: float = 1e-6) -> bool:
    keys = set(new_w) | set(old_w)
    return any(abs(float(new_w.get(k, 0.0)) - float(old_w.get(k, 0.0))) > tol for k in keys)


def _group_used(weights: dict[str, float], founders: dict[str, str], group: str) -> float:
    return sum(v for c, v in weights.items() if founders.get(c, "") == group)


def _waterfill(
    codes: list[str],
    founders: dict[str, str],
    max_fund: float,
    max_founder: float,
    budget: float,
    weights: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Eşit paylaşımı fon ve PYŞ tavanlarıyla kısıtla; artan bütçeyi döndür.

    Kurucu bilgisi eksik fon fail-closed dışarıda kalır; kota tükendiğinde pay eklenmez.
    """
    eps = 1e-12
    w = dict(weights)
    remaining = float(budget)
    active = list(dict.fromkeys(codes))
    for _ in range(1000):
        active = [c for c in active if founders.get(c, "") not in ("", None) and max_fund - w.get(c, 0.0) > eps]
        active = [c for c in active if max_founder - _group_used(w, founders, founders[c]) > eps]
        if not active or remaining <= eps:
            break
        share = remaining / len(active)
        group_n: dict[str, int] = {}
        for c in active:
            group_n[founders[c]] = group_n.get(founders[c], 0) + 1
        rooms = []
        for c in active:
            g = founders[c]
            room_g = max_founder - _group_used(w, founders, g)
            rooms.append(min(share, max_fund - w.get(c, 0.0), room_g / group_n[g]))
        step = min(rooms)
        if step <= eps:
            break
        for c in active:
            w[c] = w.get(c, 0.0) + step
        remaining -= step * len(active)
    return w, max(0.0, remaining)


def _allocate_targets(
    chosen: list[str],
    ppf_codes: list[str],
    founders: dict[str, str],
    cfg: dict[str, Any],
) -> tuple[dict[str, float], float]:
    """F15: seçilenlere eşit başlangıç, fon/PYŞ tavanı, artan EMK PPF sepetine, kalan serbest nakit."""
    constraints = cfg["legs"]["bes"]["constraints"]
    max_fund = float(constraints.get("max_weight_per_fund", 0.25))
    max_founder = float(constraints.get("max_weight_per_founder", 0.30))
    w, residual = _waterfill(chosen, founders, max_fund, max_founder, 1.0, {})
    if ppf_codes:
        w, residual = _waterfill(ppf_codes, founders, max_fund, max_founder, residual, w)
    return w, residual


def _b0_returns(
    nav: pd.DataFrame,
    codes: list[str],
    windows: tuple[int, ...] = (21, 63, 126),
) -> dict[str, float | None]:
    """EMK PPF sepetinin log getirileri (B0 karşılaştırması); sepet yoksa hata (F16)."""
    from janus.backtest.data import cash_proxy_returns

    if not codes:
        raise RuntimeError("BES nakit vekili yok: EMK para piyasası fonu sepeti bulunamadı")
    r = cash_proxy_returns(nav, codes)
    lr = np.log1p(r)
    return {f"b0_ret_{w}d": float(lr.tail(w).sum()) if len(lr) >= w else None for w in windows}


def _excluded_state_funds(fund_master: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Rapora yazılacak devlet katkısı dışlanan fon listesi."""
    from janus.portfolio.universe import tr_fold

    fm = fund_master.drop_duplicates("fund_code").copy()
    # EMK olanlar
    emk = fm[fm["fund_class"].fillna("").str.upper() == "EMK"].copy()
    if emk.empty:
        return emk
    pat = "|".join(
        re.escape(tr_fold(x)) for x in cfg["legs"]["bes"]["universe"].get("exclude_state_contribution_patterns", [])
    )
    if not pat:
        return emk.iloc[0:0]
    norm_name = emk["name"].fillna("").astype(str).map(tr_fold)
    norm_cat = emk["category"].fillna("").astype(str).map(tr_fold)
    mask = norm_name.str.contains(pat, regex=True) | norm_cat.str.contains(pat, regex=True)
    return emk.loc[mask, ["fund_code", "name", "category", "umbrella_type"]]


def bes_plan(
    store,
    cfg: dict[str, Any],
    date: str,
    root: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """BES aylık plan üret; `bes_plan` tablosuna yaz ve raporla.

    Çıktı:
      - data/predictions/bes_selection_<date>.parquet
      - reports/bes_plan_<date>.md
    """
    from janus.cli import _select_impl

    d = pd.Timestamp(date).normalize()
    _ensure_bes_plan_table(store)

    plan_cfg = cfg["legs"]["bes"]["plan"]
    year, month = d.year, d.month

    feat_path = root / "data" / "features" / "bes_features.parquet"
    pred_path = root / "data" / "predictions" / "bes_predictions.parquet"
    cal_path = root / "data" / "predictions" / "bes_calibrated_target_020.parquet"
    for p, name in ((feat_path, "bes_features"), (pred_path, "bes_predictions"), (cal_path, "bes_calibrated")):
        if not p.exists():
            return {"status": "error", "message": f"{name}.parquet yok — önce `janus bes build`"}

    preds = pd.read_parquet(pred_path)
    feats = pd.read_parquet(feat_path)
    cal = pd.read_parquet(cal_path)
    from janus.portfolio.universe import bes_scope_mask

    current_fm = store.latest_fund_master()
    scope = bes_scope_mask(current_fm, cfg)
    allowed_codes = set(scope.index[scope.to_numpy()].astype(str))
    preds = preds[preds["fund_code"].astype(str).isin(allowed_codes)]
    feats = feats[feats["fund_code"].astype(str).isin(allowed_codes)]
    cal = cal[cal["fund_code"].astype(str).isin(allowed_codes)]

    # D günü için BES tahmini var mı?
    if preds[pd.to_datetime(preds["decision_at"]) == d].empty:
        return {"status": "error", "message": f"{date} için BES tahmini yok"}

    sel_path = root / "data" / "predictions" / f"bes_selection_{date}.parquet"
    select_cfg = dict(cfg.get("conformal", {}))
    select_cfg["fdr_q_grid"] = [0.20]  # BES plan yalnızca kanonik q20
    select_summary = _select_impl(date, preds, feats, cal, sel_path, select_cfg)
    selected = pd.read_parquet(sel_path)
    selected_funds = selected[selected["selected_q20"].fillna(False)]["fund_code"].tolist()

    nav = store.nav_wide()
    fm = store.latest_fund_master()

    # F16: plan anında EMK para piyasası sepetini yeniden doğrula; yoksa güvenli hata (satır/rapor yazma).
    from janus.backtest.data import cash_proxy_codes

    ppf_codes = cash_proxy_codes(nav, fm, cfg, leg="bes")
    if not ppf_codes:
        return {
            "status": "error",
            "message": "BES nakit vekili yok: EMK para piyasası fonu sepeti bulunamadı; plan üretilmedi",
        }

    max_funds = int(cfg["legs"]["bes"]["constraints"]["max_funds_per_change"])
    chosen = selected_funds[:max_funds]
    meta = fm.drop_duplicates("fund_code").set_index("fund_code")
    # PYŞ kimliği kanonik `founder_code`; yoksa `founder` adı. Aynı kod farklı adlarla bölünmez.
    founder_col = (
        meta["founder_code"]
        if "founder_code" in meta.columns and meta["founder_code"].notna().any()
        else meta["founder"]
        if "founder" in meta.columns
        else pd.Series("", index=meta.index)
    )
    founders = {
        str(c): ("" if str(v).strip().lower() in ("", "none", "nan") else str(v).strip())
        for c, v in founder_col.fillna("").items()
    }
    weights, residual = _allocate_targets(chosen, ppf_codes, founders, cfg)

    # F14: yalnız gerçek fon dağılımı değişiminde (riskli pay oluştuğunda veya kayıtlı hedef
    # vektörden farklılaştığında) hak harca; ilk plan + boş seçim = varsayılan B0.
    ppf_set = set(ppf_codes)
    has_risky = any(code not in ppf_set for code in weights)
    latest_id = _latest_change_id(store)
    old_w = _change_targets(store, latest_id) if latest_id else {}
    real_change = _targets_differ(weights, old_w) if old_w else has_risky

    # Rapor her durumda yazılır (gölge; hak harcamaz).
    out = Path(out_dir) if out_dir else root / cfg.get("reporting", {}).get("orders_dir", "reports")
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"bes_plan_{date}.md"
    b0 = _b0_returns(nav, ppf_codes)
    excluded = _excluded_state_funds(fm, cfg)

    if not real_change:
        change_id = latest_id or "-"
        md = _format_plan_md(
            date,
            change_id,
            selected,
            weights,
            residual,
            b0,
            excluded,
            cfg["legs"]["bes"]["execution"].get("valor_assumption_id", "A8"),
            note="değişiklik yok — hak harcanmadı (B0 varsayılanı)",
        )
        md_path.write_text(md, encoding="utf-8")
        return {
            "status": "no_change",
            "date": date,
            "change_id": change_id,
            "n_selected": len(chosen),
            "n_available": int(select_summary["n_funds"]),
            "md_path": str(md_path),
        }

    nxt = _next_change_id(store, year, int(plan_cfg["planned"]), int(plan_cfg["reserve"]))
    if nxt is None:
        return {"status": "limit_reached", "message": f"{year} yılında değişiklik hakkı doldu"}
    change_id, _ = nxt

    # Append-only: mevcut satırlar silinmez/güncellenmez; yeni change_id satırları eklenir.
    rows = []
    for code, tw in weights.items():
        if code in ppf_set:
            rows.append([change_id, d.date(), year, month, code, float(tw), None, None, None, "B0 nakit"])
            continue
        row = selected[selected["fund_code"] == code]
        pv = float(row.iloc[0]["p_value"]) if not row.empty and pd.notna(row.iloc[0]["p_value"]) else None
        lo = float(row.iloc[0]["lower"]) if not row.empty and pd.notna(row.iloc[0]["lower"]) else None
        q5 = float(row.iloc[0]["q50"]) if not row.empty and pd.notna(row.iloc[0]["q50"]) else None
        rows.append([change_id, d.date(), year, month, code, float(tw), pv, lo, q5, "seçim"])
    store.con.executemany(
        "INSERT INTO bes_plan (change_id, date, year, month, fund_code, target_weight, p_value, lower, q50, note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    md = _format_plan_md(
        date,
        change_id,
        selected,
        weights,
        residual,
        b0,
        excluded,
        cfg["legs"]["bes"]["execution"].get("valor_assumption_id", "A8"),
        note="gerçek değişiklik",
    )
    md_path.write_text(md, encoding="utf-8")

    return {
        "status": "ok",
        "date": date,
        "change_id": change_id,
        "n_selected": len(chosen),
        "n_available": int(select_summary["n_funds"]),
        "md_path": str(md_path),
    }


def _format_plan_md(
    date: str,
    change_id: str,
    selected: pd.DataFrame,
    weights: dict[str, float],
    residual: float,
    b0: dict[str, float | None],
    excluded: pd.DataFrame,
    valor_assumption: str,
    note: str = "",
) -> str:
    lines = [
        f"# BES Plan — {date}",
        f"**change_id:** {change_id}  ",
        f"**durum:** {note}  ",
        "**PIT notu:** Bu gölge plan D-anı PIT meta/snapshot sepet kanıtı değildir (açık soru).  ",
        f"**valör varsayımı:** {valor_assumption} (alış T+1 / satış T+2)  ",
        f"**hedef fon sayısı:** {len(weights)}  ",
        "",
        "| fon | hedef % | p_value | lower | q50 |",
        "|-----|---------|---------|-------|-----|",
    ]
    for code, tw in weights.items():
        row = selected[selected["fund_code"] == code]
        pv = f"{row.iloc[0]['p_value']:.3f}" if not row.empty and pd.notna(row.iloc[0]["p_value"]) else "-"
        lo = f"{row.iloc[0]['lower']:.4f}" if not row.empty and pd.notna(row.iloc[0]["lower"]) else "-"
        q5 = f"{row.iloc[0]['q50']:.4f}" if not row.empty and pd.notna(row.iloc[0]["q50"]) else "-"
        lines.append(f"| {code} | {100.0 * tw:.1f} | {pv} | {lo} | {q5} |")
    if not weights:
        lines.append("| — | 0.0 | - | - | - |")
    lines.extend(["", f"**serbest nakit (getiri atfedilmedi):** %{100.0 * residual:.2f}", ""])
    lines.extend(["## B0 (EMK PPF sepeti) karşılaştırma", ""])
    for k in sorted(b0):
        w = k.split("_")[-1]
        v = b0[k]
        lines.append(f"- {w} log getiri: {v * 100:.2f}%" if v is not None else f"- {w}: hesaplanamadı")
    if not excluded.empty:
        lines.extend(["", "## Dışlanan devlet katkısı fonları", ""])
        lines.extend(f"- `{r.fund_code}` — {r.name} ({r.category})" for _, r in excluded.iterrows())
    else:
        lines.extend(["", "## Dışlanan devlet katkısı fonları", "Yok."])
    lines.extend(["", "_Bu rapor gölgedir; emir/Telegram üretilmez._"])
    return "\n".join(lines)
