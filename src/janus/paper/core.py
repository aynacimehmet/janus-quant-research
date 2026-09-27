"""Kağıt-ticaret defteri ve öneri listesi (S5-1).

DuckDB paper_* tabloları yalnızca `janus paper` komutlarından yazılır.
Motor çekirdeği `janus.backtest.ledger.Ledger` ile aynıdır.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

from janus.backtest.costs import FundMeta, build_fund_meta
from janus.backtest.data import _execution_profiles_asof, cash_proxy_codes, cash_proxy_returns
from janus.backtest.engine import _with_cash_slot
from janus.backtest.ledger import Ledger, Lot
from janus.data.quality import (
    execution_status_masks,
    expected_last_nav_date,
    flag_data_stale,
    next_business_day,
    quality_summary,
)
from janus.strategies.hrp import (
    apply_caps,
    cluster_labels,
    constrain_founder_targets,
    hrp_weights,
    ledoit_wolf_cov,
    select_with_limits,
)

SLOT_CODE = "CASH_PROXY"
_PIT_PROFILE_START = pd.Timestamp("2026-09-22")


def _profile_execution_reasons(
    profiles: pd.DataFrame,
    asof: pd.Timestamp | str,
    morning_cutoff: str = "09:15",
    codes: list[str] | tuple[str, ...] = (),
) -> dict[str, dict[str, str]]:
    """Validate the latest profile available by cutoff; a selected stale profile never falls back."""
    decision = pd.Timestamp(asof).tz_localize(None).normalize()
    if decision < _PIT_PROFILE_START:
        return {}
    known_codes = set(map(str, codes))
    if profiles.empty or "fund_code" not in profiles or "snapshot_date" not in profiles:
        return {code: {"BUY": "no_pit_snapshot", "SELL": "no_pit_snapshot"} for code in known_codes}
    hour, minute = (int(part) for part in morning_cutoff.split(":"))
    cutoff = decision + pd.Timedelta(hours=hour, minutes=minute)
    known_codes.update(profiles["fund_code"].astype(str))
    available = _execution_profiles_asof(profiles, decision, morning_cutoff)
    latest = available.drop_duplicates("fund_code", keep="last").set_index("fund_code")
    status_buy, status_sell, status_known, status_mismatch = execution_status_masks(available)
    mismatch_count = int(status_mismatch.sum())
    if mismatch_count:
        logger.warning("paper execution status metni ile kayıtlı bayrak uyuşmuyor: {} profil", mismatch_count)
    source_codes = available.reset_index(drop=True)["fund_code"].astype(str)
    status_buy_by_code = pd.Series(status_buy.to_numpy(bool), index=source_codes)
    status_sell_by_code = pd.Series(status_sell.to_numpy(bool), index=source_codes)
    status_known_by_code = pd.Series(status_known.to_numpy(bool), index=source_codes)
    reason_map: dict[str, dict[str, str]] = {}
    for code in sorted(known_codes):
        if code not in latest.index:
            reason_map[code] = {"BUY": "no_pit_snapshot", "SELL": "no_pit_snapshot"}
            continue
        row = latest.loc[code]
        reason = ""
        if code not in status_known_by_code.index or not bool(status_known_by_code.loc[code]):
            reason = "unknown_tefas_status"
        success = pd.to_datetime(row.get("last_success_at"), errors="coerce")
        if not reason and pd.isna(success):
            reason = reason or "missing_last_success_at"
        elif not reason:
            success = pd.Timestamp(success)
            if success.tzinfo is not None:
                success = success.tz_localize(None)
            if success > cutoff:
                reason = reason or "future_last_success_at"
            elif cutoff - success > pd.Timedelta(days=7):
                reason = reason or "stale_last_success_at"

        required = ("buy_valor", "sell_valor", "tax_category")
        if not reason:
            missing = [field for field in required if pd.isna(row.get(field))]
            if missing:
                reason = "missing_execution_field:" + ",".join(missing)
        if not reason:
            for field in ("buy_valor", "sell_valor"):
                value = pd.to_numeric(pd.Series([row.get(field)]), errors="coerce").iloc[0]
                if pd.isna(value) or not np.isfinite(value) or value < 0 or value != np.floor(value):
                    reason = f"invalid_{field}"
                    break
        if not reason and not str(row.get("tax_category", "")).strip():
            reason = "missing_execution_field:tax_category"
        if reason:
            reason_map[code] = {"BUY": reason, "SELL": reason}
        else:
            sides = {}
            if not bool(status_buy_by_code.loc[code]):
                sides["BUY"] = "can_buy=False"
            if not bool(status_sell_by_code.loc[code]):
                sides["SELL"] = "can_sell=False"
            if sides:
                reason_map[code] = sides
    return reason_map


def _publication_assumptions(
    profiles: pd.DataFrame,
    asof: pd.Timestamp | str,
    morning_cutoff: str = "09:15",
) -> dict[str, str]:
    """Label date-only post-PIT snapshots rather than silently treating them as timestamped evidence."""
    decision = pd.Timestamp(asof).tz_localize(None).normalize()
    if decision < _PIT_PROFILE_START or profiles.empty or not {"fund_code", "snapshot_date"}.issubset(profiles.columns):
        return {}
    latest = _execution_profiles_asof(profiles, decision, morning_cutoff)
    latest = latest.drop_duplicates("fund_code", keep="last").set_index("fund_code")
    published_raw = (
        latest["source_published_at"] if "source_published_at" in latest else pd.Series(pd.NaT, index=latest.index)
    )
    published = pd.to_datetime(published_raw, errors="coerce")
    return {
        str(code): "date_only_pit_assumption" if pd.isna(published.loc[code]) else "source_timestamp"
        for code in latest.index
    }


def _b0_profile_error() -> str:
    return (
        "B0 canlı/PIT sepetinde seçilebilir fon yok: Para Piyasası filtresine ek olarak founder_code, "
        "info_ok profili ve buy_valor/sell_valor alanları zorunludur; valörler 2026-09-22 öncesine taşınmaz"
    )


PAPER_DDL = {
    "paper_positions": """
        CREATE TABLE IF NOT EXISTS paper_positions (
            updated_at TIMESTAMP, fund_code VARCHAR, units DOUBLE, cost_basis DOUBLE
        )""",
    "paper_b0_memberships": """
        CREATE TABLE IF NOT EXISTS paper_b0_memberships (
            membership_date DATE, snapshot_date DATE, fund_code VARCHAR, founder_code VARCHAR,
            buy_valor INTEGER, sell_valor INTEGER,
            PRIMARY KEY (membership_date, fund_code)
        )""",
    "paper_lots": """
        CREATE TABLE IF NOT EXISTS paper_lots (
            updated_at TIMESTAMP, fund_code VARCHAR, units DOUBLE, cost_per_unit DOUBLE,
            bought_date DATE, tax_rate DOUBLE, available_date DATE,
            bought_idx BIGINT, available_idx BIGINT
        )""",
    "paper_receivables": """
        CREATE TABLE IF NOT EXISTS paper_receivables (
            settle_idx BIGINT PRIMARY KEY, amount DOUBLE NOT NULL
        )""",
    "paper_order_results": """
        CREATE TABLE IF NOT EXISTS paper_order_results (
            proposal_id VARCHAR, order_index INTEGER, fund_code VARCHAR, side VARCHAR,
            requested_value DOUBLE, outcome VARCHAR, reason VARCHAR, units DOUBLE,
            fill_date DATE, PRIMARY KEY (proposal_id, order_index)
        )""",
    "paper_cash": """
        CREATE TABLE IF NOT EXISTS paper_cash (
            updated_at TIMESTAMP, cash DOUBLE, initial_capital DOUBLE
        )""",
    "paper_proposals": """
        CREATE TABLE IF NOT EXISTS paper_proposals (
            proposal_id VARCHAR, date DATE, created_at TIMESTAMP, expires_at TIMESTAMP,
            status VARCHAR, orders_json VARCHAR, evidence_count INTEGER DEFAULT 0,
            evidence_codes VARCHAR DEFAULT '[]', target_weights_json VARCHAR DEFAULT '{}',
            execution_blocks_json VARCHAR DEFAULT '[]'
        )""",
    "paper_fills": """
        CREATE TABLE IF NOT EXISTS paper_fills (
            fill_id VARCHAR, proposal_id VARCHAR, fund_code VARCHAR, side VARCHAR,
            units DOUBLE, price DOUBLE, gross DOUBLE, fee DOUBLE, tax DOUBLE,
            realized_gain DOUBLE, fill_date DATE
        )""",
    "paper_equity": """
        CREATE TABLE IF NOT EXISTS paper_equity (
            portfolio_name VARCHAR, date DATE, equity DOUBLE, cash DOUBLE, receivables DOUBLE,
            risky_value DOUBLE, slot_value DOUBLE, nav_asof DATE,
            PRIMARY KEY (portfolio_name, date)
        )""",
    "paper_shadow_runs": """
        CREATE TABLE IF NOT EXISTS paper_shadow_runs (
            portfolio_name VARCHAR, date DATE, status VARCHAR, error VARCHAR,
            n_rows BIGINT, nav_asof DATE, PRIMARY KEY (portfolio_name, date)
        )""",
    "paper_reconcile": """
        CREATE TABLE IF NOT EXISTS paper_reconcile (
            date DATE, equity DOUBLE, cash DOUBLE, receivables DOUBLE,
            risky_value DOUBLE, slot_value DOUBLE, identity_ok BOOLEAN,
            liquidation_value DOUBLE, max_weight_diff DOUBLE,
            n_expired INTEGER, PRIMARY KEY (date)
        )""",
}


def _ensure_paper_tables(store) -> None:
    for ddl in PAPER_DDL.values():
        store.con.execute(ddl)
    migrations = {
        "paper_lots": ("bought_idx BIGINT", "available_idx BIGINT"),
        "paper_proposals": (
            "evidence_count INTEGER DEFAULT 0",
            "evidence_codes VARCHAR DEFAULT '[]'",
            "target_weights_json VARCHAR DEFAULT '{}'",
            "execution_blocks_json VARCHAR DEFAULT '[]'",
        ),
    }
    for table, columns in migrations.items():
        for column in columns:
            store.con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column}")
    # Migration: S5-1 öncesi paper_equity şeması portfolio_name içermez
    cols = store.con.execute("PRAGMA table_info('paper_equity')").df()
    if not cols.empty and "portfolio_name" not in cols["name"].tolist():
        raise ValueError("Eski paper_equity şeması veri kaybı riski taşıyor; önce `janus paper reset --archive-v1`")


def _fmt_proposal_id(d: pd.Timestamp, seq: int = 1) -> str:
    return f"{d:%Y%m%d}-{seq:02d}"


@dataclass
class PaperLedger:
    """Kağıt defter: Ledger çekirdeğini DuckDB paper_* tablolarına bağlar."""

    store: object  # Store
    cfg: dict
    capital: float = 100.0
    asof: pd.Timestamp | None = None
    meta: FundMeta = field(init=False)
    ledger: Ledger = field(init=False)
    _b0_codes: tuple[str, ...] = field(init=False, default=())
    _b0_history_codes: tuple[str, ...] = field(init=False, default=())
    _execution_blocks: dict[str, dict[str, str]] = field(init=False, default_factory=dict)
    _publication_assumptions: dict[str, str] = field(init=False, default_factory=dict)
    _freshly_initialized: bool = field(init=False, default=False)
    meta_asof: pd.Timestamp = field(init=False)
    nav_asof: pd.Timestamp = field(init=False)

    def __post_init__(self) -> None:
        _ensure_paper_tables(self.store)
        tz_name = self.cfg.get("project", {}).get("timezone", "Europe/Istanbul")
        self.asof = (
            pd.Timestamp(datetime.now(ZoneInfo(tz_name)).date())
            if self.asof is None
            else pd.Timestamp(self.asof).normalize()
        )
        self._build_meta()
        self._load_or_init()

    def _build_meta(self) -> None:
        cutoff = pd.Timestamp(self.asof).normalize()
        master_rows = self.store.con.execute("SELECT * FROM fund_master").df()
        known_master_codes = set(master_rows.get("fund_code", pd.Series(dtype=str)).astype(str))
        nav = self.store.nav_wide().loc[lambda x: x.index <= cutoff]
        if nav.empty:
            raise ValueError(f"as-of {cutoff.date()} için kullanılabilir NAV yok")
        self.nav_asof = pd.Timestamp(nav.index[-1]).normalize()
        morning_cutoff = self.cfg.get("project", {}).get("runs", {}).get("morning", "09:15")
        self._execution_blocks = _profile_execution_reasons(
            master_rows, cutoff, morning_cutoff, codes=[str(code) for code in nav.columns]
        )
        self._publication_assumptions = _publication_assumptions(master_rows, cutoff, morning_cutoff)
        master_rows = _execution_profiles_asof(master_rows, cutoff, morning_cutoff)
        if master_rows.empty:
            raise ValueError(f"as-of {cutoff.date()} için PIT fund_master snapshot yok")
        fm = master_rows.copy()
        self.meta_asof = pd.Timestamp(fm["snapshot_date"].max()).normalize()
        codes = [str(c) for c in nav.columns]
        # NAV remains available for marking existing positions even when today's execution profile is absent.
        profile_rows = fm.drop_duplicates("fund_code").set_index("fund_code").reindex(codes)
        profile_rows.index.name = "fund_code"
        fm = profile_rows.reset_index()
        selection_fm = fm.copy()
        nav_counts = nav.notna().sum()
        selection_fm["n_nav"] = selection_fm["fund_code"].astype(str).map(nav_counts).fillna(0).astype(int)
        for code, sides in self._execution_blocks.items():
            mask = selection_fm["fund_code"].astype(str).eq(code)
            if "BUY" in sides:
                selection_fm.loc[mask, "can_buy"] = False
            if "SELL" in sides:
                selection_fm.loc[mask, "can_sell"] = False
        cash_codes = cash_proxy_codes(
            nav,
            selection_fm,
            self.cfg,
            require_execution_profile=True,
            profile_asof=self.asof,
        )
        cash_returns = cash_proxy_returns(nav, cash_codes)
        cash_meta = fm.drop_duplicates("fund_code").set_index("fund_code").reindex(cash_codes)
        cats = cash_meta.get("tax_category", pd.Series(dtype=object)).dropna()
        rates = pd.to_numeric(cash_meta.get("withholding_rate", pd.Series(dtype=float)), errors="coerce")
        prepared = {
            "nav": nav,
            "fund_master": fm,
            "cash_codes": cash_codes,
            "cash_returns": cash_returns,
            "cash_category": str(cats.mode().iloc[0]) if not cats.empty else None,
            "cash_rate": float(rates.mean()) if rates.notna().any() else 0.175,
        }
        nav = prepared["nav"]
        self._nav = nav
        fm = prepared["fund_master"]
        cash_codes = prepared["cash_codes"]
        meta = build_fund_meta(fm, codes, self.cfg)
        can_buy = meta.can_buy.copy()
        can_sell = meta.can_sell.copy()
        for i, code in enumerate(codes):
            sides = self._execution_blocks.get(str(code), {})
            if "BUY" in sides:
                can_buy[i] = False
            else:
                can_buy[i] = bool(can_buy[i])
            if "SELL" in sides:
                can_sell[i] = False
            else:
                can_sell[i] = bool(can_sell[i])
        object.__setattr__(meta, "can_buy", can_buy)
        object.__setattr__(meta, "can_sell", can_sell)
        cash_returns = cash_proxy_returns(nav, cash_codes)
        self._cash_nav = (1 + cash_returns).cumprod()
        self._cash_category = prepared.get("cash_category")
        self._cash_rate = prepared.get("cash_rate", 0.175)
        self._b0_codes = tuple(str(code) for code in cash_codes)
        self.meta = _with_cash_slot(meta, self._cash_category, self._cash_rate)
        asof_date = pd.Timestamp(self.asof).date() if self.asof is not None else None
        membership_query = "SELECT DISTINCT fund_code FROM paper_b0_memberships"
        membership_params: list[object] = []
        if asof_date is not None:
            membership_query += " WHERE membership_date <= ? AND snapshot_date <= ?"
            membership_params.extend([asof_date, asof_date])
        membership_rows = self.store.con.execute(membership_query, membership_params).fetchall()
        self._b0_history_codes = tuple(sorted({str(row[0]) for row in membership_rows} | set(self._b0_codes)))
        master_codes = known_master_codes
        missing_master = set(self._b0_history_codes).difference(master_codes)
        if missing_master:
            missing = ", ".join(sorted(missing_master))
            raise ValueError(f"Kayıtlı B0 fonu {missing} artık PIT fund_master içinde yok")
        no_nav_members = [
            code for code in self._b0_history_codes if code not in self._nav or not self._nav[code].notna().any()
        ]
        if no_nav_members:
            missing = ", ".join(no_nav_members)
            raise ValueError(f"Kayıtlı B0 fonu {missing} için as-of öncesi erişilebilir NAV yok")
        missing_members = set(self._b0_history_codes).difference(meta.index)
        if missing_members:
            missing = ", ".join(sorted(missing_members))
            raise ValueError(f"Kayıtlı B0 fonu {missing} artık PIT fund_master/meta evreninde yok")
        self._slot_i = len(self.meta.codes) - 1
        self._fund_master = fm

    def _nav_dates(self) -> np.ndarray:
        nav = self._nav
        if nav.empty:
            return np.array([], dtype="datetime64[ns]")
        return nav.index.to_numpy()

    def _date_idx(self, d) -> int | None:
        dates = pd.DatetimeIndex(self.ledger.dates)
        if d is None or pd.isna(d):
            return None
        ts = pd.Timestamp(d)
        if ts in dates:
            return int(dates.get_loc(ts))
        return None

    def _load_or_init(self) -> None:
        if self.asof is not None:
            latest_state = self.store.con.execute(
                "SELECT max(date) FROM paper_equity WHERE portfolio_name='live'"
            ).fetchone()[0]
            if latest_state is not None and pd.Timestamp(latest_state).date() > pd.Timestamp(self.asof).date():
                raise ValueError(
                    f"as-of {pd.Timestamp(self.asof).date()} öncesi defter durumunu yeniden kuracak olay kaydı yok"
                )
        cash_df = self.store.con.execute("SELECT * FROM paper_cash ORDER BY updated_at DESC LIMIT 1").df()
        legacy_proxy_positions = int(
            self.store.con.execute(
                "SELECT count(*) FROM paper_positions WHERE fund_code=? AND coalesce(units, 0) > 1e-12", [SLOT_CODE]
            ).fetchone()[0]
        )
        legacy_proxy_lots = int(
            self.store.con.execute(
                "SELECT count(*) FROM paper_lots WHERE fund_code=? AND coalesce(units, 0) > 1e-12", [SLOT_CODE]
            ).fetchone()[0]
        )
        if legacy_proxy_positions or legacy_proxy_lots:
            raise ValueError(
                "Eski CASH_PROXY bakiye/lot bulundu; gerçek B0 fonlarına otomatik dönüşüm güvenli değil. "
                "Açık kullanıcı kararı gerekir; defter değiştirilmedi."
            )
        if cash_df.empty:
            other_state = sum(
                int(self.store.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for table in ("paper_positions", "paper_lots", "paper_receivables")
            )
            if other_state:
                raise ValueError("paper_cash yok ama mevcut pozisyon/lot/alacak var; init mevcut defteri silemez")
            if not self._b0_codes:
                raise ValueError(_b0_profile_error())
            self.init_capital(self.capital)
            return
        pos_df = self.store.con.execute("SELECT * FROM paper_positions").df()
        lots_df = self.store.con.execute("SELECT * FROM paper_lots").df()
        position_codes = set(
            pos_df.loc[pd.to_numeric(pos_df.get("units"), errors="coerce").fillna(0) > 1e-12, "fund_code"]
        )
        lot_codes = set(
            lots_df.loc[pd.to_numeric(lots_df.get("units"), errors="coerce").fillna(0) > 1e-12, "fund_code"]
        )
        missing_codes = (position_codes | lot_codes).difference(self.meta.index)
        if missing_codes:
            missing = ", ".join(sorted(map(str, missing_codes)))
            raise ValueError(f"Kayıtlı paper pozisyon/lot fonu {missing} artık PIT fund_master/meta evreninde yok")
        self.ledger = Ledger(
            meta=self.meta,
            cash=float(cash_df["cash"].iloc[0]),
            cash_tax_rate=self._cash_rate,
            dates=self._nav_dates(),
        )
        for _, row in pos_df.iterrows():
            code = row["fund_code"]
            i = int(self.meta.index.get_loc(code))
            self.ledger.units[i] = float(row["units"])
        for _, row in lots_df.iterrows():
            code = row["fund_code"]
            i = int(self.meta.index.get_loc(code))
            bought_idx = self._date_idx(row.get("bought_date"))
            avail_idx = self._date_idx(row.get("available_date"))
            saved_bought = row.get("bought_idx")
            saved_available = row.get("available_idx")
            bought_idx = int(saved_bought) if pd.notna(saved_bought) else bought_idx
            avail_idx = int(saved_available) if pd.notna(saved_available) else avail_idx
            self.ledger.lots[i].append(
                Lot(
                    units=float(row["units"]),
                    cost_per_unit=float(row["cost_per_unit"]),
                    bought_idx=bought_idx if bought_idx is not None else 0,
                    tax_rate=float(row["tax_rate"]),
                    available_idx=avail_idx if avail_idx is not None else len(self.ledger.dates) + 1,
                )
            )
        rec_df = self.store.con.execute("SELECT settle_idx, amount FROM paper_receivables").df()
        self.ledger.receivables = {int(r.settle_idx): float(r.amount) for r in rec_df.itertuples(index=False)}
        self._record_b0_memberships()

    def _record_b0_memberships(self) -> None:
        """Append dated B0 identity for each live basket member; never erase prior membership."""
        if not self._b0_codes:
            return
        membership_date = pd.Timestamp(self.asof).date()
        profile = self._fund_master.drop_duplicates("fund_code").set_index("fund_code")
        identity = profile.reindex(self._b0_codes)
        found = {str(code) for code in identity.index if pd.notna(identity.loc[code, "snapshot_date"])}
        if found != set(self._b0_codes):
            missing = ", ".join(sorted(set(self._b0_codes).difference(found)))
            raise ValueError(f"B0 üyeliğinin tarihli kimliği eksik: {missing}")
        self.store.con.executemany(
            "INSERT OR IGNORE INTO paper_b0_memberships VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    membership_date,
                    pd.Timestamp(identity.loc[code, "snapshot_date"]).date(),
                    str(code),
                    identity.loc[code, "founder_code"],
                    int(identity.loc[code, "buy_valor"]),
                    int(identity.loc[code, "sell_valor"]),
                )
                for code in self._b0_codes
            ],
        )

    def init_capital(self, capital: float = 100.0) -> None:
        """İlk sermayeyi yalnız serbest nakit olarak yazar; B0 alımı ilk proposal/fill yolundadır."""
        capital = float(capital)
        if self._freshly_initialized:
            if not np.isclose(capital, self.capital):
                raise ValueError("bu oturumda defter zaten başlatıldı; mevcut pozisyonlar init ile sıfırlanamaz")
            return

        legacy_proxy_positions = int(
            self.store.con.execute(
                "SELECT count(*) FROM paper_positions WHERE fund_code=? AND coalesce(units, 0) > 1e-12", [SLOT_CODE]
            ).fetchone()[0]
        )
        legacy_proxy_lots = int(
            self.store.con.execute(
                "SELECT count(*) FROM paper_lots WHERE fund_code=? AND coalesce(units, 0) > 1e-12", [SLOT_CODE]
            ).fetchone()[0]
        )
        if legacy_proxy_positions or legacy_proxy_lots:
            raise ValueError(
                "Eski CASH_PROXY bakiye/lot bulundu; gerçek B0 fonlarına otomatik dönüşüm güvenli değil. "
                "Açık kullanıcı kararı gerekir; defter değiştirilmedi."
            )
        persisted_state = sum(
            int(self.store.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in ("paper_cash", "paper_positions", "paper_lots", "paper_receivables")
        )
        if persisted_state:
            raise ValueError("mevcut paper defteri var; init mevcut nakit/pozisyon/lotları sıfırlayamaz")
        self.capital = float(capital)
        self.ledger = Ledger(
            meta=self.meta,
            cash=self.capital,
            cash_tax_rate=self._cash_rate,
            dates=self._nav_dates(),
        )
        self._persist()
        snapshot_date = pd.Timestamp(self.asof).normalize()
        self.snapshot_equity(snapshot_date, portfolio_name="live")
        self._freshly_initialized = True

    def _latest_idx(self) -> int:
        n = len(self.ledger.dates)
        return max(0, n - 1)

    def _slot_price(self, idx: int | None = None) -> float | None:
        if self._cash_nav.empty:
            return None
        pos = len(self._cash_nav) - 1 if idx is None else int(idx)
        if pos < 0 or pos >= len(self._cash_nav):
            return None
        return float(self._cash_nav.iloc[pos])

    def _persist(self) -> None:
        # Validate and record the membership's PIT identity before rewriting any ledger state.
        self._record_b0_memberships()
        now = datetime.now()
        self.store.con.execute("DELETE FROM paper_cash")
        self.store.con.execute(
            "INSERT INTO paper_cash VALUES (?, ?, ?)",
            [now, float(self.ledger.cash), self.capital],
        )
        self.store.con.execute("DELETE FROM paper_positions")
        self.store.con.execute("DELETE FROM paper_lots")
        dates = pd.DatetimeIndex(self.ledger.dates)
        self.store.con.execute("DELETE FROM paper_receivables")
        for settle_idx, amount in self.ledger.receivables.items():
            self.store.con.execute("INSERT INTO paper_receivables VALUES (?, ?)", [int(settle_idx), float(amount)])
        for i, code in enumerate(self.meta.codes):
            units = float(self.ledger.units[i])
            if units <= 1e-12:
                continue
            cost = sum(lot.cost_per_unit * lot.units for lot in self.ledger.lots[i]) / units if units > 0 else 0.0
            self.store.con.execute(
                "INSERT INTO paper_positions VALUES (?, ?, ?, ?)",
                [now, str(code), units, cost],
            )
            for lot in self.ledger.lots[i]:
                bought_date = dates[lot.bought_idx].date() if lot.bought_idx < len(dates) else None
                avail_date = dates[lot.available_idx].date() if 0 <= lot.available_idx < len(dates) else None
                self.store.con.execute(
                    """INSERT INTO paper_lots
                    (updated_at, fund_code, units, cost_per_unit, bought_date, tax_rate, available_date, bought_idx, available_idx)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        now,
                        str(code),
                        float(lot.units),
                        float(lot.cost_per_unit),
                        bought_date,
                        float(lot.tax_rate),
                        avail_date,
                        int(lot.bought_idx),
                        int(lot.available_idx),
                    ],
                )

    def latest_nav(self) -> pd.Series:
        nav = self._nav
        if nav.empty:
            return pd.Series(dtype=float)
        # Use only the as-of panel, carrying forward the last accessible NAV for every fund.
        # This is valuation only; order/fill paths still inspect the raw date-specific NAV and stale flags.
        marks = nav.where(nav > 0).ffill().iloc[-1].copy()
        held_codes = {str(self.meta.codes[i]) for i, units in enumerate(self.ledger.units) if units > 1e-12}
        unpriced = [
            code
            for code in held_codes
            if code not in marks.index or not np.isfinite(marks.get(code, np.nan)) or marks.get(code, 0.0) <= 0
        ]
        if unpriced:
            raise ValueError(
                f"Kayıtlı paper pozisyonu için as-of öncesi erişilebilir NAV yok: {', '.join(sorted(unpriced))}"
            )
        for code in set(self._b0_codes) | set(self._b0_history_codes):
            if code not in marks.index or not np.isfinite(marks.get(code, np.nan)) or marks.get(code, 0.0) <= 0:
                raise ValueError(f"B0 fonu {code} için as-of öncesi erişilebilir NAV yok")
        return marks

    def equity(self) -> float:
        nav = self.latest_nav()
        if nav.empty:
            return float(self.ledger.cash)
        arr = nav.reindex(self.meta.index).to_numpy(float).copy()
        return float(self.ledger.equity(arr, None, 0.0))

    def position_components(self, nav: pd.Series | None = None) -> tuple[np.ndarray, float, float]:
        """Fon değerleri ve gerçek B0/risky piyasa değerleri; CASH_PROXY sanal pozisyon değildir."""
        marks = self.latest_nav() if nav is None else nav
        arr = marks.reindex(self.meta.index).to_numpy(float).copy()
        position_values = self.ledger.position_value(arr, None, 0.0)
        b0_codes = tuple(set(self._b0_codes) | set(self._b0_history_codes))
        b0_mask = np.isin(self.meta.codes, b0_codes)
        if self._slot_i is not None:
            b0_mask[self._slot_i] = False
        slot_value = float(position_values[b0_mask].sum())
        risky_mask = ~b0_mask
        if self._slot_i is not None:
            risky_mask[self._slot_i] = False
        risky_value = float(position_values[risky_mask].sum())
        return position_values, risky_value, slot_value

    def peak_equity(self) -> float:
        if self.asof is None:
            row = self.store.con.execute("SELECT max(equity) AS peak FROM paper_equity").fetchone()
        else:
            row = self.store.con.execute(
                "SELECT max(equity) AS peak FROM paper_equity WHERE date <= ?", [pd.Timestamp(self.asof).date()]
            ).fetchone()
        peak = row[0] if row and row[0] is not None else None
        return float(peak) if peak is not None else self.equity()

    def drawdown(self) -> float:
        eq = self.equity()
        peak = self.peak_equity()
        return 1.0 - eq / peak if peak > 0 else 0.0

    def snapshot_equity(self, date: pd.Timestamp, portfolio_name: str = "live") -> None:
        nav = self.latest_nav()
        if nav.empty or self._slot_i is None:
            return
        cash = float(self.ledger.cash)
        receivables = float(self.ledger.receivable_total())
        _, risky_value, slot_value = self.position_components(nav)
        equity = cash + receivables + risky_value + slot_value
        nav_asof = nav.name if isinstance(nav.name, pd.Timestamp) else pd.to_datetime(nav.name)
        self.store.con.execute(
            "DELETE FROM paper_equity WHERE portfolio_name = ? AND date = ?",
            [portfolio_name, date.date()],
        )
        self.store.con.execute(
            """
            INSERT INTO paper_equity (portfolio_name, date, equity, cash, receivables, risky_value, slot_value, nav_asof)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [portfolio_name, date.date(), equity, cash, receivables, risky_value, slot_value, nav_asof.date()],
        )


def source_stale(store, cfg: dict, asof: pd.Timestamp | None = None) -> tuple[bool, dict]:
    """Kaynak eski mi? D verilmişse yalnız o tarihe kadar yayımlanan girdileri kullan."""
    wide = store.nav_wide()
    fm = store.latest_fund_master()
    if asof is not None:
        d = pd.Timestamp(asof).normalize()
        wide = wide.loc[wide.index <= d]
        rows = store.con.execute("SELECT * FROM fund_master WHERE snapshot_date <= ?", [d.date()]).df()
        if rows.empty:
            return True, {"source_stale": True, "reason": "as-of PIT fund_master snapshot yok"}
        fm = rows[rows["snapshot_date"] == rows["snapshot_date"].max()]
    q = quality_summary(wide, fm, cfg, asof=asof, eligible=None)
    return bool(q.get("source_stale", False)), q


def target_weights_from_selection(
    selection: pd.DataFrame,
    nav: pd.DataFrame,
    fund_master: pd.DataFrame,
    cfg: dict,
    dd: float = 0.0,
) -> pd.Series:
    """B2c-fdr: BH seçimi boşsa sepet; doluysa HRP + tavanlar + riskli tavan + DD kuralı."""
    tefas = cfg["legs"]["tefas"]
    constraints = tefas["constraints"]
    risk = tefas["risk"]
    hrp_cfg = cfg.get("hrp", {})

    sel = selection.copy()
    sel["selected_q20"] = sel["selected_q20"].fillna(False).astype(bool)
    selected = sel.loc[sel["selected_q20"], "fund_code"].tolist()
    if not selected:
        return pd.Series({SLOT_CODE: 1.0})

    fm = fund_master.drop_duplicates("fund_code").set_index("fund_code")
    founders = fm.get("founder_code")

    available = [c for c in selected if c in nav.columns]
    if len(available) < len(selected):
        logger.warning("seçimdeki {} fon NAV'da yok", len(selected) - len(available))
    if not available:
        return pd.Series({SLOT_CODE: 1.0})

    rets = np.log(nav[available]).diff().dropna()
    cov_days = int(hrp_cfg.get("cov_days", 126))
    rets = rets.iloc[-cov_days:]
    rets = rets.loc[:, rets.notna().mean() >= 0.8]
    if rets.shape[1] == 0:
        return pd.Series({SLOT_CODE: 1.0})

    cov = ledoit_wolf_cov(rets)
    clusters = pd.Series(
        cluster_labels(cov, hrp_cfg.get("cluster_distance", 0.4), hrp_cfg.get("linkage", "single")),
        index=rets.columns,
    )
    scores = sel.set_index("fund_code")["lower"].reindex(rets.columns).dropna().sort_values(ascending=False)
    chosen = select_with_limits(
        scores,
        n=len(rets.columns),
        groups=founders,
        clusters=clusters,
        max_per_group=int(constraints.get("max_funds_per_founder", 3)),
        max_per_cluster=int(constraints.get("max_funds_per_hrp_cluster", 3)),
        keep=set(),
    )
    if not chosen:
        return pd.Series({SLOT_CODE: 1.0})

    idx = [list(rets.columns).index(c) for c in chosen]
    w = pd.Series(hrp_weights(cov[np.ix_(idx, idx)], hrp_cfg.get("linkage", "single")), index=chosen)
    w = apply_caps(
        w,
        constraints.get("max_weight_per_fund", 0.25),
        constraints.get("max_weight_per_founder", 0.30),
        founders,
    )

    risky_ceiling = constraints.get("pilot_risky_ceiling", 0.30)
    if dd > risk.get("dd_trigger_medium", 0.12):
        risky_ceiling = min(risky_ceiling, risk.get("dd_exposure", 0.65))

    if w.sum() > risky_ceiling:
        w = w * risky_ceiling / w.sum()

    cash_w = max(0.0, 1.0 - w.sum())
    out = w.copy()
    out[SLOT_CODE] = cash_w
    return out


def orders_from_targets(
    target: pd.Series,
    paper: PaperLedger,
    cfg: dict,
    proposal_id: str,
    decision_idx: int | None = None,
    dd: float = 0.0,
) -> pd.DataFrame:
    """Hedef farkından emir üret; 21 işlem günü kanıt tutma ve icra kapılarını uygula."""
    tefas = cfg["legs"]["tefas"]
    drift = float(tefas["rebalance"]["drift_threshold"])
    drift_tax = float(tefas["rebalance"]["drift_threshold_taxable_sale"])
    nav = paper.latest_nav()
    arr = nav.reindex(paper.meta.index).to_numpy(float).copy()
    decision_idx = paper._latest_idx() if decision_idx is None else int(decision_idx)
    decision_day = pd.Timestamp(paper.asof).normalize()
    publish_time = str(cfg.get("calendar", {}).get("nav_publish_time", "10:00"))
    publish_hour, publish_minute = (int(part) for part in publish_time.split(":"))
    nav_reference = expected_last_nav_date(decision_day + pd.Timedelta(hours=publish_hour, minutes=publish_minute), cfg)
    max_stale_days = int(tefas.get("universe", {}).get("max_stale_days", 2))
    stale_codes = flag_data_stale(
        paper._nav.where(paper._nav > 0), max_stale_days=max_stale_days, reference=nav_reference, cfg=cfg
    )
    slot_price = paper._slot_price(decision_idx)
    if paper._slot_i is not None and slot_price is not None:
        arr[paper._slot_i] = slot_price
    current_weights = pd.Series(paper.ledger.weights(arr, None, 0.0), index=paper.meta.index)
    meta_idx = {code: i for i, code in enumerate(paper.meta.codes)}

    # The public target keeps one logical B0 slot; executable paper orders address its real members.
    if SLOT_CODE in target.index:
        slot_weight = float(target.get(SLOT_CODE, 0.0))
        target = target.drop(SLOT_CODE)
        if paper._b0_codes:
            for code in paper._b0_codes:
                target.loc[code] = float(target.get(code, 0.0)) + slot_weight / len(paper._b0_codes)

    # Apply founder capacity once to B0 + risky targets. Locked/min-hold lots are
    # protected floors; any rejected target weight remains uninvested cash.
    b0_codes = set(paper._b0_codes) | set(paper._b0_history_codes)
    min_hold = int(cfg.get("paper", {}).get("min_hold_days", 21))
    dd_override = dd > float(tefas["risk"].get("dd_trigger_medium", 0.12))
    protected = pd.Series(False, index=target.index.union(current_weights.index))
    for code in protected.index:
        i = meta_idx.get(code)
        if i is None or current_weights.get(code, 0.0) <= 1e-12:
            continue
        lots = paper.ledger.lots[i]
        first_buy_idx = min((lot.bought_idx for lot in lots), default=decision_idx)
        in_min_hold = code not in b0_codes and not dd_override and decision_idx - first_buy_idx < min_hold
        cannot_sell = (
            not bool(paper.meta.can_sell[i])
            or "SELL" in paper._execution_blocks.get(str(code), {})
            or bool(stale_codes.get(code, True))
            or paper.ledger.sellable_units(i, decision_idx) + 1e-12 < float(paper.ledger.units[i])
        )
        protected.loc[code] = in_min_hold or cannot_sell
    founder_map = paper._fund_master.drop_duplicates("fund_code").set_index("fund_code").get("founder_code")
    target = constrain_founder_targets(
        target,
        current_weights,
        founder_map if founder_map is not None else pd.Series(dtype="string"),
        max_weight=float(tefas["constraints"].get("max_weight_per_founder", 0.30)),
        max_funds=int(tefas["constraints"].get("max_funds_per_founder", 3)),
        protected=protected,
    )

    idx = target.index.union(current_weights.index)
    target = target.reindex(idx).fillna(0.0)
    current = current_weights.reindex(idx).fillna(0.0)
    diff = target - current
    equity = max(float(paper.equity()), 0.0)
    available_cash = float(paper.ledger.cash) + sum(
        amount for settle_idx, amount in paper.ledger.receivables.items() if settle_idx <= decision_idx
    )
    risky_buy_codes = [
        code for code in idx if code not in b0_codes and diff[code] > 1e-9 and not bool(stale_codes.get(code, True))
    ]
    risky_buy_value = float(diff.reindex(risky_buy_codes).sum()) * equity
    risky_buy_scale = min(1.0, available_cash / risky_buy_value) if risky_buy_value > 0 else 1.0
    cash_after_risky = max(0.0, available_cash - risky_buy_value * risky_buy_scale)
    b0_buy_codes = [
        code for code in idx if code in b0_codes and diff[code] > 1e-9 and not bool(stale_codes.get(code, True))
    ]
    b0_buy_value = float(diff.reindex(b0_buy_codes).sum()) * equity
    b0_buy_scale = min(1.0, cash_after_risky / b0_buy_value) if b0_buy_value > 0 else 1.0
    orders = []
    for code in idx:
        if code == SLOT_CODE:
            continue
        i = meta_idx.get(code)
        if i is None:
            continue
        if bool(stale_codes.get(code, True)):
            continue
        delta = float(diff[code])
        if abs(delta) < 1e-9:
            continue
        action = "BUY" if delta > 0 else "SELL"
        if action in paper._execution_blocks.get(str(code), {}):
            continue
        if action == "BUY" and code not in b0_codes:
            delta *= risky_buy_scale
            if delta <= 1e-9:
                continue
        if action == "BUY" and code in b0_codes:
            delta *= b0_buy_scale
            if delta <= 1e-9:
                continue
        if action == "SELL":
            if not paper.meta.can_sell[i] or not np.isfinite(arr[i]):
                continue
            lots = paper.ledger.lots[i]
            first_buy_idx = min((lot.bought_idx for lot in lots), default=decision_idx)
            if code not in b0_codes and not dd_override and decision_idx - first_buy_idx < min_hold:
                continue
            if paper.ledger.sellable_units(i, decision_idx) <= 1e-12:
                continue
            gain_ratio = paper.ledger.unrealized_gain_ratio(i, arr[i])
            threshold = drift_tax if gain_ratio > 0.5 else drift
            if abs(delta) < threshold and target[code] > 0:
                continue
        elif not paper.meta.can_buy[i] or not np.isfinite(arr[i]):
            continue
        elif delta < drift and current[code] > 0:
            continue
        orders.append(
            {
                "proposal_id": proposal_id,
                "fund_code": code,
                "action": action,
                "target_w": float(current[code] + delta),
                "current_w": float(current[code]),
                "delta_w": delta,
                "buy_valor": int(paper.meta.buy_valor[i]),
                "sell_valor": int(paper.meta.sell_valor[i]),
            }
        )
    return pd.DataFrame(
        orders,
        columns=["proposal_id", "fund_code", "action", "target_w", "current_w", "delta_w", "buy_valor", "sell_valor"],
    )


def format_orders_md(
    orders: pd.DataFrame,
    proposal_id: str,
    date: str,
    warnings: list[str],
    proof_count: int | None = None,
    hold_codes: list[str] | None = None,
) -> str:
    """Yüzde raporu (Markdown); BH kanıt adedini emir adedinden ayırır."""
    n_proof = len(orders) if proof_count is None else int(proof_count)
    hold_codes = hold_codes or []
    lines = [
        f"# Öneri listesi — {date}",
        f"**proposal_id:** {proposal_id}  ",
        f"**kanıt:** {'var' if n_proof else 'yok'} ({n_proof} fon)",
        "",
    ]
    if orders.empty:
        if hold_codes:
            lines.append(
                f"Emir yok — kanıt kaybı; 21 iş günü dolana kadar mevcut pozisyonları tut ({', '.join(hold_codes)})"
            )
        else:
            lines.append("Emir yok — tut (nakit sepeti)")
    else:
        lines.extend(
            [
                "| fon | yön | hedef % | mevcut % | delta % | alış valör | satış valör |",
                "|-----|-----|---------|----------|---------|------------|-------------|",
            ]
        )
        for _, row in orders.iterrows():
            lines.append(
                f"| {row['fund_code']} | {row['action']} | {row['target_w'] * 100:+.1f} | "
                f"{row['current_w'] * 100:.1f} | {row['delta_w'] * 100:+.1f} | "
                f"T+{row['buy_valor']} | T+{row['sell_valor']} |"
            )
    if warnings:
        lines.extend(["", "## Uyarılar", *(f"- {w}" for w in warnings)])
    return "\n".join(lines)


def format_orders_telegram(
    orders: pd.DataFrame,
    proposal_id: str,
    date: str,
    warnings: list[str],
    proof_count: int | None = None,
    hold_codes: list[str] | None = None,
) -> str:
    """Telegram HTML: yüzde, kod, yön, valör; koruma süresi görünür."""
    n = len(orders) if proof_count is None else int(proof_count)
    hold_codes = hold_codes or []
    head = f"<b>JANUS öneri — {date}</b>\nkanıt: {'var' if n else 'yok'} ({n} fon) | {html.escape(proposal_id)}"
    if orders.empty:
        if hold_codes:
            action = f"kanıt kaybı — {', '.join(html.escape(c) for c in hold_codes)} 21 iş günü dolana kadar tutuluyor"
        else:
            action = "tut (nakit sepeti)"
        return (
            head
            + f"\n<i>{action}</i>"
            + ("\n" + "\n".join(f"⚠ {html.escape(w)}" for w in warnings) if warnings else "")
        )
    lines = [head, ""]
    for _, row in orders.iterrows():
        v = f"alış T+{row['buy_valor']}/satış T+{row['sell_valor']}"
        lines.append(
            f"• <code>{html.escape(row['fund_code'])}</code> {row['action']} → "
            f"hedef %{row['target_w'] * 100:.1f} (Δ %{row['delta_w'] * 100:+.1f}) {v}"
        )
    if warnings:
        lines.extend(["", *(f"⚠ {html.escape(w)}" for w in warnings)])
    return "\n".join(lines)


def paper_propose(store, cfg: dict, date: str, root: Path, out_dir: Path | None = None) -> dict:
    """Bir karar günü için PIT sınırlarına uyan kağıt önerisi üret; bilinmeyen geçmiş durumunda fail-closed."""
    d = pd.Timestamp(date).normalize()
    _ensure_paper_tables(store)
    warnings: list[str] = []
    latest_state = store.con.execute("SELECT max(date) FROM paper_equity WHERE portfolio_name='live'").fetchone()[0]
    if latest_state is not None and d.date() < latest_state:
        warnings.append("geçmiş defter durumu yeniden kurulamıyor; eski öneri üretilmedi")
    stale, q = source_stale(store, cfg, asof=d)
    if stale:
        warnings.append(f"kaynak eski/as-of verisi yok ({q.get('fresh_days_behind', 0):.0f} iş günü) → emir üretilmez")
    selection_path = root / "data" / "predictions" / f"selection_{d:%Y-%m-%d}.parquet"
    if not selection_path.exists():
        warnings.append(f"{selection_path.name} yok → emir üretilmez")
    if warnings:
        return {
            "date": str(d.date()),
            "status": "hold",
            "proposal_id": None,
            "n_orders": 0,
            "warnings": warnings,
            "telegram": format_orders_telegram(pd.DataFrame(), "", str(d.date()), warnings, proof_count=0),
        }

    selection = pd.read_parquet(selection_path)
    if "decision_at" not in selection or not (pd.to_datetime(selection["decision_at"]).dt.normalize() == d).all():
        warnings.append("seçim decision_at as-of tarihiyle eşleşmiyor → emir üretilmez")
        return {
            "date": str(d.date()),
            "status": "hold",
            "proposal_id": None,
            "n_orders": 0,
            "warnings": warnings,
            "telegram": format_orders_telegram(pd.DataFrame(), "", str(d.date()), warnings, proof_count=0),
        }
    try:
        paper = PaperLedger(store, cfg, asof=d)
    except ValueError as exc:
        warnings.append(f"PIT defter girdisi yok: {exc}")
        return {
            "date": str(d.date()),
            "status": "hold",
            "proposal_id": None,
            "n_orders": 0,
            "warnings": warnings,
            "telegram": format_orders_telegram(pd.DataFrame(), "", str(d.date()), warnings, proof_count=0),
        }
    selection = selection.copy()
    selection["selected_q20"] = selection["selected_q20"].fillna(False).astype(bool)
    evidence_codes = sorted(selection.loc[selection["selected_q20"], "fund_code"].astype(str).unique())
    proof_count = len(evidence_codes)
    if 0 < len(paper._b0_codes) < 3:
        warnings.append(f"B0 uygun aday sayısı {len(paper._b0_codes)}; 3–5 hedefinin altında, uyarıyla devam")
    paper.snapshot_equity(d, portfolio_name="live")
    relevant_codes = set(selection["fund_code"].astype(str))
    relevant_codes.update(paper._b0_codes)
    relevant_codes.update(paper._b0_history_codes)
    relevant_codes.update(str(code) for i, code in enumerate(paper.meta.codes) if paper.ledger.units[i] > 1e-12)
    if "umbrella_type" in paper._fund_master:
        relevant_codes.update(
            paper._fund_master.loc[
                paper._fund_master["umbrella_type"].fillna("").astype(str).str.contains("Para Piyasası"),
                "fund_code",
            ].astype(str)
        )
    execution_blocks = []
    for code in sorted(relevant_codes):
        sides = paper._execution_blocks.get(code, {})
        assumption = paper._publication_assumptions.get(code)
        for side, reason in sorted(sides.items()):
            execution_blocks.append(
                {
                    "record_type": "block",
                    "fund_code": code,
                    "side": side,
                    "reason": reason,
                    "publication_time_assumption": assumption,
                }
            )
        if assumption == "date_only_pit_assumption":
            execution_blocks.append(
                {
                    "record_type": "publication_assumption",
                    "fund_code": code,
                    "side": "INFO",
                    "reason": assumption,
                    "publication_time_assumption": assumption,
                }
            )
    for code in sorted(relevant_codes):
        sides = paper._execution_blocks.get(code, {})
        assumption = paper._publication_assumptions.get(code)
        if not sides and assumption != "date_only_pit_assumption":
            continue
        if sides:
            reasons = ", ".join(sorted(set(sides.values())))
            warnings.append(f"fon {code} icra engeli: {reasons}")
        if assumption == "date_only_pit_assumption":
            warnings.append(f"fon {code} yayın zamanı: date_only_pit_assumption")
    if not paper._b0_codes:
        warnings.append("B0 uygun aday sayısı 0; yeni proposal/emir yok, mevcut pozisyonlar son NAV ile tutuluyor")
        return {
            "date": str(d.date()),
            "status": "hold",
            "proposal_id": None,
            "n_orders": 0,
            "proof_count": proof_count,
            "warnings": warnings,
            "telegram": format_orders_telegram(pd.DataFrame(), "", str(d.date()), warnings, proof_count=proof_count),
        }
    dd = paper.drawdown()
    target = target_weights_from_selection(selection, paper._nav, paper._fund_master, cfg, dd=dd)
    decision_idx = paper._date_idx(d)
    if decision_idx is None:
        if not paper._nav.empty and d == next_business_day(paper._nav.index[-1], cfg):
            # D decision uses the latest published NAV (D−1); the D price is not expected yet.
            decision_idx = len(paper.ledger.dates) - 1
        else:
            warnings.append("karar günü son NAV'dan sonraki ilk iş günü değil → emir üretilmedi")
            return {
                "date": str(d.date()),
                "status": "hold",
                "proposal_id": None,
                "n_orders": 0,
                "warnings": warnings,
                "telegram": format_orders_telegram(
                    pd.DataFrame(), "", str(d.date()), warnings, proof_count=proof_count
                ),
            }
    proposal_id = _fmt_proposal_id(d)
    orders = orders_from_targets(target, paper, cfg, proposal_id, decision_idx=decision_idx, dd=dd)
    min_hold = int(cfg.get("paper", {}).get("min_hold_days", 21))
    dd_override = dd > float(cfg["legs"]["tefas"]["risk"].get("dd_trigger_medium", 0.12))
    hold_codes = []
    if not dd_override:
        for i, code in enumerate(paper.meta.codes[:-1]):
            lots = paper.ledger.lots[i]
            first_buy_idx = min((lot.bought_idx for lot in lots), default=decision_idx)
            if paper.ledger.units[i] > 1e-12 and decision_idx - first_buy_idx < min_hold:
                hold_codes.append(str(code))
    if hold_codes:
        warnings.append(f"kanıt kaybında asgari tutma süresi sürüyor: {', '.join(hold_codes)}")
    existing = store.con.execute(
        "SELECT proposal_id, orders_json FROM paper_proposals WHERE date = ?", [d.date()]
    ).fetchone()
    if existing:
        proposal_id = existing[0]
        orders = pd.DataFrame(json.loads(existing[1]))
    else:
        expires = datetime.combine(d.date(), time(12, 0))
        store.con.execute(
            """INSERT INTO paper_proposals
            (proposal_id, date, created_at, expires_at, status, orders_json, evidence_count, evidence_codes,
             target_weights_json, execution_blocks_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                proposal_id,
                d.date(),
                datetime.now(),
                expires,
                "proposed",
                orders.to_json(orient="records"),
                proof_count,
                json.dumps(evidence_codes),
                json.dumps(target.to_dict()),
                json.dumps(execution_blocks),
            ],
        )

    out = Path(out_dir) if out_dir else root / cfg.get("reporting", {}).get("orders_dir", "reports")
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / f"orders_{d:%Y-%m-%d}.md"
    csv_path = out / f"orders_{d:%Y-%m-%d}.csv"
    md_path.write_text(
        format_orders_md(orders, proposal_id, str(d.date()), warnings, proof_count, hold_codes), encoding="utf-8"
    )
    orders.to_csv(csv_path, index=False)
    return {
        "date": str(d.date()),
        "status": "proposed",
        "proposal_id": proposal_id,
        "n_orders": len(orders),
        "proof_count": proof_count,
        "risky_weight": float(target.drop(SLOT_CODE, errors="ignore").sum()),
        "cash_weight": float(target.get(SLOT_CODE, 0.0)),
        "drawdown": dd,
        "meta_asof": str(paper.meta_asof.date()),
        "nav_asof": str(paper.nav_asof.date()),
        "warnings": warnings,
        "telegram": format_orders_telegram(orders, proposal_id, str(d.date()), warnings, proof_count, hold_codes),
        "md_path": str(md_path),
        "csv_path": str(csv_path),
    }
