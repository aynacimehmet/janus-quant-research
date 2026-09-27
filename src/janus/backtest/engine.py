"""Backtest motoru: gün döngüsü (sıralı), fonlar üzerinde vektörize; as-of disiplini; sapma eşikleri; DD tetikleyici.

Strateji arayüzü: `strategy(ctx) -> (weights: pd.Series[fund→ağırlık, toplam ≤ 1], exposure: float)`.
ctx.nav yalnızca D-1'e kadar olan NAV'ları içerir (sinyal D-1 NAV görür); emirler D NAV'ında dolar.
Gate (rejim) stratejinin döndürdüğü exposure'ı belirler; DD > dd_trigger ise motor exposure'ı risk.dd_exposure ile sınırlar (0c-5: DD tetikleyicisinin hedefi exposure_levels['medium']'dan ayrı).

P05 (LEDGER §1): data_stale (NAV gecikmesi) ≠ suspended (resmî/senaryo). İkisinde de işlem yok;
değerleme ve DD/kapı official_value (son NAV, haircut YOK) ile; stress_value (suspended pozisyonlara
haircut) ayrı seri olarak raporlanır. D05: alışta can_buy, satışta can_sell; satılamayan elde tutulan
pozisyon kilitli sayılır, hedefler gerçekleştirilebilir kısma normalize edilir (LEDGER §6).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np
import pandas as pd

from janus.backtest.costs import FundMeta
from janus.backtest.ledger import Ledger

SLOT_CODE = "CASH_PROXY"


@dataclass
class StrategyContext:
    idx: int
    date: pd.Timestamp
    nav: pd.DataFrame  # as-of: satırlar ≤ D-1
    meta: FundMeta
    weights_now: pd.Series
    equity: float
    drawdown: float
    calendar: pd.DatetimeIndex
    extra: dict[str, Any] = field(default_factory=dict)


class Strategy(Protocol):
    name: str

    def __call__(self, ctx: StrategyContext) -> tuple[pd.Series, float]: ...


@dataclass
class BacktestConfig:
    initial_capital: float = 100.0
    drift_threshold: float = 0.03
    drift_threshold_taxable_sale: float = 0.06
    dd_trigger: float = 0.12
    dd_release: float = 0.06
    dd_exposure: float = 0.65  # DD tetikleyicisinin hedefi (config risk.dd_exposure)
    exposure_levels: dict[str, float] = field(default_factory=lambda: {"low": 0.30, "medium": 0.65, "full": 1.00})
    suspension_haircut: float = 0.30
    max_stale_days: int = 2
    cash_tax_rate: float = 0.175
    schedule: str = "monthly_first_business_day"
    min_trade_value: float = 0.0
    warmup_days: int = 252
    sensitivity: bool = False  # günlük nakit vergi yaklaşımı (P02); varsayılan: defter içi sepet slotu

    @classmethod
    def from_cfg(cls, cfg: dict) -> BacktestConfig:
        t = cfg["legs"]["tefas"]
        bt = cfg.get("backtest", {})
        return cls(
            initial_capital=float(bt.get("initial_capital", 100.0)),
            drift_threshold=float(t["rebalance"]["drift_threshold"]),
            drift_threshold_taxable_sale=float(t["rebalance"]["drift_threshold_taxable_sale"]),
            dd_trigger=float(t["risk"]["dd_trigger_medium"]),
            dd_release=float(t["risk"].get("dd_release", 0.06)),
            dd_exposure=float(t["risk"].get("dd_exposure", 0.65)),
            exposure_levels=dict(t["risk"]["exposure_levels"]),
            suspension_haircut=float(t["risk"]["suspension_haircut"]),
            max_stale_days=int(t["universe"].get("max_stale_days", 2)),
            cash_tax_rate=float(bt.get("cash_tax_rate", 0.175)),
            schedule=str(t["rebalance"]["schedule"]),
            warmup_days=int(bt.get("warmup_days", 252)),
            sensitivity=bool(bt.get("sensitivity", False)),
        )


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series
    cash_index: pd.Series
    weights: pd.DataFrame  # rebalance günlerinde hedef ağırlıklar
    weights_realized: pd.DataFrame  # rebalance günlerinde kapanış gerçekleşen ağırlıklar (F-11)
    fills: pd.DataFrame
    events: pd.DataFrame  # rebalance / dd_trigger / dd_release
    taxes_paid: float
    fees_paid: float
    turnover: float  # toplam alım+satım / ortalama değer (yıllıklandırılmamış)
    n_rebalances: int
    equity_stress: pd.Series  # suspended pozisyonlara haircut uygulanmış tasfiye değeri (P05)
    liquidation_value: float  # §3: V − tüm pozisyonlar bugün satılsa ödenecek stopaj
    unrealized_tax: float
    config: BacktestConfig

    @property
    def returns(self) -> pd.Series:
        return np.log(self.equity).diff().dropna()


def rebalance_days(calendar: pd.DatetimeIndex, schedule: str) -> np.ndarray:
    """Takvimde planlı rebalance günlerinin indeksleri."""
    if schedule == "monthly_first_business_day":
        ym = calendar.to_period("M")
        first = pd.Series(np.arange(len(calendar)), index=calendar).groupby(ym).min()
        return first.to_numpy()
    if schedule == "quarterly_first_business_day":
        q = calendar.to_period("Q")
        return pd.Series(np.arange(len(calendar)), index=calendar).groupby(q).min().to_numpy()
    if schedule == "weekly_monday":
        return np.where(calendar.dayofweek == 0)[0]
    if schedule == "daily":
        return np.arange(len(calendar))
    raise ValueError(f"bilinmeyen schedule: {schedule}")


def stale_flags(nav_wide: pd.DataFrame, max_stale_days: int) -> np.ndarray:
    """Her (gün, fon) için: son geçerli NAV'dan bu yana > max_stale_days satır geçtiyse True (askı adayı)."""
    valid = nav_wide.notna().to_numpy()
    n, m = valid.shape
    last = np.full(m, -(10**9))
    out = np.zeros((n, m), dtype=bool)
    for t in range(n):
        last = np.where(valid[t], t, last)
        out[t] = (t - last) > max_stale_days
    return out


def run_backtest(
    nav_wide: pd.DataFrame,
    meta: FundMeta,
    strategy: Strategy,
    cfg: BacktestConfig,
    cash_returns: pd.Series | None = None,
    suspended: pd.DataFrame | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    cash_nav: pd.Series | None = None,
    cash_category: str | None = None,
    cash_rate: float | None = None,
    sensitivity: bool | None = None,
    execution_meta_by_date: pd.DataFrame | None = None,
    allow_fee_stress: bool = False,
) -> BacktestResult:
    """nav_wide: DatetimeIndex × fon (meta.codes ile aynı sıra). cash_returns: günlük basit getiri (nakit fonu).

    P02: cash_nav verilirse (ve sensitivity kapalıysa) nakit vekili defter içi sepet slotudur (SLOT_CODE,
    valör 0/0, T+0 alım/bozum, stopaj bozumda); serbest nakit her gün slot'a sweep edilir, risky alışlar
    slot bozumuyla finanse edilir. cash_returns + sensitivity=True → eski günlük vergi yaklaşımı.
    cash_nav yokken cash_returns verilmesi sensitivity=True gerektirir; iki nakit girdisi de yoksa nakit
    getirisizdir.
    """
    sens = cfg.sensitivity if sensitivity is None else sensitivity
    if cash_nav is None and cash_returns is not None and not sens:
        raise ValueError("cash_returns günlük vergili fallback'i için sensitivity=True açıkça belirtilmelidir")
    use_slot = cash_nav is not None and not sens
    if use_slot:
        meta = _with_cash_slot(meta, cash_category, cfg.cash_tax_rate if cash_rate is None else cash_rate)
    nav_wide = nav_wide.reindex(columns=list(meta.codes))
    cal = nav_wide.index
    if start is not None:
        cal = cal[cal >= pd.Timestamp(start)]
    if end is not None:
        cal = cal[cal <= pd.Timestamp(end)]
    nav = nav_wide.loc[cal]
    if use_slot:
        slot_px = cash_nav.reindex(cal).ffill()
        if slot_px.isna().any():
            raise ValueError("cash_nav başlangıçtan itibaren eksik değer içeriyor (F-13)")
        nav = nav.assign(**{SLOT_CODE: (slot_px / slot_px.iloc[0]).to_numpy()})
    nav_ff = nav.ffill()  # değerleme için son bilinen NAV
    nav_arr, nav_ff_arr = nav.to_numpy(float), nav_ff.to_numpy(float)
    r_cash = cash_returns.reindex(cal).fillna(0.0).to_numpy(float) if cash_returns is not None else np.zeros(len(cal))
    susp = (
        suspended.reindex(index=cal, columns=list(meta.codes)).fillna(False).to_numpy(bool)
        if suspended is not None
        else np.zeros((len(cal), len(meta.codes)), dtype=bool)
    )  # yalnızca resmî/senaryo askısı; DD official_value ile, stress_value haircutlu

    if execution_meta_by_date is not None:
        required = {"buy_valor", "sell_valor", "entry_fee", "exit_fee", "can_buy", "can_sell", "tax_category"}
        missing = required.difference(execution_meta_by_date.columns)
        if missing:
            raise ValueError(f"execution_meta_by_date missing required columns: {sorted(missing)}")
        if not isinstance(execution_meta_by_date.index, pd.MultiIndex) or execution_meta_by_date.index.nlevels != 2:
            raise ValueError("execution_meta_by_date index must be MultiIndex(decision_date, fund_code)")
        if list(execution_meta_by_date.index.names) != ["decision_date", "fund_code"]:
            raise ValueError("execution_meta_by_date index names must be decision_date, fund_code")

    led = Ledger(meta, cash=cfg.initial_capital, cash_tax_rate=cfg.cash_tax_rate, dates=cal.to_numpy())
    slot_i = len(meta.codes) - 1 if use_slot else None
    sched = set(rebalance_days(cal, cfg.schedule).tolist())
    equity = np.zeros(len(cal))
    equity_stress = np.zeros(len(cal))
    if use_slot:
        cash_index = (slot_px / slot_px.iloc[0]).to_numpy() * cfg.initial_capital  # B0 slot MTM (Ekle-3)
    else:
        r_eff = r_cash * (1.0 - cfg.cash_tax_rate)
        r_eff[0] = 0.0  # ilk gün tahakkuk yok (defterle aynı)
        cash_index = np.cumprod(1.0 + r_eff) * cfg.initial_capital
    peak, gate_forced, events, weight_rows, realized_rows = cfg.initial_capital, False, [], {}, {}
    traded_value = 0.0
    start_idx = min(cfg.warmup_days, max(len(cal) - 1, 0))

    for t in range(len(cal)):
        daily_meta, execution_reasons = _execution_meta_for_day(
            meta, execution_meta_by_date, cal[t], allow_fee_stress=allow_fee_stress
        )
        led.meta = daily_meta
        led.settle(t)
        if t > 0 and not use_slot:
            led.accrue_cash(r_cash[t])  # günlük vergi yaklaşımı yalnızca sensitivite yolunda (P02)
        p_dec = nav_ff_arr[t - 1] if t > 0 else nav_ff_arr[0]  # F-04: karar D-1 bilgisiyle
        eq_open = led.equity(p_dec, None, 0.0)  # official_value: DD/kapı haircut'sız (P05)
        dd = 1.0 - eq_open / peak if peak > 0 else 0.0

        # --- DD tetikleyici (hysteresis) ---
        trigger_now = False
        if dd > cfg.dd_trigger and not gate_forced:
            gate_forced, trigger_now = True, True
            events.append({"idx": t, "date": cal[t], "event": "dd_trigger", "dd": dd})
        elif gate_forced and dd < cfg.dd_release and t in sched:
            gate_forced = False
            events.append({"idx": t, "date": cal[t], "event": "dd_release", "dd": dd})

        if t >= start_idx and (t in sched or trigger_now):
            ctx = StrategyContext(
                idx=t,
                date=cal[t],
                nav=nav.iloc[:t].drop(columns=[SLOT_CODE], errors="ignore"),  # F-10
                meta=daily_meta,
                weights_now=pd.Series(led.weights(p_dec, None, 0.0), index=meta.index),
                equity=eq_open,
                drawdown=dd,
                calendar=cal,
            )
            w_target, exposure = strategy(ctx)
            if gate_forced:
                exposure = min(exposure, cfg.dd_exposure)
            w = w_target.reindex(meta.index).fillna(0.0).clip(lower=0.0).to_numpy()
            s = w.sum()
            if s > 1.0:
                w = w / s
            w = w * exposure
            # bugün fiyatı olmayan, verisi eskiyen ya da askıdaki fon işlem görmez (P05)
            tradable = np.isfinite(nav_arr[t]) & ~susp[t]  # F-05: stale zaten NaN üzerinden işlem yok
            sellable = tradable & daily_meta.can_sell
            buyable = tradable & daily_meta.can_buy
            cur_w = led.weights(p_dec, None, 0.0)
            desired_w = w.copy()
            if execution_meta_by_date is not None:
                for i, code in enumerate(daily_meta.codes):
                    if code == SLOT_CODE or not tradable[i]:
                        continue
                    action = "BUY" if desired_w[i] > cur_w[i] + 1e-12 else "SELL"
                    if abs(desired_w[i] - cur_w[i]) <= 1e-12:
                        continue
                    allowed = daily_meta.can_buy[i] if action == "BUY" else daily_meta.can_sell[i]
                    reason = execution_reasons[action][i]
                    if not allowed and reason != "ok":
                        events.append(
                            {
                                "idx": t,
                                "date": cal[t],
                                "event": "execution_blocked",
                                "code": str(code),
                                "action": action,
                                "reason": reason,
                            }
                        )
            locked = (cur_w > 1e-12) & ~sellable  # satılamayan elde tutulan pozisyon (LEDGER §6)
            if use_slot:
                locked[slot_i] = False  # slot her zaman T+0 bozulabilir (Ekle-2)
            locked_w = float(cur_w[locked].sum())
            w = np.where(locked, cur_w, w)  # kilitli pozisyonun hedefi = mevcut ağırlık
            free_mask = ~locked
            if use_slot:
                free_mask[slot_i] = False  # slot sapma eşiğine ve turnover'a girmez (Ekle-2)
                w[slot_i] = 0.0
            free_sum = float(w[free_mask].sum())
            budget = max(1.0 - locked_w, 0.0)
            if free_sum > budget and free_sum > 0:
                w[free_mask] *= budget / free_sum  # gerçekleştirilebilir kısma normalize
            slot_target = max(0.0, 1.0 - locked_w - free_sum) if use_slot else 0.0
            w_reb = w
            if use_slot:
                w_reb = w.copy()
                w_reb[slot_i] = cur_w[slot_i]  # slot _rebalance_to'da işlem görmez (sweep/bozum motorunda)
            traded_value += _rebalance_to(
                led,
                w_reb,
                nav_arr[t],  # fill fiyatı = D
                t,
                cfg,
                sellable,
                buyable,
                None if not use_slot else (slot_i, nav_arr[:, slot_i]),
                val_price=p_dec,  # F-03/F-04: değerleme D-1 ffill
                force=trigger_now,  # F-02: DD tetik günü satış eşiği atlanır
            )
            if use_slot:
                w[slot_i] = slot_target  # risky ağırlık = 1 − slot ağırlığı (Ekle-2)
            weight_rows[cal[t]] = w
            realized_rows[cal[t]] = led.weights(nav_ff_arr[t], None, 0.0)  # F-11: kapanış gerçekleşen
            events.append(
                {
                    "idx": t,
                    "date": cal[t],
                    "event": "rebalance",
                    "dd": dd,
                    "exposure": exposure,
                    "locked_weight": locked_w,
                }
            )

        if use_slot and led.cash > 1e-9:  # serbest nakit aynı gün slot'a sweep edilir (P02, T+0)
            led.buy(slot_i, led.cash, float(nav_arr[t, slot_i]), t)
        equity[t] = led.equity(nav_ff_arr[t], None, 0.0)  # F-04: gün sonu değerleme D NAV
        equity_stress[t] = led.equity(nav_ff_arr[t], susp[t], cfg.suspension_haircut)
        peak = max(peak, equity[t])

    eq_s = pd.Series(equity, index=cal, name="equity")
    fills = pd.DataFrame([f.__dict__ for f in led.fills])
    if not fills.empty:
        fills["date"] = cal[fills["idx"].to_numpy()]
    weights = pd.DataFrame(weight_rows, index=meta.index).T if weight_rows else pd.DataFrame(columns=meta.index)
    weights_realized = (
        pd.DataFrame(realized_rows, index=meta.index).T if realized_rows else pd.DataFrame(columns=meta.index)
    )
    avg_eq = float(eq_s.iloc[start_idx:].mean()) if len(eq_s) > start_idx else cfg.initial_capital
    return BacktestResult(
        name=getattr(strategy, "name", strategy.__class__.__name__),
        equity=eq_s,
        cash_index=pd.Series(cash_index, index=cal),
        weights=weights,
        weights_realized=weights_realized,
        fills=fills,
        events=pd.DataFrame(events),
        taxes_paid=led.taxes_paid,
        fees_paid=led.fees_paid,
        turnover=traded_value / avg_eq if avg_eq > 0 else 0.0,
        n_rebalances=sum(1 for e in events if e["event"] == "rebalance"),
        equity_stress=pd.Series(equity_stress, index=cal, name="equity_stress"),
        liquidation_value=float(led.equity(nav_ff_arr[-1], None, 0.0) - led.unrealized_tax(nav_ff_arr[-1])),
        unrealized_tax=float(led.unrealized_tax(nav_ff_arr[-1])),
        config=cfg,
    )


def _with_cash_slot(meta: FundMeta, category: str | None, fallback_rate: float) -> FundMeta:
    """P02: sepet slotunu meta'ya ekler — valör 0/0 (T+0), komisyon 0; seçim/HRP evreni eligible ile dışında."""
    return FundMeta(
        codes=np.append(meta.codes, SLOT_CODE).astype(object),
        buy_valor=np.append(meta.buy_valor, 0),
        sell_valor=np.append(meta.sell_valor, 0),
        entry_fee=np.append(meta.entry_fee, 0.0),
        exit_fee=np.append(meta.exit_fee, 0.0),
        tax_rate=np.append(meta.tax_rate, fallback_rate),
        tax_unknown=np.append(meta.tax_unknown, False),
        equity_intensive=np.append(meta.equity_intensive, False),
        can_buy=np.append(meta.can_buy, True),
        can_sell=np.append(meta.can_sell, True),
        tax_category=np.append(meta.tax_category.astype(object), category),
        tax_schedule=meta.tax_schedule,
    )


def _execution_meta_for_day(
    base: FundMeta,
    execution_meta_by_date: pd.DataFrame | None,
    decision_date: pd.Timestamp,
    allow_fee_stress: bool = False,
) -> tuple[FundMeta, dict[str, np.ndarray]]:
    """Apply only decision-day execution fields; omitted profiles fail closed per fund."""
    if execution_meta_by_date is None:
        n = len(base.codes)
        return base, {"BUY": np.full(n, "ok", dtype=object), "SELL": np.full(n, "ok", dtype=object)}

    fund_positions = np.flatnonzero(base.codes != SLOT_CODE)
    fund_codes = base.codes[fund_positions]
    daily_index = pd.MultiIndex.from_product(
        [[pd.Timestamp(decision_date).normalize()], fund_codes], names=["decision_date", "fund_code"]
    )
    rows = execution_meta_by_date.reindex(daily_index).droplevel("decision_date").reindex(fund_codes)
    present = rows["buy_valor"].notna() & rows["sell_valor"].notna()
    buy_valor = pd.to_numeric(rows["buy_valor"], errors="coerce").to_numpy(float)
    sell_valor = pd.to_numeric(rows["sell_valor"], errors="coerce").to_numpy(float)
    # Source fees are never costs (ADR-27); a named sensitivity may explicitly inject synthetic costs.
    entry_fee = (
        pd.to_numeric(rows["entry_fee"], errors="coerce").fillna(0.0).to_numpy(float)
        if allow_fee_stress
        else np.zeros(len(rows), dtype=float)
    )
    exit_fee = (
        pd.to_numeric(rows["exit_fee"], errors="coerce").fillna(0.0).to_numpy(float)
        if allow_fee_stress
        else np.zeros(len(rows), dtype=float)
    )
    tax_category = rows["tax_category"].astype("string").str.strip()
    valid_tax_category = (tax_category.notna() & tax_category.ne("")).to_numpy(bool)
    tax_category_values = tax_category.astype(object).where(tax_category.notna() & tax_category.ne(""), None).to_numpy()
    valid_valor = (
        np.isfinite(buy_valor)
        & np.isfinite(sell_valor)
        & (buy_valor >= 0)
        & (sell_valor >= 0)
        & (buy_valor == np.floor(buy_valor))
        & (sell_valor == np.floor(sell_valor))
    )
    valid_fee = np.isfinite(entry_fee) & np.isfinite(exit_fee) & (entry_fee >= 0) & (exit_fee >= 0)
    valid_execution = present.to_numpy(bool) & valid_valor & valid_fee
    valid = valid_execution & valid_tax_category
    can_buy = rows["can_buy"].fillna(False).to_numpy(bool) & valid
    can_sell = rows["can_sell"].fillna(False).to_numpy(bool) & valid

    source_buy_reason = rows["buy_reason"].fillna("no_execution_profile").astype(object).to_numpy()
    source_sell_reason = rows["sell_reason"].fillna("no_execution_profile").astype(object).to_numpy()

    def _reason(source: np.ndarray, allowed: np.ndarray, direction: str) -> np.ndarray:
        unknown_source = np.isin(source, ["ok", "no_execution_profile"])
        missing_execution = np.where(unknown_source, "missing_execution_profile", source)
        missing_category = np.where(unknown_source, "missing_tax_category", source)
        blocked_direction = np.where(unknown_source, f"can_{direction}_false", source)
        return np.where(
            ~valid_execution,
            missing_execution,
            np.where(
                ~valid_tax_category,
                missing_category,
                np.where(allowed, "ok", blocked_direction),
            ),
        )

    buy_reason = _reason(source_buy_reason, can_buy, "buy")
    sell_reason = _reason(source_sell_reason, can_sell, "sell")

    # Keep CASH_PROXY's static T+0 terms/category; risky lots use the decision-day category at purchase.
    daily_buy_valor = base.buy_valor.copy()
    daily_sell_valor = base.sell_valor.copy()
    daily_entry_fee = base.entry_fee.copy()
    daily_exit_fee = base.exit_fee.copy()
    daily_can_buy = base.can_buy.copy()
    daily_can_sell = base.can_sell.copy()
    daily_tax_category = (
        base.tax_category.copy() if base.tax_category is not None else np.full(len(base.codes), None, dtype=object)
    )
    daily_buy_valor[fund_positions] = np.where(valid_valor, buy_valor, 0).astype(int)
    daily_sell_valor[fund_positions] = np.where(valid_valor, sell_valor, 0).astype(int)
    daily_entry_fee[fund_positions] = entry_fee
    daily_exit_fee[fund_positions] = exit_fee
    daily_can_buy[fund_positions] = can_buy
    daily_can_sell[fund_positions] = can_sell
    daily_tax_category[fund_positions] = tax_category_values
    applied = replace(
        base,
        buy_valor=daily_buy_valor,
        sell_valor=daily_sell_valor,
        entry_fee=daily_entry_fee,
        exit_fee=daily_exit_fee,
        can_buy=daily_can_buy,
        can_sell=daily_can_sell,
        tax_category=daily_tax_category,
    )
    full_buy_reason = np.full(len(base.codes), "ok", dtype=object)
    full_sell_reason = np.full(len(base.codes), "ok", dtype=object)
    full_buy_reason[fund_positions] = buy_reason
    full_sell_reason[fund_positions] = sell_reason
    return applied, {"BUY": full_buy_reason, "SELL": full_sell_reason}


def _rebalance_to(
    led: Ledger,
    w_target: np.ndarray,
    price: np.ndarray,
    t: int,
    cfg: BacktestConfig,
    sellable: np.ndarray,
    buyable: np.ndarray,
    slot: tuple[int, np.ndarray] | None = None,
    val_price: np.ndarray | None = None,
    force: bool = False,
) -> float:
    """Hedef ağırlıklara sapma eşiğiyle geç: satışlar can_sell, alışlar can_buy maskesiyle (D05). Döner: işlem hacmi.

    F-03/F-04: val_price ile değerleme/eşik, price ile dolum. F-02: force=True satış eşiğini atlar.
    """
    val_price = price if val_price is None else val_price
    eq = led.equity(val_price, None, 0.0)
    cur_val = led.units * np.nan_to_num(val_price, nan=0.0)
    cur_w = cur_val / eq if eq > 0 else np.zeros_like(cur_val)
    diff = w_target - cur_w
    traded = 0.0
    # --- satışlar ---
    for i in np.where((diff < 0) & sellable)[0]:
        if not force:
            thr = (
                cfg.drift_threshold_taxable_sale
                if led.unrealized_gain_ratio(i, val_price[i]) > 0.5
                else cfg.drift_threshold
            )
            if -diff[i] < thr and w_target[i] > 0:  # küçük sapma, pozisyon korunuyor
                continue
        sell_val = min(-diff[i] * eq, cur_val[i]) if w_target[i] > 0 else cur_val[i]
        if sell_val <= cfg.min_trade_value:
            continue
        units = sell_val / price[i]
        f = led.sell(i, units, price[i], t)
        if f:
            traded += f.gross
    # --- alışlar (nakitle sınırlı; satış nakdi valör nedeniyle henüz gelmemiş olabilir) ---
    buy_idx = np.where((diff > 0) & buyable)[0]
    wants = np.array([diff[i] * eq if diff[i] >= cfg.drift_threshold or cur_w[i] == 0 else 0.0 for i in buy_idx])
    total = wants.sum()
    if total > 0:
        if slot is not None and total > max(led.cash, 0.0) + 1e-9:
            slot_i, slot_px = slot
            p_slot = float(slot_px[t])
            slot_val = float(led.units[slot_i]) * p_slot
            if slot_val > 0 and np.isfinite(p_slot) and p_slot > 0:
                need = min(total - max(led.cash, 0.0), slot_val)
                led.sell(slot_i, need / p_slot, p_slot, t)  # T+0 bozum, stopaj lot bazında (P02)
        scale = min(1.0, max(led.cash, 0.0) / total)
        for i, amt in zip(buy_idx, wants * scale, strict=True):
            if amt > cfg.min_trade_value:
                f = led.buy(i, amt, price[i], t)
                if f:
                    traded += f.gross
    return traded
