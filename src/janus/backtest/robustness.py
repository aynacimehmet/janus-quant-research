"""Sağlamlık: başlangıç tarihi taraması, maliyet/valör stresi, askı senaryosu, parametre taraması + PBO (CSCV) + deflated Sharpe."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pandas as pd

from janus.backtest.costs import FundMeta
from janus.backtest.engine import BacktestConfig, BacktestResult, rebalance_days, run_backtest
from janus.backtest.ledger import Ledger
from janus.backtest.metrics import deflated_sharpe, summary


def _stats(res: BacktestResult, equity: pd.Series | None = None) -> dict:
    """Özet metrikler; equity=None → res.equity, askı senaryosunda res.equity_stress (tasfiye değeri)."""
    m = summary(res.equity if equity is None else equity, res.cash_index, res.taxes_paid, res.fees_paid, res.turnover)
    return {
        "cagr": m["cagr"],
        "excess_cagr": m["excess_cagr"],
        "sharpe": m["sharpe"],
        "mdd": m["mdd"],
        "turnover": m["turnover_per_year"],
    }


def start_date_sweep(
    nav: pd.DataFrame,
    meta: FundMeta,
    strategy_factory,
    cfg: BacktestConfig,
    cash_returns: pd.Series,
    n_starts: int = 12,
    cash_nav: pd.Series | None = None,
    cash_category: str | None = None,
    cash_rate: float | None = None,
    execution_meta_by_date: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Isınma sonrası ilk n_starts ay başından başlayan koşular → dağılım (min/medyan/maks)."""
    cal = nav.index
    firsts = rebalance_days(cal, "monthly_first_business_day")
    firsts = [i for i in firsts if i >= cfg.warmup_days][:n_starts]
    rows = []
    for i in firsts:
        sub_cfg = replace(cfg, warmup_days=0)
        res = run_backtest(
            nav.iloc[i - cfg.warmup_days :],
            meta,
            strategy_factory(),
            replace(sub_cfg, warmup_days=cfg.warmup_days),
            cash_returns=cash_returns,
            cash_nav=cash_nav,
            cash_category=cash_category,
            cash_rate=cash_rate,
            execution_meta_by_date=execution_meta_by_date,
        )
        rows.append({"start": cal[i], **_stats(res)})
    return pd.DataFrame(rows).set_index("start")


def stressed_meta(meta: FundMeta, extra_fee: float = 0.0, valor_plus: int = 0) -> FundMeta:
    return replace(
        meta,
        entry_fee=meta.entry_fee + extra_fee,
        exit_fee=meta.exit_fee + extra_fee,
        sell_valor=meta.sell_valor + valor_plus,
    )


def stressed_execution_meta(
    execution_meta_by_date: pd.DataFrame | None, extra_fee: float = 0.0, valor_plus: int = 0
) -> pd.DataFrame | None:
    """Apply execution stress to the daily profiles as well as the legacy static FundMeta."""
    if execution_meta_by_date is None:
        return None
    stressed = execution_meta_by_date.copy()
    if extra_fee:
        stressed["entry_fee"] = pd.to_numeric(stressed["entry_fee"], errors="coerce") + extra_fee
        stressed["exit_fee"] = pd.to_numeric(stressed["exit_fee"], errors="coerce") + extra_fee
    if valor_plus:
        stressed["sell_valor"] = pd.to_numeric(stressed["sell_valor"], errors="coerce") + valor_plus
    return stressed


def b0_fund_valor_stress(
    nav: pd.DataFrame,
    codes: list[str],
    meta: FundMeta,
    execution_meta_by_date: pd.DataFrame | None,
    select_end: str | pd.Timestamp,
    initial_capital: float = 100.0,
    valor_plus: int = 1,
    synthetic_liquidation_idx: int | None = None,
    cash_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Report terminal B0 execution stress and optional explicit synthetic reinvestment sensitivity.

    This is an execution sensitivity, not the index-proxy B0 backtest or a claim about historical
    proposals. Membership is supplied by the caller; this helper never chooses or rebalances funds.
    The default mode retains synthetic period-end liquidation, so settlement timing is shown even
    when receivables keep end-of-period equity (and therefore CAGR) unchanged. The optional mode
    requires an explicit pre-terminal liquidation index and cash-return series.
    """
    columns = [
        "mode",
        "scenario",
        "period",
        "cagr",
        "status",
        "settle_idx",
        "event_note",
        "assumption_label",
        "fees_paid",
        "taxes_paid",
        "remaining_lots",
        "units_available_at_liquidation",
        "sellable_units",
        "settled_cash",
        "equity_identity_max_abs",
        "receivables_final",
        "position_value_final",
        "ending_equity",
    ]
    scenarios = ("fon-baz", "satış valörü +1")
    periods = ("full", "select", "external")
    if (synthetic_liquidation_idx is None) != (cash_returns is None):
        raise ValueError("sentetik yeniden yatırım için synthetic_liquidation_idx ve cash_returns birlikte gerekir")

    def unavailable(reason: str, assumption_label: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "scenario": scenario,
                    "mode": "fixed_terminal",
                    "period": period,
                    "cagr": np.nan,
                    "status": f"değerlendirilemedi: {reason}",
                    "settle_idx": pd.NA,
                    "event_note": "olay yok/gözlenemedi",
                    "assumption_label": assumption_label,
                    "fees_paid": np.nan,
                    "taxes_paid": np.nan,
                    "remaining_lots": pd.NA,
                    "units_available_at_liquidation": np.nan,
                    "sellable_units": np.nan,
                    "settled_cash": np.nan,
                    "equity_identity_max_abs": np.nan,
                    "receivables_final": np.nan,
                    "position_value_final": np.nan,
                    "ending_equity": np.nan,
                }
                for scenario in scenarios
                for period in periods
            ],
            columns=columns,
        )

    cal = pd.DatetimeIndex(nav.index)
    assumption_label = (
        "A3 varsayımlı, PIT kanıtı değil"
        if len(cal) and cal.min().normalize() < pd.Timestamp("2026-09-22")
        else "PIT yürütme profili"
    )
    if not codes:
        return unavailable("sabit B0 fon üyeliği verilmedi", assumption_label)
    if len(set(codes)) != len(codes) or any(code not in nav.columns for code in codes):
        return unavailable("fon üyeleri NAV panelinde tekil/tam değil", assumption_label)
    fund_nav = nav.loc[:, codes].apply(pd.to_numeric, errors="coerce")
    if fund_nav.empty or not np.isfinite(fund_nav.to_numpy(float)).all() or fund_nav.le(0).any().any():
        return unavailable("sabit üyelerde tam tarihçe NAV yok", assumption_label)
    if execution_meta_by_date is None:
        return unavailable("as-of yürütme profili yok", assumption_label)
    if not isinstance(execution_meta_by_date.index, pd.MultiIndex) or list(execution_meta_by_date.index.names) != [
        "decision_date",
        "fund_code",
    ]:
        return unavailable("yürütme profili as-of indeksinde değil", assumption_label)

    positions = pd.Index(meta.codes).get_indexer(codes)
    if (positions < 0).any():
        return unavailable("fon metası eksik", assumption_label)
    base_meta = replace(
        meta,
        codes=np.asarray(codes, dtype=object),
        buy_valor=meta.buy_valor[positions].copy(),
        sell_valor=meta.sell_valor[positions].copy(),
        entry_fee=meta.entry_fee[positions].copy(),
        exit_fee=meta.exit_fee[positions].copy(),
        tax_rate=meta.tax_rate[positions].copy(),
        tax_unknown=meta.tax_unknown[positions].copy(),
        equity_intensive=meta.equity_intensive[positions].copy(),
        can_buy=meta.can_buy[positions].copy(),
        can_sell=meta.can_sell[positions].copy(),
        tax_category=meta.tax_category[positions].copy() if meta.tax_category is not None else None,
    )

    required = ["buy_valor", "sell_valor", "can_buy", "can_sell", "tax_category"]

    def profile_at(date: pd.Timestamp) -> tuple[pd.DataFrame | None, list[str]]:
        """Return the date's profile plus per-fund missing/invalid issue strings (empty if usable).

        Issues are diagnostic only: the caller keeps the portfolio-level fail-closed behaviour
        required by ADR-0023 (a missing/blocked fund cannot silently rebalance a fixed B0 basket).
        """
        keys = pd.MultiIndex.from_product([[date], codes], names=["decision_date", "fund_code"])
        profile = execution_meta_by_date.reindex(keys).droplevel("decision_date").reindex(codes)
        missing_columns = [col for col in required if col not in profile]
        if missing_columns:
            return None, [f"kolon yok: {', '.join(missing_columns)}"]
        issues: list[str] = []
        for code in codes:
            row = profile.loc[code]
            for col in ("buy_valor", "sell_valor"):
                value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
                if not np.isfinite(value) or value < 0 or value != np.floor(value):
                    issues.append(f"{code} eksik/geçersiz {col}")
            for col in ("can_buy", "can_sell"):
                if pd.isna(row[col]):
                    issues.append(f"{code} eksik {col}")
            category = row["tax_category"]
            if not (isinstance(category, str) and category.strip()):
                issues.append(f"{code} eksik/boş tax_category")
        if issues:
            return None, issues
        profile["tax_category"] = profile["tax_category"].map(str.strip)
        for col in ("buy_valor", "sell_valor"):
            profile[col] = pd.to_numeric(profile[col], errors="coerce").astype(int)
        # ADR-27: raw profile fees do not participate in TEFAS execution costs.
        profile["entry_fee"] = 0.0
        profile["exit_fee"] = 0.0
        return profile, []

    def profile_issue_note(when: str, date: pd.Timestamp, issues: list[str]) -> str:
        return f"{when} {date.date()}: {', '.join(issues)}" if issues else ""

    first_profile, first_issues = profile_at(cal[0])
    last_profile, last_issues = profile_at(cal[-1])
    if first_profile is None or last_profile is None:
        detail = "; ".join(
            note
            for note in (
                profile_issue_note("ilk", cal[0], first_issues),
                profile_issue_note("son", cal[-1], last_issues),
            )
            if note
        )
        return unavailable(f"ilk alım/son satış gününde profil eksik veya geçersiz [{detail}]", assumption_label)
    if not first_profile["can_buy"].astype(bool).all():
        return unavailable("ilk gün B0 üyelerinin tamamı alıma açık değil", assumption_label)
    if not last_profile["can_sell"].astype(bool).all():
        return unavailable("dönem sonu B0 üyelerinin tamamı bozumuna açık değil", assumption_label)

    def apply_profile(profile: pd.DataFrame, sell_plus: int) -> FundMeta:
        return replace(
            base_meta,
            buy_valor=profile["buy_valor"].to_numpy(dtype=int),
            sell_valor=profile["sell_valor"].to_numpy(dtype=int) + sell_plus,
            entry_fee=profile["entry_fee"].to_numpy(dtype=float),
            exit_fee=profile["exit_fee"].to_numpy(dtype=float),
            can_buy=profile["can_buy"].to_numpy(dtype=bool),
            can_sell=profile["can_sell"].to_numpy(dtype=bool),
            tax_category=profile["tax_category"].to_numpy(dtype=object),
        )

    report_rows: list[dict] = []
    for scenario, sell_plus in (("fon-baz", 0), ("satış valörü +1", valor_plus)):
        ledger = Ledger(apply_profile(first_profile, 0), cash=initial_capital, dates=cal.to_numpy())
        amount = initial_capital / len(codes)
        for i in range(len(codes)):
            if ledger.buy(i, amount, float(fund_nav.iloc[0, i]), 0) is None:
                return unavailable("ilk gün eşit ağırlıklı alım tamamlanamadı", assumption_label)

        equity = np.empty(len(cal), dtype=float)
        # Entry cost is zero under ADR-27; equity starts at pre-trade capital and tax/settlement remain.
        equity[0] = initial_capital
        for t in range(1, len(cal)):
            ledger.settle(t)
            equity[t] = ledger.equity(fund_nav.iloc[t].to_numpy(float))

        ledger.meta = apply_profile(last_profile, sell_plus)
        for i in range(len(codes)):
            units = ledger.units[i]
            if ledger.sellable_units(i, len(cal) - 1) + 1e-12 < units:
                return unavailable("dönem sonunda alım valörü dolmamış lot var", assumption_label)
            if ledger.sell(i, units, float(fund_nav.iloc[-1, i]), len(cal) - 1) is None:
                return unavailable("sentetik dönem sonu tasfiyesi tamamlanamadı", assumption_label)
        equity[-1] = ledger.equity(fund_nav.iloc[-1].to_numpy(float))
        settle_idx = max(ledger.receivables, default=len(cal) - 1)

        periods_idx = {
            "full": np.arange(len(cal)),
            "select": np.flatnonzero(cal <= pd.Timestamp(select_end)),
            "external": np.flatnonzero(cal > pd.Timestamp(select_end)),
        }
        for period, indices in periods_idx.items():
            period_eq = equity[indices]
            years = (len(period_eq) - 1) / 252
            period_independently_accounted = period == "full"
            cagr = (
                float((period_eq[-1] / period_eq[0]) ** (1 / years) - 1)
                if period_independently_accounted and len(period_eq) > 1 and years > 0
                else np.nan
            )
            report_rows.append(
                {
                    "scenario": scenario,
                    "mode": "fixed_terminal",
                    "period": period,
                    "cagr": cagr,
                    "status": (
                        "ok"
                        if period_independently_accounted
                        else "değerlendirilemedi: dönem sonu bağımsız tasfiye/yeniden giriş yok"
                    ),
                    "settle_idx": settle_idx,
                    "event_note": (
                        "sentetik dönem sonu tasfiye; öneri olayı değil; alacak equity içinde"
                        if period_independently_accounted
                        else "fon-lot dönem getirisi değildir; terminal lotlar select/external sınırında tasfiye edilip yeniden alınmadı"
                    ),
                    "assumption_label": assumption_label,
                    "fees_paid": ledger.fees_paid,
                    "taxes_paid": ledger.taxes_paid,
                    "remaining_lots": sum(bool(lots) for lots in ledger.lots),
                    "units_available_at_liquidation": float("nan"),
                    "sellable_units": float("nan"),
                    "settled_cash": ledger.cash,
                    "equity_identity_max_abs": float("nan"),
                    "receivables_final": ledger.receivable_total(),
                    "position_value_final": float(ledger.position_value(fund_nav.iloc[-1].to_numpy(float)).sum()),
                    "ending_equity": ledger.equity(fund_nav.iloc[-1].to_numpy(float)),
                }
            )

    if synthetic_liquidation_idx is not None and cash_returns is not None:
        liquidation_idx = int(synthetic_liquidation_idx)
        if liquidation_idx != synthetic_liquidation_idx or not 0 < liquidation_idx < len(cal) - 1:
            raise ValueError("synthetic_liquidation_idx 0 ile son NAV indeksi arasında olmalı")
        if not isinstance(cash_returns, pd.Series) or cash_returns.index.has_duplicates:
            raise ValueError("cash_returns sentetik duyarlılık için tekil indeksli pandas Series olmalı")
        aligned_returns = pd.to_numeric(cash_returns.reindex(cal), errors="coerce")
        return_values = aligned_returns.to_numpy(dtype=float)
        if not np.isfinite(return_values).all() or (return_values <= -1.0).any():
            raise ValueError("cash_returns tüm NAV tarihleri için sonlu ve -100%'den büyük olmalı")

        liquidation_profile, liquidation_issues = profile_at(cal[liquidation_idx])
        liquidation_status = "değerlendirilemedi: önceden belirlenen tasfiye gününde profil/bozum kapısı geçersiz"
        liquidation_note = profile_issue_note("tasfiye", cal[liquidation_idx], liquidation_issues)
        if liquidation_note:
            liquidation_status += f" [{liquidation_note}]"
        if liquidation_profile is None or not liquidation_profile["can_sell"].astype(bool).all():
            return pd.concat(
                [
                    pd.DataFrame(report_rows, columns=columns),
                    pd.DataFrame(
                        [
                            {
                                "mode": "synthetic_reinvestment",
                                "scenario": scenario,
                                "period": "full",
                                "cagr": np.nan,
                                "status": liquidation_status,
                                "settle_idx": pd.NA,
                                "event_note": "sentetik/varsayımsal yeniden yatırım olayı yok",
                                "assumption_label": f"sentetik duyarlılık; {assumption_label}",
                                "fees_paid": np.nan,
                                "taxes_paid": np.nan,
                                "remaining_lots": pd.NA,
                                "units_available_at_liquidation": np.nan,
                                "sellable_units": np.nan,
                                "settled_cash": np.nan,
                                "equity_identity_max_abs": np.nan,
                                "receivables_final": np.nan,
                                "position_value_final": np.nan,
                                "ending_equity": np.nan,
                            }
                            for scenario in scenarios
                        ],
                        columns=columns,
                    ),
                ],
                ignore_index=True,
            )

        for scenario, sell_plus in (("fon-baz", 0), ("satış valörü +1", valor_plus)):
            ledger = Ledger(apply_profile(first_profile, 0), cash=initial_capital, dates=cal.to_numpy())
            amount = initial_capital / len(codes)
            for i in range(len(codes)):
                if ledger.buy(i, amount, float(fund_nav.iloc[0, i]), 0) is None:
                    return unavailable("sentetik yeniden yatırım için ilk gün alımı tamamlanamadı", assumption_label)

            ledger.meta = apply_profile(liquidation_profile, sell_plus)
            equity = np.empty(len(cal), dtype=float)
            equity[0] = initial_capital
            identity_errors: list[float] = []
            available_at_sale = np.zeros(len(codes), dtype=float)
            settle_idx = liquidation_idx
            for t in range(1, len(cal)):
                settled_today = ledger.settle(t)
                if t == liquidation_idx:
                    available_at_sale = np.asarray(
                        [ledger.sellable_units(i, liquidation_idx) for i in range(len(codes))], dtype=float
                    )
                    units_to_sell = ledger.units.copy()
                    if (available_at_sale + 1e-12 < units_to_sell).any():
                        return unavailable("sentetik tasfiye gününde alım valörü dolmamış lot var", assumption_label)
                    for i, units in enumerate(units_to_sell):
                        if ledger.sell(i, float(units), float(fund_nav.iloc[t, i]), t) is None:
                            return unavailable("sentetik tasfiye tamamlanamadı", assumption_label)
                    settle_idx = max(ledger.receivables, default=liquidation_idx)
                if t > liquidation_idx and settled_today <= 0 and ledger.cash > 0:
                    # Counterfactual B0 reinvestment begins only on days after ledger settlement.
                    ledger.accrue_cash(float(return_values[t]))
                mark = ledger.equity(fund_nav.iloc[t].to_numpy(float))
                identity_value = (
                    ledger.cash
                    + ledger.receivable_total()
                    + float(np.dot(ledger.units, fund_nav.iloc[t].to_numpy(float)))
                )
                identity_errors.append(abs(mark - identity_value))
                equity[t] = mark

            years = (len(equity) - 1) / 252
            cagr = float((equity[-1] / equity[0]) ** (1 / years) - 1) if years > 0 else np.nan
            report_rows.append(
                {
                    "mode": "synthetic_reinvestment",
                    "scenario": scenario,
                    "period": "full",
                    "cagr": cagr,
                    "status": "ok",
                    "settle_idx": settle_idx,
                    "event_note": (
                        f"tasfiye idx={liquidation_idx}; Ledger.settle sonrası varsayımsal yeniden yatırım "
                        "(sentetik B0 getiri serisi; Ledger.accrue_cash vergili günlük nakit yaklaşımı); "
                        "üretim akışı/öneri değildir"
                    ),
                    "assumption_label": (
                        "ufuk-içi sentetik/varsayımsal yeniden yatırım; günlük vergili nakit yaklaşımı gerçek fon-lot "
                        "nakit getirisi değildir ve kağıt defter/canlı üretim sonucu değildir; gerçek tarihsel PPF "
                        f"getirisi veya alfa kanıtı değildir; {assumption_label}"
                    ),
                    "fees_paid": ledger.fees_paid,
                    "taxes_paid": ledger.taxes_paid,
                    "remaining_lots": sum(bool(lots) for lots in ledger.lots),
                    "units_available_at_liquidation": float(available_at_sale.sum()),
                    "sellable_units": float(sum(ledger.sellable_units(i, len(cal) - 1) for i in range(len(codes)))),
                    "settled_cash": ledger.cash,
                    "equity_identity_max_abs": max(identity_errors, default=0.0),
                    "receivables_final": ledger.receivable_total(),
                    "position_value_final": float(ledger.position_value(fund_nav.iloc[-1].to_numpy(float)).sum()),
                    "ending_equity": ledger.equity(fund_nav.iloc[-1].to_numpy(float)),
                }
            )
    return pd.DataFrame(report_rows, columns=columns)


def suspension_scenario(
    nav: pd.DataFrame,
    meta: FundMeta,
    strategy_factory,
    cfg: BacktestConfig,
    cash_returns: pd.Series,
    founders: pd.Series,
    months: int = 6,
    seeds: int = 5,
    cash_nav: pd.Series | None = None,
    cash_category: str | None = None,
    cash_rate: float | None = None,
    execution_meta_by_date: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rastgele bir tarihte, o an elde tutulan fonlardan birinin kurucusunun TÜM fonları `months` ay askıya alınır.

    Raporlama stress_value (haircut'lı tasfiye değeri) iledir (P05); DD/kapı official_value kullanır.
    """
    base = run_backtest(
        nav,
        meta,
        strategy_factory(),
        cfg,
        cash_returns=cash_returns,
        cash_nav=cash_nav,
        cash_category=cash_category,
        cash_rate=cash_rate,
        execution_meta_by_date=execution_meta_by_date,
    )
    if base.weights.empty:
        return pd.DataFrame()
    rows = [{"seed": -1, "scenario": "baz", **_stats(base)}]
    reb_dates = list(base.weights.index)
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        d = reb_dates[int(rng.integers(len(reb_dates) // 4, len(reb_dates)))]
        held = base.weights.loc[d]
        held = held[held > 0]
        if held.empty:
            continue
        code = str(rng.choice(held.index))
        founder = founders.get(code, code)
        codes = [c for c in nav.columns if founders.get(c, c) == founder]
        susp = pd.DataFrame(False, index=nav.index, columns=nav.columns)
        i0 = nav.index.get_loc(d)
        susp.iloc[i0 : i0 + 21 * months, [nav.columns.get_loc(c) for c in codes]] = True
        res = run_backtest(
            nav,
            meta,
            strategy_factory(),
            cfg,
            cash_returns=cash_returns,
            suspended=susp,
            cash_nav=cash_nav,
            cash_category=cash_category,
            cash_rate=cash_rate,
            execution_meta_by_date=execution_meta_by_date,
        )
        rows.append(
            {"seed": seed, "scenario": f"{founder} askı {months} ay ({len(codes)} fon, {d.date()})", **_stats(res)}
        )
    return pd.DataFrame(rows)


def pbo_cscv(returns: np.ndarray, n_blocks: int = 8) -> dict:
    """Probability of Backtest Overfitting (Bailey ve ark. 2017, CSCV). returns: T × K günlük log getiri (K konfigürasyon)."""
    T, K = returns.shape
    if K < 2 or T < n_blocks * 5:
        return {"pbo": float("nan"), "n_combos": 0}
    blocks = np.array_split(np.arange(T), n_blocks)
    half = n_blocks // 2
    below = 0
    logits = []
    combos = list(itertools.combinations(range(n_blocks), half))
    for train in combos:
        tr = np.concatenate([blocks[b] for b in train])
        te = np.concatenate([blocks[b] for b in range(n_blocks) if b not in train])
        sr_is = returns[tr].mean(0) / (returns[tr].std(0, ddof=1) + 1e-12)
        sr_oos = returns[te].mean(0) / (returns[te].std(0, ddof=1) + 1e-12)
        best = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[best]).mean()  # 0..1 (1 = OOS'ta da en iyi)
        w = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
        below += rank < 0.5
    return {"pbo": below / len(combos), "n_combos": len(combos), "mean_logit": float(np.mean(logits))}


def parameter_sweep(
    nav: pd.DataFrame,
    meta: FundMeta,
    factory,
    grid: dict,
    cfg: BacktestConfig,
    cash_returns: pd.Series,
    cash_nav: pd.Series | None = None,
    cash_category: str | None = None,
    cash_rate: float | None = None,
    execution_meta_by_date: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """factory(**params) → strateji. Tüm kombinasyonları koşar; PBO ve en iyinin deflated Sharpe'ını döner."""
    keys = list(grid)
    rows, rets = [], []
    for values in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, values, strict=True))
        cfg_over = {k[4:]: v for k, v in params.items() if k.startswith("cfg:")}  # örn. "cfg:schedule"
        sparams = {k: v for k, v in params.items() if not k.startswith("cfg:")}
        res = run_backtest(
            nav,
            meta,
            factory(**sparams),
            replace(cfg, **cfg_over),
            cash_returns=cash_returns,
            cash_nav=cash_nav,
            cash_category=cash_category,
            cash_rate=cash_rate,
            execution_meta_by_date=execution_meta_by_date,
        )
        r = np.log(res.equity).diff().dropna()
        rc = np.log(res.cash_index).diff().reindex(r.index).fillna(0.0)
        rets.append((r - rc).to_numpy())
        rows.append({**params, **_stats(res)})
    df = pd.DataFrame(rows)
    R = np.column_stack(rets)
    per_obs_sr = R.mean(0) / (R.std(0, ddof=1) + 1e-12)
    best = int(np.argmax(per_obs_sr))
    dsr = deflated_sharpe(
        float(per_obs_sr[best]),
        n_trials=R.shape[1],
        sr_var=float(per_obs_sr.var(ddof=1)) if R.shape[1] > 1 else 0.0,
        n_obs=R.shape[0],
        skew_=float(pd.Series(R[:, best]).skew()),
        kurt_=float(pd.Series(R[:, best]).kurt() + 3),
    )
    info = {"best": df.iloc[best].to_dict(), "deflated_sharpe_prob": dsr, **pbo_cscv(R)}
    return df, info
