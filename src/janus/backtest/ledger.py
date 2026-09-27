"""Defter: nakit, pozisyonlar (numpy), FIFO lotlar, valör alacakları, stopaj — backtest ve JanusEnv ortak çekirdeği.

Kurallar (TECH_RULES): alış D NAV'ında (sinyal D-1); satış nakdi D + sell_valor iş günü sonra; alınan pay
D + buy_valor iş günü sonra satılabilir; stopaj satışta gerçekleşen kâr × lot oranı; askıdaki fon satılamaz
(stres değerinde NAV × (1 − haircut)). Değer = nakit + alacaklar + Σ pay × NAV_eff.
P04: lot stopajı alım tarihinden — config yedek schedule, yoksa borsapy paket içi tablo, o da yoksa tek oran.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from janus.backtest.costs import FundMeta


@dataclass
class Lot:
    units: float
    cost_per_unit: float  # giriş komisyonu dahil
    bought_idx: int  # takvim indeksi
    tax_rate: float
    available_idx: int  # bu indeksten itibaren satılabilir


@dataclass
class Fill:
    idx: int
    code: str
    side: str  # BUY / SELL
    units: float
    price: float
    gross: float  # BUY: ödenen nakit; SELL: brüt hasılat (komisyon sonrası)
    fee: float
    tax: float
    realized_gain: float


@dataclass
class Ledger:
    meta: FundMeta
    cash: float
    cash_tax_rate: float = 0.175  # nakit/para piyasası getirisinin stopajı (yaklaşık)
    dates: np.ndarray = None  # takvim tarihleri (datetime64); P04 lot stopajı için
    units: np.ndarray = field(init=False)
    lots: list[list[Lot]] = field(init=False)
    receivables: dict[int, float] = field(default_factory=dict)  # settle_idx → tutar
    fills: list[Fill] = field(default_factory=list)
    taxes_paid: float = 0.0
    fees_paid: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.meta.codes)
        self.units = np.zeros(n)
        self.lots = [[] for _ in range(n)]

    # ---- değerleme --------------------------------------------------------------------------
    def receivable_total(self) -> float:
        return float(sum(self.receivables.values()))

    def position_value(self, nav: np.ndarray, suspended: np.ndarray | None = None, haircut: float = 0.0) -> np.ndarray:
        eff = np.nan_to_num(nav, nan=0.0)
        if suspended is not None and haircut > 0:
            eff = np.where(suspended, eff * (1.0 - haircut), eff)
        return self.units * eff

    def equity(self, nav: np.ndarray, suspended: np.ndarray | None = None, haircut: float = 0.0) -> float:
        return self.cash + self.receivable_total() + float(self.position_value(nav, suspended, haircut).sum())

    def weights(self, nav: np.ndarray, suspended: np.ndarray | None = None, haircut: float = 0.0) -> np.ndarray:
        eq = self.equity(nav, suspended, haircut)
        return self.position_value(nav, suspended, haircut) / eq if eq > 0 else np.zeros_like(self.units)

    # ---- gün başı / gün sonu -------------------------------------------------------------------
    def settle(self, idx: int) -> float:
        """idx gününde vadesi gelen satış alacaklarını nakde çevirir."""
        due = [k for k in self.receivables if k <= idx]
        amt = float(sum(self.receivables.pop(k) for k in due))
        self.cash += amt
        return amt

    def accrue_cash(self, r_cash: float) -> float:
        """Nakit (para piyasası fonu) getirisi, stopaj sonrası."""
        if self.cash <= 0 or not np.isfinite(r_cash):
            return 0.0
        gain = self.cash * r_cash * (1.0 - self.cash_tax_rate)
        self.cash += gain
        return gain

    # ---- işlemler --------------------------------------------------------------------------
    def sellable_units(self, i: int, idx: int) -> float:
        return float(sum(lot.units for lot in self.lots[i] if lot.available_idx <= idx))

    def _lot_tax_rate(self, i: int, idx: int) -> float:
        """P04: rate(lot) = withholding_rate(tax_category, purchase_date).

        Sıra: config yedek schedule (tarih × kategori) → borsapy paket içi tablo (ağsız) → meta.tax_rate.
        """
        cat = self.meta.tax_category[i] if self.meta.tax_category is not None else None
        d = pd.Timestamp(self.dates[idx]).date() if self.dates is not None else None
        rate = None
        for start, rates in self.meta.tax_schedule or ():
            if d is not None and d >= start:
                r = rates.get(cat, rates.get("default"))
                if r is not None and np.isfinite(float(r)):
                    rate = float(r)  # son eşleşen rejim
        if rate is not None:
            return rate
        if cat and d is not None:
            try:
                from borsapy.tax import get_withholding_tax_rate  # noqa: PLC0415

                r = get_withholding_tax_rate(cat, d)
                if r is not None and np.isfinite(float(r)):
                    return float(r)
            except Exception:  # noqa: BLE001 — tablo/kategori yoksa tek oran
                pass
        return float(self.meta.tax_rate[i])

    def unrealized_tax(self, nav: np.ndarray) -> float:
        """§3: tüm pozisyonlar bugün satılsa ödenecek stopaj (lot bazında, zarar mahsupsuz)."""
        tot = 0.0
        for i, lots in enumerate(self.lots):
            if not lots:
                continue
            p = float(np.nan_to_num(nav[i], nan=0.0)) * (1.0 - float(self.meta.exit_fee[i]))
            tot += sum(lot.tax_rate * max(lot.units * (p - lot.cost_per_unit), 0.0) for lot in lots)
        return float(tot)

    def unrealized_gain_ratio(self, i: int, price: float) -> float:
        """Satış vergi doğurur mu? (kârdaki lot payı) → sapma eşiği seçimi için."""
        tot = sum(lot.units for lot in self.lots[i])
        if tot <= 0:
            return 0.0
        gain_units = sum(lot.units for lot in self.lots[i] if price * (1 - self.meta.exit_fee[i]) > lot.cost_per_unit)
        return gain_units / tot

    def buy(self, i: int, amount: float, price: float, idx: int) -> Fill | None:
        """`amount` kadar nakit harcayarak alım (giriş komisyonu içinden). Pay D+buy_valor'da satılabilir."""
        if amount <= 0 or not np.isfinite(price) or price <= 0 or amount > self.cash + 1e-9:
            return None
        fee_rate = float(self.meta.entry_fee[i])
        cost_per_unit = price * (1.0 + fee_rate)
        units = amount / cost_per_unit
        fee = amount - units * price
        self.cash -= amount
        self.units[i] += units
        self.lots[i].append(
            Lot(units, cost_per_unit, idx, self._lot_tax_rate(i, idx), idx + int(self.meta.buy_valor[i]))
        )
        self.fees_paid += fee
        f = Fill(idx, str(self.meta.codes[i]), "BUY", units, price, amount, fee, 0.0, 0.0)
        self.fills.append(f)
        return f

    def sell(self, i: int, units: float, price: float, idx: int) -> Fill | None:
        """FIFO satış; stopaj gerçekleşen kâr üzerinden; nakit D+sell_valor'da gelir."""
        if units <= 0 or not np.isfinite(price) or price <= 0:
            return None
        units = min(units, self.sellable_units(i, idx))
        if units <= 1e-12:
            return None
        exit_rate = float(self.meta.exit_fee[i])
        remaining, gross, cost_total, tax, fee = units, 0.0, 0.0, 0.0, 0.0
        new_lots: list[Lot] = []
        for lot in self.lots[i]:
            if remaining <= 1e-12 or lot.available_idx > idx:
                new_lots.append(lot)
                continue
            take = min(lot.units, remaining)
            proceeds = take * price * (1.0 - exit_rate)
            fee += take * price * exit_rate
            cost = take * lot.cost_per_unit
            gain = proceeds - cost
            tax += lot.tax_rate * max(gain, 0.0)
            gross += proceeds
            cost_total += cost
            remaining -= take
            if lot.units - take > 1e-12:
                new_lots.append(
                    Lot(lot.units - take, lot.cost_per_unit, lot.bought_idx, lot.tax_rate, lot.available_idx)
                )
        self.lots[i] = new_lots
        self.units[i] -= units
        if self.units[i] < 1e-12:
            self.units[i] = 0.0
        net = gross - tax
        settle_idx = idx + int(self.meta.sell_valor[i])
        if settle_idx <= idx:
            self.cash += net
        else:
            self.receivables[settle_idx] = self.receivables.get(settle_idx, 0.0) + net
        self.taxes_paid += tax
        self.fees_paid += fee
        f = Fill(idx, str(self.meta.codes[i]), "SELL", units, price, gross, fee, tax, gross - cost_total)
        self.fills.append(f)
        return f
