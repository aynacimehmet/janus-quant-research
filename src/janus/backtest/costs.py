"""Fon meta verisi (valör, komisyon, vergi) — fund_master'dan backtest için sabit tablo.

Valör yönü: config `execution.valor_mapping` ile çevrilebilir (TEFAS çapraz kontrolü sonucu).
Bilinmeyen vergi kategorisi → muhafazakâr %17,5 ve `tax_unknown=True` bayrağı (asla 0 varsayılmaz).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from janus.data.quality import trade_status_masks

DEFAULT_UNKNOWN_TAX = 0.175


@dataclass(frozen=True)
class FundMeta:
    """Kolonlar fon sırasına hizalı numpy dizileri (engine iç döngüsünde hızlı erişim)."""

    codes: np.ndarray  # str
    buy_valor: np.ndarray  # int (iş günü) — alınan pay ne zaman satılabilir
    sell_valor: np.ndarray  # int (iş günü) — satış nakdi ne zaman gelir
    entry_fee: np.ndarray  # oran (0.01 = %1)
    exit_fee: np.ndarray  # oran
    tax_rate: np.ndarray  # stopaj oranı (satışta gerçekleşen kâr üzerinden)
    tax_unknown: np.ndarray  # bool
    equity_intensive: np.ndarray  # bool (pay senedi yoğun → %0)
    can_buy: np.ndarray  # bool (D05; bugünkü tefas_status → A3 'bugünkü politika')
    can_sell: np.ndarray  # bool
    tax_category: np.ndarray = None  # object; None → borsapy tablosu kullanılamaz (A5: %17,5)
    tax_schedule: tuple = ()  # P04 yedek/ayna: ((from_date, {kategori: oran, default: oran}), ...) tarih sıralı

    @property
    def index(self) -> pd.Index:
        return pd.Index(self.codes)


def build_fund_meta(fund_master: pd.DataFrame, codes: list[str] | pd.Index, cfg: dict, leg: str = "tefas") -> FundMeta:
    """fund_master (son snapshot) → FundMeta. Eksik alanlar muhafazakâr varsayılanla doldurulur.

    `leg`: "tefas" | "bes" — BES'te stopaj yok (tax_rate=0), valör config varsayımı (A8).
    """
    ex = cfg["legs"][leg].get("execution", {})
    mapping = ex.get("valor_mapping", {"buy": "buy_valor", "sell": "sell_valor"})
    fee_scale = float(ex.get("fee_scale", 0.01))  # fund_master'daki ücret yüzde ise 0.01
    fm = fund_master.drop_duplicates("fund_code").set_index("fund_code").reindex(codes)

    def col(name: str, default: float) -> np.ndarray:
        s = pd.to_numeric(fm[name], errors="coerce") if name in fm.columns else pd.Series(np.nan, index=fm.index)
        return s.fillna(default).to_numpy(dtype=float)

    buy_v = col(mapping["buy"], ex.get("default_buy_valor", 1)).astype(int)
    sell_v = col(mapping["sell"], ex.get("default_sell_valor", 2)).astype(int)
    if leg in {"tefas", "bes"}:
        # ADR-27: kaynak fee alanları metadata'dır; TEFAS/BES execution komisyonu yoktur.
        entry = np.zeros(len(fm), dtype=float)
        exit_ = np.zeros(len(fm), dtype=float)
    else:
        entry = col("entry_fee", 0.0) * fee_scale
        exit_ = col("exit_fee", 0.0) * fee_scale
    if leg == "bes":
        # BES'te stopaj yok (S5-5); bilinmeyen vergi kategorisi değil
        tax_arr = np.zeros(len(fm), dtype=float)
        unknown = np.zeros(len(fm), dtype=bool)
    else:
        tax = (
            pd.to_numeric(fm["withholding_rate"], errors="coerce")
            if "withholding_rate" in fm.columns
            else pd.Series(np.nan, index=fm.index)
        )
        unknown = tax.isna().to_numpy()
        tax_arr = tax.fillna(DEFAULT_UNKNOWN_TAX).to_numpy(dtype=float)
    eq = (
        (fm["tax_category"].astype(str) == "pay_senedi_yogun").to_numpy()
        if "tax_category" in fm.columns
        else np.zeros(len(fm), bool)
    )
    cat = (
        fm["tax_category"].astype(object).where(fm["tax_category"].notna(), None).to_numpy()
        if "tax_category" in fm.columns
        else np.full(len(fm), None, dtype=object)
    )
    sched_raw = cfg["legs"]["tefas"].get("tax", {}).get("withholding_schedule", []) or []
    sched = tuple(
        sorted(((pd.Timestamp(x["from"]).date(), dict(x.get("rates", {}))) for x in sched_raw), key=lambda x: x[0])
    )
    can_buy, can_sell = trade_status_masks(fm.reset_index())  # boş/NaN status → ikisi True (D05)
    return FundMeta(
        codes=np.asarray(list(codes), dtype=object),
        buy_valor=buy_v,
        sell_valor=sell_v,
        entry_fee=entry,
        exit_fee=exit_,
        tax_rate=tax_arr,
        tax_unknown=unknown,
        equity_intensive=eq,
        can_buy=can_buy.reindex(list(codes)).fillna(True).to_numpy(bool),
        can_sell=can_sell.reindex(list(codes)).fillna(True).to_numpy(bool),
        tax_category=cat,
        tax_schedule=sched,
    )


def meta_for_synthetic(
    codes: list[str],
    buy_valor: int = 1,
    sell_valor: int = 2,
    tax_rate: float = 0.175,
    no_buy: list[str] | tuple[str, ...] = (),
    no_sell: list[str] | tuple[str, ...] = (),
) -> FundMeta:
    n = len(codes)
    return FundMeta(
        codes=np.asarray(codes, dtype=object),
        buy_valor=np.full(n, buy_valor),
        sell_valor=np.full(n, sell_valor),
        entry_fee=np.zeros(n),
        exit_fee=np.zeros(n),
        tax_rate=np.full(n, tax_rate),
        tax_unknown=np.zeros(n, bool),
        equity_intensive=np.zeros(n, bool),
        can_buy=~np.isin(codes, list(no_buy)),
        can_sell=~np.isin(codes, list(no_sell)),
        tax_category=np.full(n, None, dtype=object),
    )


def as_date(d) -> date:
    return pd.Timestamp(d).date()
