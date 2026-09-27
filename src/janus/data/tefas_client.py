"""TEFAS veri istemcisi: borsapy üzerine ince adaptör + test için sahte istemci.

Keşif (borsapy 0.11.0, 22.09.2026):
- screen_funds(fund_type, limit)     → fonGetiriBazliBilgiGetir: TÜM fonlar tek çağrı; 'fund_type' kolonu = fonTurAciklama (şemsiye türü)
- management_fees(fund_type)         → fonYonetimBazliBilgiGetir: founder_code (kurucuKod), applied_fee, max_expense_ratio
- Fund(code).info                    → fonBilgiGetir + fonProfilBilgiGetir: isin, category (fonKategori), fund_size, investor_count,
                                       buy_valor/sell_valor, entry_fee/exit_fee, first/last_trading_time, tefas_status, kap_link;
                                       founder/manager alanları BOŞ (v0.11'de kaynak yok) → kurucu adı fon adından türetilir
- Fund(code).history(period="5y")    → fonFiyatBilgiGetir: azami 5 yıl; Date indeksli 'Price'
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd
from loguru import logger


class TefasClient(Protocol):
    def list_funds(self, fund_type: str = "YAT") -> pd.DataFrame: ...
    def management_fees(self, fund_type: str = "YAT") -> pd.DataFrame: ...
    def fund_info(self, code: str) -> dict[str, Any]: ...
    def history(self, code: str, period: str = "5y") -> pd.DataFrame: ...


class BorsapyClient:
    """Gerçek istemci. Her fon çağrısı arasında `sleep` saniye bekler; hata durumunda üstel geri çekilmeyle yeniden dener."""

    def __init__(self, sleep: float = 0.25, retries: int = 3):
        import borsapy as bp  # noqa: PLC0415  (yalnızca gerçek koşuda gerekli)

        self._bp = bp
        self.sleep = sleep
        self.retries = retries

    def _retry(self, fn: Callable[[], Any], what: str) -> Any:
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            try:
                out = fn()
                if self.sleep:
                    time.sleep(self.sleep)
                return out
            except Exception as e:  # noqa: BLE001 — sağlayıcı hataları çeşitli
                if attempt == self.retries:
                    raise
                logger.warning("{} deneme {}/{} başarısız: {}", what, attempt, self.retries, type(e).__name__)
                time.sleep(delay)
                delay *= 2

    def list_funds(self, fund_type: str = "YAT") -> pd.DataFrame:
        df = self._retry(lambda: self._bp.screen_funds(fund_type=fund_type, limit=100_000), "screen_funds")
        df = pd.DataFrame(df).rename(columns={"fund_type": "umbrella_type"})
        df["fund_class"] = fund_type
        return df

    def management_fees(self, fund_type: str = "YAT") -> pd.DataFrame:
        return pd.DataFrame(self._retry(lambda: self._bp.management_fees(fund_type=fund_type), "management_fees"))

    def fund_info(self, code: str) -> dict[str, Any]:
        return dict(self._retry(lambda: self._bp.Fund(code).info, f"info {code}"))

    def history(self, code: str, period: str = "5y") -> pd.DataFrame:
        return self._retry(lambda: self._bp.Fund(code).history(period=period), f"history {code}")


class FakeTefasClient:
    """Deterministik sentetik istemci (testler ve `--dry-run`). Ağ yok."""

    def __init__(self, n: int = 12, days: int = 400, seed: int = 0, today: str = "2026-09-22"):
        self.today = pd.Timestamp(today)
        rng = np.random.default_rng(seed)
        founders = [
            "POYRAZ PORTFÖY",
            "LODOS PORTFÖY",
            "POLARİS PORTFÖY",
            "KUZEY PORTFÖY",
            "GÜNDOĞAN PORTFÖY",
            "BATI PORTFÖY",
        ]
        tefas_umbrellas = [
            "Hisse Senedi Şemsiye Fonu",
            "Serbest Şemsiye Fonu",
            "Para Piyasası Şemsiye Fonu",
            "Değişken Şemsiye Fonu",
        ]
        emk_umbrellas = [
            "Para Piyasası Emeklilik Fonu",
            "Hisse Senedi Emeklilik Fonu",
            "Karma Emeklilik Fonu",
            "Devlet Katkısı Emeklilik Fonu",
        ]
        rows = []
        for i in range(n):
            f = founders[i % len(founders)]
            u = tefas_umbrellas[i % len(tefas_umbrellas)]
            code = f"F{i:02d}"
            rows.append(
                {
                    "fund_code": code,
                    "name": f"{f} {u.replace(' Şemsiye Fonu', '')} FONU {i}",
                    "umbrella_type": u,
                    "founder_code": f[:3],
                    "return_1y": float(rng.normal(30, 20)),
                }
            )
        self._list = pd.DataFrame(rows)
        emk_rows = []
        rng2 = np.random.default_rng(seed + 1)
        for i in range(max(6, n // 2)):
            f = founders[i % len(founders)]
            u = emk_umbrellas[i % len(emk_umbrellas)]
            code = f"E{i:02d}"
            emk_rows.append(
                {
                    "fund_code": code,
                    "name": f"{f} {u.replace(' Emeklilik Fonu', '')} EMEKLİLİK FONU {i}",
                    "umbrella_type": u,
                    "founder_code": f[:3],
                    "return_1y": float(rng2.normal(25, 15)),
                }
            )
        self._emk_list = pd.DataFrame(emk_rows)
        self._days = days
        self._rng = rng
        self._rng2 = rng2

    def list_funds(self, fund_type: str = "YAT") -> pd.DataFrame:
        src = self._emk_list if fund_type == "EMK" else self._list
        df = src[["fund_code", "name", "umbrella_type", "return_1y"]].copy()
        df["fund_class"] = fund_type
        return df

    def management_fees(self, fund_type: str = "YAT") -> pd.DataFrame:
        src = self._emk_list if fund_type == "EMK" else self._list
        df = src[["fund_code", "name", "founder_code"]].copy()
        df["applied_fee"] = 1.5
        df["max_expense_ratio"] = 2.0
        return df

    def fund_info(self, code: str) -> dict[str, Any]:
        if code.startswith("E"):
            src = self._emk_list.set_index("fund_code")
            i = int(code[1:])
            is_emk = True
        else:
            src = self._list.set_index("fund_code")
            i = int(code[1:])
            is_emk = False
            if i == 5:
                raise RuntimeError("simulated API error")
        row = src.loc[code]
        cat = row["umbrella_type"].replace(" Şemsiye Fonu", " Fonu").replace(" Emeklilik Fonu", " Fonu")
        if is_emk and "devlet katkısı" in row["umbrella_type"].lower():
            cat = "Devlet Katkısı Emeklilik Fonu"
        return {
            "fund_code": code,
            "name": row["name"],
            "isin": f"TRY{code}00001",
            "category": cat,
            "fund_size": 1e8 * (i + 1),
            "investor_count": 1000 * (i + 1),
            "risk_value": 4,
            "fund_class": "EMK" if is_emk else "YAT",
            "buy_valor": "1",
            "sell_valor": "2",
            "entry_fee": "0",
            "exit_fee": "0",
            "first_trading_time": "09:00",
            "last_trading_time": "13:30",
            "kap_link": "https://www.kap.org.tr/",
            "tefas_status": "İşlem Görüyor",
        }

    def history(self, code: str, period: str = "5y") -> pd.DataFrame:
        rng = self._rng2 if code.startswith("E") else self._rng
        i = int(code[1:])
        days = self._days if period == "5y" else 25
        if i == 7:
            days = 200  # genç fon (TEFAS)
        if code == "E05":
            days = 200  # genç fon (BES)
        idx = pd.bdate_range(end=self.today, periods=days)
        if code == "E03":
            idx = idx[:-3]  # 3 iş günüdür NAV yok → askıda
        price = 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=len(idx))))
        return pd.DataFrame({"Price": price, "FundSize": np.nan, "Investors": 0}, index=pd.Index(idx, name="Date"))
