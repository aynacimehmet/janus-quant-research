"""Makro ingest: EVDS (TCMB REST) + TCMB faiz geçmişi → `macro` tablosu (series, date, value, available_from).

PIT disiplini: `available_from = date + release_lag_days` — sinyaller yalnızca available_from ≤ karar tarihi olan
gözlemleri görür (aylık TÜFE ay sonundan ~5 gün sonra açıklanır; muhafazakâr gecikme konfigürasyondadır).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Protocol

import numpy as np
import pandas as pd
from loguru import logger

from janus.data.store import Store


class MacroFetcher(Protocol):
    def evds(self, codes: list[str], period: str, frequency: str) -> pd.DataFrame: ...
    def policy_rate_history(self, period: str) -> pd.Series: ...


class BorsapyMacro:
    """Gerçek çekici. EVDS anahtarı Settings'ten borsapy'ye verilir (os.environ'a yazılmaz)."""

    def __init__(self, evds_api_key: str = ""):
        import borsapy as bp  # noqa: PLC0415

        self._bp = bp
        if evds_api_key:
            bp.set_evds_key(evds_api_key)

    def evds(self, codes: list[str], period: str, frequency: str) -> pd.DataFrame:
        return self._bp.evds_download(codes, period=period, frequency=frequency)

    def policy_rate_history(self, period: str) -> pd.Series:
        df = self._bp.TCMB().history("policy", period=period)
        s = pd.to_numeric(df["lending"], errors="coerce")
        s.index = pd.to_datetime(df.index)
        return s.dropna()


_FREQ_CODE = {
    "daily": 1,
    "business": 2,
    "weekly": 3,
    "semimonthly": 4,
    "monthly": 5,
    "quarterly": 6,
    "semiannual": 7,
    "annual": 8,
}


class EvdsRestMacro:
    """borsapy'siz yedek: EVDS3 REST (https://evds3.tcmb.gov.tr/igmevdsms-dis/series=...&type=json), anahtar HTTP header'da.
    Kaynak: EVDS Python Kılavuzu (docId=4). Politika faizi için EVDS kodu konfigürasyonla verilir."""

    BASE = "https://evds3.tcmb.gov.tr/igmevdsms-dis/"

    def __init__(self, api_key: str, policy_code: str = "TP.APIFON4", timeout: int = 30):
        if not api_key:
            raise ValueError("EVDS_API_KEY boş")
        self.key = api_key
        self.policy_code = policy_code
        self.timeout = timeout

    @staticmethod
    def _dates(period: str) -> tuple[str, str]:
        end = pd.Timestamp.today().normalize()
        n = int("".join(ch for ch in period if ch.isdigit()) or 5)
        unit = period[-1]
        start = end - (pd.DateOffset(years=n) if unit == "y" else pd.DateOffset(months=n))
        return start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y")

    @staticmethod
    def parse(payload: dict, codes: list[str]) -> pd.DataFrame:
        """EVDS JSON → Date indeksli geniş tablo (kolon adı = orijinal kod). Kolonlar JSON'da '.' yerine '_' ile gelir."""
        items = payload.get("items", [])
        df = pd.DataFrame(items)
        if df.empty:
            return pd.DataFrame(columns=codes)
        out = pd.DataFrame(index=pd.to_datetime(df["Tarih"], format="mixed", dayfirst=True))
        out.index.name = "Date"
        for c in codes:
            col = c.replace(".", "_")
            out[c] = pd.to_numeric(df[col].to_numpy(), errors="coerce") if col in df.columns else np.nan
        return out.sort_index()

    def evds(self, codes: list[str], period: str, frequency: str) -> pd.DataFrame:
        import requests  # noqa: PLC0415

        start, end = self._dates(period)
        url = f"{self.BASE}series={'-'.join(codes)}&startDate={start}&endDate={end}&type=json&frequency={_FREQ_CODE.get(frequency, 1)}"
        r = requests.get(url, headers={"key": self.key}, timeout=self.timeout)
        r.raise_for_status()
        return self.parse(r.json(), codes)

    def policy_rate_history(self, period: str) -> pd.Series:
        wide = self.evds([self.policy_code], "10y" if period == "max" else period, "daily")
        s = wide[self.policy_code].dropna()
        return s[s.ne(s.shift())]  # yalnızca değişim günleri (adım fonksiyonu)


class FakeMacro:
    def __init__(self, today: str = "2026-09-22", seed: int = 0):
        self.today = pd.Timestamp(today)
        self.rng = np.random.default_rng(seed)

    def evds(self, codes: list[str], period: str, frequency: str) -> pd.DataFrame:
        if frequency == "monthly":
            idx = pd.date_range(end=self.today.replace(day=1), periods=24, freq="MS")
        else:
            idx = pd.bdate_range(end=self.today, periods=300)
        data = {c: 100 * np.exp(np.cumsum(self.rng.normal(0.001, 0.005, len(idx)))) for c in codes}
        return pd.DataFrame(data, index=pd.Index(idx, name="Date"))

    def policy_rate_history(self, period: str) -> pd.Series:
        idx = pd.to_datetime(["2025-01-23", "2025-06-19", "2026-03-12", "2026-07-24"])
        return pd.Series([45.0, 46.0, 40.0, 38.0], index=idx)


def _to_long(wide: pd.DataFrame, name_by_code: dict[str, str], lag_days: int, published_at: datetime) -> pd.DataFrame:
    long = wide.rename(columns=name_by_code).stack(future_stack=True).rename("value").reset_index()
    long.columns = ["date", "series", "value"]
    long["date"] = pd.to_datetime(long["date"]).dt.date
    long = long.dropna(subset=["value"])
    long["available_from"] = (pd.to_datetime(long["date"]) + pd.Timedelta(days=lag_days)).dt.date
    long["published_at"] = published_at
    return long[["series", "date", "value", "available_from", "published_at"]]


def _try(fetchers: list[MacroFetcher], call: str, *args: Any) -> tuple[Any, list[str]]:
    """Çekicileri sırayla dener; ilk başarılıyı döner, denenen hataları listeler."""
    errors: list[str] = []
    for f in fetchers:
        try:
            return getattr(f, call)(*args), errors
        except Exception as ex:  # noqa: BLE001
            errors.append(f"{type(f).__name__}:{type(ex).__name__}")
    raise RuntimeError("; ".join(errors) or "çekici yok")


def ingest_macro(store: Store, fetcher: MacroFetcher | list[MacroFetcher], cfg: dict[str, Any]) -> dict[str, Any]:
    """config `macro.series` girişlerini çeker; her seri ayrı try/except (biri düşerse diğerleri yazılır).
    `fetcher` bir liste ise sıralı yedek: borsapy düşerse EVDS REST denenir."""
    fetchers = list(fetcher) if isinstance(fetcher, list) else [fetcher]
    mcfg = cfg.get("macro", {})
    started = datetime.now()
    run_id = f"macro-{started:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    frames, failures, counts = [], [], {}

    # EVDS serilerini frekans ve pencereye göre grupla → tek POST
    evds_entries = [e for e in mcfg.get("series", []) if e.get("source", "evds") == "evds"]
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in evds_entries:
        groups.setdefault((e.get("frequency", "daily"), e.get("period", "5y")), []).append(e)
    for (freq, period), entries in groups.items():
        codes = [e["code"] for e in entries]
        try:
            wide, tried = _try(fetchers, "evds", codes, period, freq)
            if tried:
                logger.warning("EVDS birincil çekici düştü, yedek kullanıldı: {}", tried)
            for e in entries:
                if e["code"] not in wide.columns:
                    failures.append(f"{e['name']}:missing")
                    continue
                long = _to_long(wide[[e["code"]]], {e["code"]: e["name"]}, int(e.get("release_lag_days", 0)), started)
                frames.append(long)
                counts[e["name"]] = len(long)
        except Exception as ex:  # noqa: BLE001
            failures.extend(f"{e['name']}:{type(ex).__name__}" for e in entries)

    for e in (x for x in mcfg.get("series", []) if x.get("source") == "tcmb_policy"):
        try:
            s, _ = _try(fetchers, "policy_rate_history", e.get("period", "max"))
            wide = s.to_frame(e["name"])
            long = _to_long(wide, {e["name"]: e["name"]}, int(e.get("release_lag_days", 0)), started)
            frames.append(long)
            counts[e["name"]] = len(long)
        except Exception as ex:  # noqa: BLE001
            failures.append(f"{e['name']}:{type(ex).__name__}")

    df = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["series", "date", "value", "available_from", "published_at"])
    )
    n = store.upsert_macro(df)
    summary = {
        "run_id": run_id,
        "n_rows": int(n),
        "counts": counts,
        "failures": failures,
        "duration_s": round((datetime.now() - started).total_seconds(), 1),
    }
    status = "ok" if not failures else ("partial" if n > 0 else "failed")
    store.log_run(run_id, "ingest_macro", started, status, summary)
    logger.info("macro ingest: {} | {} satır | hatalar: {}", status, n, failures)
    return {"status": status, **summary}
