"""Faz-1 evren filtresi (ADR-14): YAT, Serbest şemsiye hariç, PYŞ kara listesi, yaş, işlem durumu, askı."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_TR = str.maketrans({"İ": "i", "I": "ı", "Ş": "ş", "Ğ": "ğ", "Ü": "ü", "Ö": "ö", "Ç": "ç"})


def tr_fold(text: str) -> str:
    """Türkçe-duyarlı küçük harfe indirme: 'ALFA CAPİTAL' → 'alfa capital' (casefold 'İ' → 'i̇' yapar, eşleşme bozulur)."""
    return text.translate(_TR).lower()


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).map(tr_fold).str.replace(r"\s+", " ", regex=True).str.strip()


def _pat(items: list[str]) -> str:
    return "|".join(re.escape(tr_fold(x)) for x in items)


def bes_scope_excluded(
    fund_master: pd.DataFrame,
    patterns: list[str],
    *,
    include_umbrella: bool = True,
    include_category: bool = True,
) -> pd.Series:
    """EMK umbrella/category desen eşleşmesini döndürür (girdi indeksinde)."""
    if not patterns or "fund_class" not in fund_master.columns:
        return pd.Series(False, index=fund_master.index, dtype=bool)
    pattern = _pat(patterns)
    if not pattern:
        return pd.Series(False, index=fund_master.index, dtype=bool)
    emk = _norm(fund_master["fund_class"]).eq("emk")
    matches = pd.Series(False, index=fund_master.index)
    if include_umbrella and "umbrella_type" in fund_master:
        matches |= _norm(fund_master["umbrella_type"]).str.contains(pattern, regex=True, na=False)
    if include_category and "category" in fund_master:
        matches |= _norm(fund_master["category"]).str.contains(pattern, regex=True, na=False)
    return emk & matches


def bes_scope_mask(fund_master: pd.DataFrame, cfg: dict) -> pd.Series:
    """F13: BES (EMK) yapısal kapsam maskesi — tek EMK uygunluk politikasının tek kaynağı.

    Kapsar: fund_class=EMK, `legs.bes.exclude_umbrella_patterns`, `exclude_state_contribution_patterns`,
    TEFAS `exclude_umbrella_patterns` ("serbest") ve BES+TEFAS birleşik kurucu kara listesi/whitelist.

    Çıktı: tekilleştirilmiş `fund_code` indeksli bool Series. Yaş, `info_ok`, status ve askı/yön
    koşulları PIT-yürütme katmanında (`phase1_mask`, özellik `policy_today_excluded`) ayrıca eklenir;
    böylece ingest, özellik, tahmin ve plan aynı yapısal kapsamı paylaşır (PIT ayrımı korunur).
    """
    fm = fund_master.drop_duplicates("fund_code").set_index("fund_code")
    u = cfg["legs"]["bes"]["universe"]
    col = lambda c: fm[c] if c in fm.columns else pd.Series("", index=fm.index)  # noqa: E731

    cls = fm["fund_class"] if "fund_class" in fm.columns else col("fund_type")  # S0 uyumluluğu
    ok = _norm(cls).eq(tr_fold(str(u["fund_type"])))

    # BES kapsam desenleri (OKS/standart/başlangıç/devlet katkı/katılım katkı)
    ok &= ~bes_scope_excluded(fm.reset_index(), cfg["legs"]["bes"].get("exclude_umbrella_patterns", [])).set_axis(
        fm.index
    )
    spat = _pat(u.get("exclude_state_contribution_patterns", []))
    if spat:
        ok &= ~(
            _norm(col("name")).str.contains(spat, regex=True) | _norm(col("category")).str.contains(spat, regex=True)
        )

    # TEFAS "serbest" kapsam deseni BES'te de aynen uygulanır.
    tu = cfg["legs"]["tefas"]["universe"]
    pat = _pat(tu.get("exclude_umbrella_patterns", []))
    if pat:
        ok &= ~(
            _norm(col("umbrella_type")).str.contains(pat, regex=True)
            | _norm(col("category")).str.contains(pat, regex=True)
        )

    # BES'te TEFAS kurucu kara listesi aynen; whitelist de paylaşılır.
    blacklist = list({*(u.get("founder_blacklist", []) or []), *(tu.get("founder_blacklist") or [])})
    bpat = _pat(blacklist)
    if bpat:  # kurucu, yönetici VEYA fon adı (kurucu adı fon adında geçer)
        ok &= ~(
            _norm(col("founder")).str.contains(bpat, regex=True)
            | _norm(col("manager")).str.contains(bpat, regex=True)
            | _norm(col("name")).str.contains(bpat, regex=True)
        )
    wpat = _pat(u.get("founder_whitelist", []) or [])
    if wpat:
        ok &= _norm(col("founder")).str.contains(wpat, regex=True)

    return pd.Series(np.asarray(ok, dtype=bool), index=fm.index, name="bes_scope")


def phase1_mask(
    fund_master: pd.DataFrame,
    cfg: dict,
    leg: str = "tefas",
    asof: pd.Timestamp | None = None,
    suspended: pd.Series | None = None,
    can_buy: pd.Series | None = None,
) -> pd.Series:
    """Girdi: fund_master (fund_code, fund_class, umbrella_type, category?, founder, manager?, name,
    inception_date | first_nav_date, tefas_status?). Çıktı: fund_code indeksli bool Series.

    `leg`: "tefas" | "bes" — BES evreni fund_class=EMK + devlet katkısı dışarı + TEFAS kara listesi aynen;
    yapısal kapsam tek kaynak `bes_scope_mask`'ten gelir (F13), yaş/info/status/askı/yön ayrıca eklenir.
    """
    u = cfg["legs"][leg]["universe"]
    fm = fund_master.drop_duplicates("fund_code").set_index("fund_code")
    asof = pd.Timestamp.today().normalize() if asof is None else pd.Timestamp(asof)
    col = lambda c: fm[c] if c in fm.columns else pd.Series("", index=fm.index)  # noqa: E731

    if leg == "bes":
        # Tek EMK uygunluk politikası: yapısal kapsam phase1_mask ile aynı fonksiyondan.
        ok = bes_scope_mask(fm.reset_index(), cfg).set_axis(fm.index)
    else:
        cls = fm["fund_class"] if "fund_class" in fm.columns else col("fund_type")  # S0 uyumluluğu
        ok = _norm(cls).eq(tr_fold(str(u["fund_type"])))
        pat = _pat(u.get("exclude_umbrella_patterns", []))
        if pat:  # şemsiye türü VEYA kategori "serbest" içeriyorsa dışarı
            ok &= ~(
                _norm(col("umbrella_type")).str.contains(pat, regex=True)
                | _norm(col("category")).str.contains(pat, regex=True)
            )
        bpat = _pat(u.get("founder_blacklist", []) or [])
        if bpat:  # kurucu, yönetici VEYA fon adı (kurucu adı fon adında geçer)
            ok &= ~(
                _norm(col("founder")).str.contains(bpat, regex=True)
                | _norm(col("manager")).str.contains(bpat, regex=True)
                | _norm(col("name")).str.contains(bpat, regex=True)
            )
        wpat = _pat(u.get("founder_whitelist", []) or [])
        if wpat:
            ok &= _norm(col("founder")).str.contains(wpat, regex=True)
        # TEFAS status metni BES'te yok/boş olabilir; BEFAS durum bilgisi ayrıca ele alınır (A8)
        spat = _pat(u.get("exclude_status_patterns", []) or [])
        if spat:
            ok &= ~_norm(col("tefas_status")).str.contains(spat, regex=True)

    birth = fm["inception_date"] if "inception_date" in fm.columns else col("first_nav_date")
    age_days = (asof - pd.to_datetime(birth, errors="coerce")).dt.days
    ok &= age_days.fillna(-1).to_numpy() >= int(u.get("min_age_days", 365))

    if "info_ok" in fm.columns:  # profil (valör/komisyon/durum) bilinmeyen fon simüle edilemez → dışarı
        ok &= fm["info_ok"].fillna(False).astype(bool).to_numpy()

    if suspended is not None:  # resmî askı VEYA veri eskiliği (çağıran karar verir) → alım evreni dışı
        ok &= ~suspended.reindex(fm.index).fillna(False).astype(bool)

    if can_buy is not None:  # LEDGER §1: alım için can_buy
        ok &= can_buy.reindex(fm.index).fillna(True).astype(bool)

    return pd.Series(np.asarray(ok, dtype=bool), index=fm.index, name="in_universe")
