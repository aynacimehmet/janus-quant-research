import pandas as pd

from janus.data.ingest_tefas import ingest_tefas
from janus.data.store import Store
from janus.data.tefas_client import FakeTefasClient
from janus.portfolio.universe import phase1_mask


class FlakyClient(FakeTefasClient):
    """İkinci koşuda F00 profili düşer; F05 her zaman düşer (FakeTefasClient)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.calls = 0

    def fund_info(self, code):
        if code == "F00" and self.calls >= 1:
            raise RuntimeError("flaky")
        return super().fund_info(code)


def test_carry_forward_and_weekly_retry(tmp_path, cfg):
    st = Store(tmp_path / "t.duckdb")
    client = FlakyClient(n=12, days=300)
    r1 = ingest_tefas(st, client, cfg, mode="initial", snapshot_date="2026-09-22")
    assert r1["failures_info"] == ["F05:RuntimeError"]
    client.calls = 1
    r2 = ingest_tefas(st, client, cfg, mode="incremental", snapshot_date="2026-09-23")
    fm = st.latest_fund_master().set_index("fund_code")
    # F00 profili bugün alınamadı ama dünden taşındı: info_ok True, info_fetched False, isin dolu
    assert (
        bool(fm.loc["F00", "info_ok"])
        and not bool(fm.loc["F00", "info_fetched"])
        and fm.loc["F00", "isin"] == "TRYF0000001"
    )
    # F05 hiç profil alamadı → info_ok False → evren dışı (zaten serbest; kural ayrıca test ediliyor)
    assert not bool(fm.loc["F05", "info_ok"])
    # Pazar dışındaki günlerde F05 yeniden denenmez (SkippedUntilWeekly), F00 denendi ve düştü
    assert any(f.startswith("F05:Skipped") for f in r2["failures_info"]) and "F00:RuntimeError" in r2["failures_info"]


def test_weekly_retry_follows_snapshot_date_not_wall_clock(tmp_path, cfg):
    """Retry günü run/as-of tarihinden (snapshot_date) türetilir; makine saatinden bağımsız olmalı.

    RED (düzeltme öncesi): karar `datetime.now()` ile veriliyordu → hangi günün retry olduğu makine saatine
    göre kayıyordu. GREEN (sonrası): 2026-09-26 Cumartesi (retry günü değil) → yeni deneme yok;
    2026-09-27 Pazar (retry günü) → yeni deneme var.
    """
    st = Store(tmp_path / "t.duckdb")
    client = FakeTefasClient(n=12, days=300)  # F05 profili her zaman hata verir
    # İlk koşu (Salı, retry günü değil): F05 profili hiç alınamaz
    r1 = ingest_tefas(st, client, cfg, mode="initial", snapshot_date="2026-09-22")
    assert r1["failures_info"] == ["F05:RuntimeError"]

    # Retry günü olmayan Cumartesi → F05 yeniden denenmez
    r_sat = ingest_tefas(st, client, cfg, mode="incremental", snapshot_date="2026-09-26")
    assert any(f.startswith("F05:Skipped") for f in r_sat["failures_info"])
    assert "F05:RuntimeError" not in r_sat["failures_info"]

    # Retry günü Pazar → F05 yeniden denenir
    r_sun = ingest_tefas(st, client, cfg, mode="incremental", snapshot_date="2026-09-27")
    assert "F05:RuntimeError" in r_sun["failures_info"]
    assert not any(f.startswith("F05:Skipped") for f in r_sun["failures_info"])


def test_universe_requires_known_profile(cfg):
    fm = pd.DataFrame(
        {
            "fund_code": ["K1", "K2"],
            "fund_class": ["YAT"] * 2,
            "umbrella_type": ["Hisse Senedi Şemsiye Fonu"] * 2,
            "category": ["Hisse Senedi Fonu"] * 2,
            "name": ["POYRAZ PORTFÖY X", "POYRAZ PORTFÖY Y"],
            "founder": ["POYRAZ PORTFÖY"] * 2,
            "tefas_status": ["TEFAS'ta işlem görüyor", None],
            "first_nav_date": ["2020-01-02"] * 2,
            "info_ok": [True, False],
        }
    )
    m = phase1_mask(fm, cfg, asof="2026-09-22")
    assert m["K1"] and not m["K2"]
