"""S5-5: BES ayağı testleri (EMK evreni, build, plan, sayaç)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import janus.cli as cli
import janus.features.macro as feat_macro
from janus.config import load_config
from janus.data.ingest_tefas import ingest_tefas
from janus.data.store import Store
from janus.data.tefas_client import FakeTefasClient
from janus.portfolio.universe import phase1_mask


def _bes_cfg():
    c = load_config()
    c["ingest"]["fund_types"] = ["EMK"]
    return c


def _make_macro(cal):
    import pandas as pd

    return pd.DataFrame(
        {
            "policy_rate": 0.4,
            "d_policy_63": 0.0,
            "cpi_yoy": 0.3,
            "real_rate": 0.1,
            "usdtry_ret63": 0.05,
            "usdtry_vol21": 0.1,
        },
        index=cal,
    )


def test_bes_universe_filters():
    cfg = _bes_cfg()
    st = Store(Path(__import__("tempfile").mkdtemp()) / "t.duckdb")  # noqa: PLW2901
    ingest_tefas(st, FakeTefasClient(n=12, days=400), cfg, mode="initial", snapshot_date="2026-09-22")
    fm = st.latest_fund_master()
    m = phase1_mask(fm, cfg, leg="bes", asof="2026-09-22")
    # E00 Hisse, E01 Karma, E02 Para Piyasası, E03 Devlet Katkısı
    assert m["E00"]  # PPF evrende
    assert "E03" not in set(fm["fund_code"])  # kapsam dışı EMK ingest'te çekilmedi
    assert not m["E05"]  # genç fon (<365 gün)


def test_bes_cash_proxy_required_no_fallback(tmp_path):
    cfg = _bes_cfg()
    # EMK evreninde PPF olmayan fon listesi (sadece Hisse/Karma)
    client = FakeTefasClient(n=12, days=400)
    client._emk_list = client._emk_list[
        ~client._emk_list["umbrella_type"].str.contains("Para Piyasası", case=False)
    ].copy()
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, client, cfg, mode="initial", snapshot_date="2026-09-22")
    from janus.cli import _features_build_impl

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    with pytest.raises(RuntimeError, match="BES nakit vekili yok"):
        _features_build_impl(st.nav_wide(), st.latest_fund_master(), cfg, st, tmp_path / "bes.parquet", leg="bes")
    monkeypatch.undo()


def test_bes_build_cli(tmp_path, monkeypatch):
    cfg = _bes_cfg()
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=400), cfg, mode="initial", snapshot_date="2026-09-22")
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: st)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    runner = CliRunner()
    r = runner.invoke(cli.app, ["bes", "build"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "data" / "features" / "bes_features.parquet").exists()
    assert (tmp_path / "data" / "predictions" / "bes_predictions.parquet").exists()
    assert (tmp_path / "data" / "predictions" / "bes_calibrated_target_020.parquet").exists()


def _force_selection(monkeypatch, chooser):
    """`janus.cli._select_impl`'i sahte seçimle değiştir; `chooser(call_idx, codes) -> picks`."""
    calls = {"n": 0}

    def fake_select(date, predictions, features, calibrated, out, cfg):
        d = pd.Timestamp(date).normalize()
        day = predictions[pd.to_datetime(predictions["decision_at"]) == d]
        codes = sorted(day["fund_code"].astype(str).unique().tolist())
        picks = set(chooser(calls["n"], codes))
        calls["n"] += 1
        n = len(codes)
        frame = pd.DataFrame(
            {
                "decision_at": [d] * n,
                "fund_code": codes,
                "p_value": [0.01] * n,
                "selected_q20": [c in picks for c in codes],
                "lower": [0.0] * n,
                "q50": [0.0] * n,
            }
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
        return {
            "status": "ok",
            "n_rows": n,
            "n_funds": n,
            "date": date,
            "n_selected_q20": len(picks),
            "q_grid": [0.2],
        }

    monkeypatch.setattr(cli, "_select_impl", fake_select)


def _last_month_dates(preds, k=4):
    dts = pd.to_datetime(preds["decision_at"]).dt.normalize().drop_duplicates().sort_values()
    months = dts.dt.to_period("M")
    return dts.groupby(months).max().sort_values().tail(k).tolist()


def test_bes_plan_cli(tmp_path, monkeypatch):
    cfg = _bes_cfg()
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=400), cfg, mode="initial", snapshot_date="2026-09-22")
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: st)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    runner = CliRunner()
    r1 = runner.invoke(cli.app, ["bes", "build"])
    assert r1.exit_code == 0, r1.output
    preds = pd.read_parquet(tmp_path / "data" / "predictions" / "bes_predictions.parquet")
    d = str(pd.to_datetime(preds["decision_at"]).max().date())
    r2 = runner.invoke(cli.app, ["bes", "plan", "--date", d])
    # F14: sentetikte seçim boş → varsayılan B0; hak harcanmaz ama gölge rapor yazılır.
    assert r2.exit_code == 0, r2.output
    assert "değişiklik yok" in r2.output
    assert st.con.execute("SELECT COUNT(*) FROM bes_plan").fetchone()[0] == 0
    md = tmp_path / "reports" / f"bes_plan_{d}.md"
    assert md.exists()
    assert "valör varsayımı" in md.read_text(encoding="utf-8")


def test_bes_plan_counter_limit(tmp_path, monkeypatch):
    cfg = _bes_cfg()
    cfg["legs"]["bes"]["plan"]["planned"] = 1
    cfg["legs"]["bes"]["plan"]["reserve"] = 0
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=400), cfg, mode="initial", snapshot_date="2026-09-22")
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    from janus.bes.build import bes_build
    from janus.bes.plan import bes_plan

    bes_build(st, cfg, tmp_path)
    preds = pd.read_parquet(tmp_path / "data" / "predictions" / "bes_predictions.parquet")
    dates = _last_month_dates(preds, k=2)
    # İlk çağrı riskli (Hisse) fon seçer; ikinci çağrı farklı dağılım → limiti aşar.
    _force_selection(monkeypatch, lambda i, codes: [codes[1]] if i == 0 else [codes[0]])
    r1 = bes_plan(st, cfg, dates[0].strftime("%Y-%m-%d"), root=tmp_path)
    assert r1["status"] == "ok"
    assert st.con.execute("SELECT COUNT(DISTINCT change_id) FROM bes_plan").fetchone()[0] == 1
    r2 = bes_plan(st, cfg, dates[1].strftime("%Y-%m-%d"), root=tmp_path)
    assert r2["status"] == "limit_reached"
    assert st.con.execute("SELECT COUNT(DISTINCT change_id) FROM bes_plan").fetchone()[0] == 1


def test_bes_plan_append_only_no_change_no_right(tmp_path, monkeypatch):
    """F14: aynı dağılım tekrar yazılmaz; eski satırlar silinmez; sessiz hak tüketimi yok."""
    cfg = _bes_cfg()
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=400), cfg, mode="initial", snapshot_date="2026-09-22")
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    from janus.bes.build import bes_build
    from janus.bes.plan import bes_plan

    bes_build(st, cfg, tmp_path)
    preds = pd.read_parquet(tmp_path / "data" / "predictions" / "bes_predictions.parquet")
    dates = _last_month_dates(preds, k=4)

    def chooser(i, codes):
        # codes[1] riskli (Hisse); codes[2] PPF üyesi → farklı dağılım/hak tüketimi.
        return [codes[2]] if i >= 3 else [codes[1]]

    _force_selection(monkeypatch, chooser)
    r1 = bes_plan(st, cfg, dates[0].strftime("%Y-%m-%d"), root=tmp_path)
    assert r1["status"] == "ok"
    first = st.con.execute("SELECT change_id, fund_code, target_weight FROM bes_plan ORDER BY fund_code").df()
    # Aynı ay aynı dağılım tekrar koşu → değişiklik yok, satır eklenmez/silinmez.
    r2 = bes_plan(st, cfg, dates[0].strftime("%Y-%m-%d"), root=tmp_path)
    assert r2["status"] == "no_change"
    second = st.con.execute("SELECT change_id, fund_code, target_weight FROM bes_plan ORDER BY fund_code").df()
    assert first.equals(second)
    # Farklı ay ama aynı hedef vektör → yine hak harcanmaz.
    r3 = bes_plan(st, cfg, dates[1].strftime("%Y-%m-%d"), root=tmp_path)
    assert r3["status"] == "no_change"
    assert st.con.execute("SELECT COUNT(DISTINCT change_id) FROM bes_plan").fetchone()[0] == 1
    # Farklı dağılım → yeni change_id, eski satırlar korunur (append-only).
    r4 = bes_plan(st, cfg, dates[2].strftime("%Y-%m-%d"), root=tmp_path)
    assert r4["status"] == "ok"
    assert st.con.execute("SELECT COUNT(DISTINCT change_id) FROM bes_plan").fetchone()[0] == 2
    assert st.con.execute("SELECT COUNT(*) FROM bes_plan WHERE change_id = ?", [r1["change_id"]]).fetchone()[0] > 0


def test_bes_plan_ppf_missing_fail_closed(tmp_path, monkeypatch):
    """F16: plan anında PPF yok → güvenli hata; hiçbir satır/rapor yazılmaz."""
    import janus.backtest.data as backtest_data
    from janus.bes.build import bes_build
    from janus.bes.plan import bes_plan

    cfg = _bes_cfg()
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=400), cfg, mode="initial", snapshot_date="2026-09-22")
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    bes_build(st, cfg, tmp_path)
    preds = pd.read_parquet(tmp_path / "data" / "predictions" / "bes_predictions.parquet")
    d = str(pd.to_datetime(preds["decision_at"]).max().date())
    monkeypatch.setattr(backtest_data, "cash_proxy_codes", lambda *args, **kwargs: [])
    res = bes_plan(st, cfg, d, root=tmp_path)
    assert res["status"] == "error"
    assert "nakit vekili yok" in res["message"]
    assert st.con.execute("SELECT COUNT(*) FROM bes_plan").fetchone()[0] == 0
    assert not (tmp_path / "reports" / f"bes_plan_{d}.md").exists()


def test_bes_plan_weights_caps_and_ppf_remainder():
    """F15: fon/PYŞ tavanları; artan PPF sepetine; kota bitince serbest nakit."""
    from janus.bes.plan import _allocate_targets

    cfg = _bes_cfg()
    founders = {"A": "F1", "P1": "F2", "P2": "F3"}
    # Tek riskli fon: %25 tavana takılır; PPF iki fon da %25 tavana takılır → %25 serbest nakit.
    w, residual = _allocate_targets(["A"], ["P1", "P2"], founders, cfg)
    assert abs(w["A"] - 0.25) < 1e-9
    assert abs(w["P1"] - 0.25) < 1e-9 and abs(w["P2"] - 0.25) < 1e-9
    assert abs(residual - 0.25) < 1e-9
    # Aynı PYŞ'ten iki riskli fon → PYŞ tavanı %30, her biri %15.
    founders2 = {"A": "F1", "B": "F1", "P1": "F2", "P2": "F3"}
    w2, residual2 = _allocate_targets(["A", "B"], ["P1", "P2"], founders2, cfg)
    assert abs(w2["A"] - 0.15) < 1e-9 and abs(w2["B"] - 0.15) < 1e-9
    assert w2["A"] + w2["B"] <= 0.30 + 1e-9
    assert sum(w2.values()) <= 1.0 + 1e-9
    # Tüm paylar tavan ve bütçe sınırları içinde.
    for group in ("F1",):
        assert sum(v for c, v in w2.items() if founders2[c] == group) <= 0.30 + 1e-9


def test_bes_scope_mask_single_policy():
    """F13: tek EMK kapsam maskesi devlet katkı + TEFAS kara listesini uygular; TEFAS sınıfını etkilemez."""
    from janus.portfolio.universe import bes_scope_mask, phase1_mask

    cfg = _bes_cfg()
    fm = pd.DataFrame(
        {
            "fund_code": ["X0", "X1", "X2", "X3"],
            "fund_class": ["EMK", "EMK", "EMK", "EMK"],
            "umbrella_type": ["Değişken", "Değişken", "Değişken", "Standart"],
            "category": ["Değişken", "Devlet Katkı", "Değişken", "OKS"],
            "name": ["Normal", "Devlet Katkı Fonu", "Kara Liste Fonu", "Standart Fon"],
            "founder": ["Kurucu", "Kurucu", "Kuzey Portföy", "Kurucu"],
            "manager": [""] * 4,
        }
    )
    mask = bes_scope_mask(fm, cfg)
    assert mask["X0"]  # normal EMK
    assert not mask["X1"]  # devlet katkı
    assert not mask["X2"]  # TEFAS kara listesi
    assert not mask["X3"]  # BES kapsam deseni (standart)

    # TEFAS/YAT sınıfı BES kapsam deseniyle elenmez (devlet katkı adı YAT'ta dışlama değildir).
    tefas = fm.copy()
    tefas["fund_class"] = "YAT"
    tefas["umbrella_type"] = "Değişken"
    tefas["category"] = "Değişken"
    tefas["founder"] = "Kurucu"
    tefas["name"] = "Devlet Katkı Fonu"
    tefas["inception_date"] = pd.Timestamp("2020-01-01")
    tefas["info_ok"] = True
    tefas_mask = phase1_mask(tefas, cfg, leg="tefas", asof="2026-09-22")
    assert tefas_mask["X1"]  # YAT "devlet katkı" adından dolayı dışlanmaz

    # phase1_mask(leg="bes") yapısal kapsamda tek maskeyle aynı sonucu verir (yaş/info ek katmanı hariç).
    fm_aged = fm.assign(inception_date=pd.Timestamp("2020-01-01"), info_ok=True)
    phase_bes = phase1_mask(fm_aged, cfg, leg="bes", asof="2026-09-22")
    for code in fm["fund_code"]:
        assert bool(phase_bes[code]) == bool(mask[code])


def test_bes_exclusion_patterns_ingest_and_universe(tmp_path):
    from janus.portfolio.universe import bes_scope_excluded

    cfg = _bes_cfg()
    cfg["legs"]["bes"]["exclude_umbrella_patterns"] = ["oks", "standart", "başlangıç", "devlet katkı", "katılım katkı"]
    client = FakeTefasClient(n=12, days=300)
    descriptors = [
        ("OKS Emeklilik Fonu", ""),
        ("Standart Emeklilik Fonu", ""),
        ("Başlangıç Emeklilik Fonu", ""),
        ("Emeklilik Değişken Fonu", "Devlet Katkı Fonu"),
        ("Emeklilik Değişken Fonu", "Katılım Katkı Fonu"),
        ("Değişken Fon", "Değişken Fon"),
        ("Para Piyasası Fonu", "Para Piyasası Fonu"),
    ]
    client._emk_list = pd.DataFrame(
        [
            {
                "fund_code": f"E{i:02d}",
                "name": f"Sentetik {i}",
                "umbrella_type": umbrella,
                "category": category,
                "founder_code": "F",
                "return_1y": 0.0,
            }
            for i, (umbrella, category) in enumerate(descriptors)
        ]
    )

    class RecordingClient(FakeTefasClient):
        def __init__(self, wrapped):
            self.__dict__.update(wrapped.__dict__)
            self.requested_info = []
            self.requested_history = []

        def list_funds(self, fund_type="YAT"):
            src = self._emk_list if fund_type == "EMK" else self._list
            cols = ["fund_code", "name", "umbrella_type", "return_1y"]
            out = src[cols + (["category"] if fund_type == "EMK" else [])].copy()
            out["fund_class"] = fund_type
            return out

        def management_fees(self, fund_type="YAT"):
            src = self._emk_list if fund_type == "EMK" else self._list
            return pd.DataFrame(
                {"fund_code": src["fund_code"], "founder_code": "F", "applied_fee": 0.0, "max_expense_ratio": 0.0}
            )

        def fund_info(self, code):
            self.requested_info.append(code)
            return {
                "fund_code": code,
                "name": f"Sentetik {code}",
                "category": self._emk_list.set_index("fund_code").loc[code, "category"],
                "fund_class": "EMK",
                "tefas_status": "İşlem Görüyor",
                "buy_valor": 1,
                "sell_valor": 2,
            }

        def history(self, code, period="5y"):
            self.requested_history.append(code)
            return pd.DataFrame({"Price": [10.0, 10.1]}, index=pd.bdate_range("2026-09-21", periods=2))

    synthetic_master = client._emk_list.copy()
    synthetic_master["fund_class"] = "EMK"
    synthetic_master["first_nav_date"] = pd.Timestamp("2025-01-01")
    synthetic_master["info_ok"] = True
    synthetic_master["founder"] = "Kurucu"
    synthetic_master["manager"] = ""
    synthetic_master["tefas_status"] = ""
    mask = phase1_mask(synthetic_master, cfg, leg="bes", asof="2026-09-22")
    assert set(mask[mask].index) == {"E05", "E06"}

    recording = RecordingClient(client)
    st = Store(tmp_path / "bes_scope.duckdb")
    ingest_tefas(st, recording, cfg, mode="initial", snapshot_date="2026-09-22")
    retained = {"E05", "E06"}
    assert set(recording.requested_info) == {"E03", "E04", *retained}
    assert set(recording.requested_history) == retained
    master = st.latest_fund_master()
    assert set(master["fund_code"]) == {"E03", "E04", *retained}
    mask = phase1_mask(master, cfg, leg="bes", asof="2026-09-22")
    assert set(mask[mask].index) <= retained

    # Kategori bilgisi snapshot'a girdikten sonra iki kategori eşleşmesi list aşamasında atlanır.
    recording.requested_info.clear()
    recording.requested_history.clear()
    ingest_tefas(st, recording, cfg, mode="incremental", snapshot_date="2026-09-23")
    assert set(recording.requested_info) == retained
    assert set(recording.requested_history) == retained

    # Aynı desenler YAT fonunu elemez; desen eşleşmeleri yalnızca EMK sınıfına uygulanır.
    tefas = pd.DataFrame(
        {"fund_code": ["T1"], "fund_class": ["YAT"], "umbrella_type": ["Standart Fon"], "category": ["OKS"]}
    )
    assert not bes_scope_excluded(tefas, cfg["legs"]["bes"]["exclude_umbrella_patterns"]).any()


def test_bes_features_exclude_preexisting_emk_from_outputs(tmp_path, monkeypatch):
    import janus.backtest.data as backtest_data
    import janus.features.fund_features as fund_features
    import janus.features.market as market_features
    from janus.cli import _features_build_impl
    from janus.features import macro as macro_module

    cfg = _bes_cfg()
    cfg["legs"]["bes"]["exclude_umbrella_patterns"] = ["oks", "standart", "başlangıç", "devlet katkı", "katılım katkı"]
    funds = ["E00", "E01", "E02", "E03", "E04", "E05", "E06"]
    descriptors = [
        "OKS Fon",
        "Standart Fon",
        "Başlangıç Fon",
        "Değişken Fon",
        "Değişken Fon",
        "Değişken Fon",
        "Para Piyasası Fon",
    ]
    categories = ["OKS", "Standart", "Başlangıç", "Devlet Katkı", "Katılım Katkı", "Değişken Fon", "Para Piyasası Fon"]
    fm = pd.DataFrame(
        {
            "fund_code": funds,
            "fund_class": "EMK",
            "umbrella_type": descriptors,
            "category": categories,
            "withholding_rate": 0.0,
        }
    )
    nav = pd.DataFrame({code: [10.0, 10.1] for code in funds}, index=pd.bdate_range("2026-09-23", periods=2))
    observed = {}
    monkeypatch.setattr(backtest_data, "cash_proxy_codes", lambda *args, **kwargs: ["E06"])
    monkeypatch.setattr(
        backtest_data, "cash_proxy_returns", lambda *args, **kwargs: pd.Series([0.001, 0.001], index=nav.index)
    )
    monkeypatch.setattr(backtest_data, "equity_index", lambda *args, **kwargs: pd.Series([1.0, 1.0], index=nav.index))
    monkeypatch.setattr(macro_module, "macro_features", lambda _store, cal: pd.DataFrame(index=cal))
    monkeypatch.setattr(market_features, "market_features", lambda *args: pd.DataFrame(index=nav.index))

    def build(panel, master, *args, **kwargs):
        observed["funds"] = set(master["fund_code"])
        n_funds = len(master)
        return pd.DataFrame(
            {
                "feature_asof": [nav.index[-1]] * n_funds,
                "fund_code": list(master["fund_code"]),
                "label_ready": [False] * n_funds,
                "eligible_at_decision": [True] * n_funds,
            }
        )

    monkeypatch.setattr(fund_features, "build_features", build)
    summary = _features_build_impl(
        nav, fm, cfg, Store(tmp_path / "existing.duckdb"), tmp_path / "bes_features.parquet", leg="bes"
    )
    assert observed["funds"] == {"E05", "E06"}
    assert summary["n_funds"] == 2


def test_bes_incremental_scope_change_fails_closed_no_prune(tmp_path, monkeypatch):
    """B1/ADR-0029/4: BES kapsam değişimi canonical'ı guard öncesi budamaz; fail-closed, dosya değişmez."""
    import janus.backtest.data as backtest_data
    import janus.features.fund_features as fund_features
    import janus.features.market as market_features
    from janus.cli import _features_build_impl
    from janus.features import macro as macro_module

    cfg = _bes_cfg()
    cfg["legs"]["bes"]["exclude_umbrella_patterns"] = ["oks"]
    fm = pd.DataFrame(
        {
            "fund_code": ["E00", "E05"],
            "fund_class": ["EMK", "EMK"],
            "umbrella_type": ["OKS", "Değişken"],
            "category": ["OKS", "Değişken"],
            "withholding_rate": [0.0, 0.0],
        }
    )
    nav = pd.DataFrame({c: 10.0 for c in ["E00", "E05"]}, index=pd.bdate_range("2026-06-01", periods=90))
    out = tmp_path / "bes_features.parquet"
    rows = [
        {"feature_asof": d, "fund_code": c, "label_ready": False, "eligible_at_decision": True}
        for d in nav.index[:60]
        for c in ["E00", "E05"]
    ]
    pd.DataFrame(rows).to_parquet(out, index=False)
    before_bytes = out.read_bytes()
    before = pd.read_parquet(out)

    monkeypatch.setattr(backtest_data, "cash_proxy_codes", lambda *a, **k: ["E05"])
    monkeypatch.setattr(backtest_data, "cash_proxy_returns", lambda *a, **k: pd.Series(0.001, index=nav.index))
    monkeypatch.setattr(backtest_data, "equity_index", lambda *a, **k: pd.Series(1.0, index=nav.index))
    monkeypatch.setattr(macro_module, "macro_features", lambda _st, cal: pd.DataFrame(index=cal))
    monkeypatch.setattr(market_features, "market_features", lambda *a, **k: pd.DataFrame(index=nav.index))

    def build(panel, master, *args, **kwargs):
        n = len(master)
        return pd.DataFrame(
            {
                "feature_asof": [nav.index[-1]] * n,
                "fund_code": list(master["fund_code"]),
                "label_ready": [False] * n,
                "eligible_at_decision": [True] * n,
            }
        )

    monkeypatch.setattr(fund_features, "build_features", build)

    with pytest.raises(RuntimeError, match="fail-closed"):
        _features_build_impl(nav, fm, cfg, Store(tmp_path / "legacy.duckdb"), out, incremental=True, leg="bes")
    # Kapsam dışı E00 guard öncesi budanmadı: kanonik byte/değer+dtype sabit.
    assert out.read_bytes() == before_bytes
    pd.testing.assert_frame_equal(pd.read_parquet(out), before, check_exact=True, check_dtype=True)
    assert "E00" in set(pd.read_parquet(out)["fund_code"])


def test_bes_build_no_prune_before_impl_calls(tmp_path, monkeypatch):
    """B1/ADR-0029/4: bes_build predictions/calibration'ı impl çağrılmadan önce mutate etmez."""
    import janus.cli as cli_module
    from janus.bes.build import bes_build

    cfg = _bes_cfg()
    st = Store(tmp_path / "legacy_outputs.duckdb")
    st.nav_wide = lambda: pd.DataFrame({"E05": [10.0]}, index=pd.to_datetime(["2026-09-23"]))
    st.latest_fund_master = lambda: pd.DataFrame({"fund_code": ["E05"], "fund_class": ["EMK"]})
    feat_path = tmp_path / "data" / "features" / "bes_features.parquet"
    pred_path = tmp_path / "data" / "predictions" / "bes_predictions.parquet"
    cal_path = tmp_path / "data" / "predictions" / "bes_calibrated_target_020.parquet"
    pred_path.parent.mkdir(parents=True)
    legacy = pd.DataFrame({"fund_code": ["E00", "E05"], "decision_at": pd.to_datetime(["2026-09-23"] * 2)})
    legacy.to_parquet(pred_path, index=False)
    legacy.to_parquet(cal_path, index=False)
    pred_bytes = pred_path.read_bytes()
    cal_bytes = cal_path.read_bytes()
    seen: dict[str, set[str]] = {}

    def build_features(*args, **kwargs):
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"fund_code": ["E05"], "feature_asof": pd.to_datetime(["2026-09-23"])}).to_parquet(
            feat_path, index=False
        )
        return {"n_rows": 1}

    def build_predictions(features, out, model_kwargs=None, incremental=False, cfg=None, **kwargs):
        # impl'e girmeden önce legacy dosya budanmamış olmalı (kapsam dışı E00 hâlâ orada).
        assert pred_path.read_bytes() == pred_bytes
        seen["pred_codes"] = set(pd.read_parquet(pred_path)["fund_code"])
        features.to_parquet(out, index=False)
        return {"n_pred_rows": len(features)}

    def calibrate(predictions, features, target, out, conf_cfg, incremental=False, **kwargs):
        assert cal_path.read_bytes() == cal_bytes
        seen["cal_codes"] = set(pd.read_parquet(cal_path)["fund_code"])
        predictions.to_parquet(out, index=False)
        return {"n_rows": len(predictions)}

    monkeypatch.setattr(cli_module, "_features_build_impl", build_features)
    monkeypatch.setattr(cli_module, "_predictions_build_impl", build_predictions)
    monkeypatch.setattr(cli_module, "_predictions_calibrate_impl", calibrate)
    monkeypatch.setattr(cli_module, "_model_kwargs_from_cpcv", lambda: {})
    result = bes_build(st, cfg, tmp_path, incremental=True)
    assert result["status"] == "ok"
    assert seen["pred_codes"] == {"E00", "E05"}
    assert seen["cal_codes"] == {"E00", "E05"}


def test_features_leg_class_and_file_separation(tmp_path, monkeypatch):
    """TEFAS ve BES yolları ayrı sınıf/dosya: `leg` yalnız kendi evrenini özellik üretir."""
    import janus.backtest.data as backtest_data
    import janus.features.fund_features as fund_features
    import janus.features.market as market_features
    from janus.cli import _features_build_impl
    from janus.features import macro as macro_module

    cfg = _bes_cfg()
    funds = ["T1", "E1"]
    fm = pd.DataFrame(
        {
            "fund_code": funds,
            "fund_class": ["YAT", "EMK"],
            "umbrella_type": ["Hisse", "Değişken"],
            "category": ["Hisse", "Değişken"],
            "withholding_rate": [0.175, 0.0],
        }
    )
    nav = pd.DataFrame({c: [10.0, 10.1] for c in funds}, index=pd.bdate_range("2026-09-23", periods=2))
    seen: dict[str, set[str]] = {}
    monkeypatch.setattr(backtest_data, "cash_proxy_codes", lambda *args, **kwargs: ["E1"])
    monkeypatch.setattr(
        backtest_data, "cash_proxy_returns", lambda *args, **kwargs: pd.Series([0.001, 0.001], index=nav.index)
    )
    monkeypatch.setattr(backtest_data, "equity_index", lambda *args, **kwargs: pd.Series([1.0, 1.0], index=nav.index))
    monkeypatch.setattr(macro_module, "macro_features", lambda _store, cal: pd.DataFrame(index=cal))
    monkeypatch.setattr(market_features, "market_features", lambda *args: pd.DataFrame(index=nav.index))

    def build(panel, master, *args, **kwargs):
        seen[master["fund_class"].iloc[0]] = set(master["fund_code"])
        n = len(master)
        return pd.DataFrame(
            {
                "feature_asof": [nav.index[-1]] * n,
                "fund_code": list(master["fund_code"]),
                "label_ready": [False] * n,
                "eligible_at_decision": [True] * n,
            }
        )

    monkeypatch.setattr(fund_features, "build_features", build)
    _features_build_impl(
        nav, fm, cfg, Store(tmp_path / "tefas.duckdb"), tmp_path / "fund_features.parquet", leg="tefas"
    )
    _features_build_impl(nav, fm, cfg, Store(tmp_path / "bes.duckdb"), tmp_path / "bes_features.parquet", leg="bes")
    assert seen["YAT"] == {"T1"}
    assert seen["EMK"] == {"E1"}


def test_bes_cash_proxy_leg_separation():
    """F13/F16: BES PPF sepeti yalnız EMK'dan; YAT PPF karışmaz, yoksa fail-closed."""
    from janus.backtest.data import cash_proxy_codes

    cfg = _bes_cfg()
    fm = pd.DataFrame(
        {
            "fund_code": ["Y1", "E1"],
            "fund_class": ["YAT", "EMK"],
            "umbrella_type": ["Para Piyasası", "Para Piyasası"],
            "category": ["Para Piyasası", "Para Piyasası"],
            "founder": ["Kurucu", "Kurucu"],
            "founder_code": ["F1", "F2"],
            "n_nav": [100, 100],
        }
    )
    nav = pd.DataFrame({"Y1": [10.0, 10.1], "E1": [10.0, 10.1]}, index=pd.bdate_range("2026-09-23", periods=2))
    assert cash_proxy_codes(nav, fm, cfg, leg="bes") == ["E1"]
    # Yalnız YAT para piyasası varsa BES nakit vekili yok (fallback yok).
    assert cash_proxy_codes(nav, fm[fm["fund_class"] == "YAT"], cfg, leg="bes") == []


def test_bes_plan_missing_founder_no_right(tmp_path, monkeypatch):
    """F14/F15: kurucusu bilinmeyen seçim tahsis edilmez; ilk planda hak harcanmaz."""
    from janus.bes.build import bes_build
    from janus.bes.plan import bes_plan

    cfg = _bes_cfg()
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=400), cfg, mode="initial", snapshot_date="2026-09-22")
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    bes_build(st, cfg, tmp_path)
    preds = pd.read_parquet(tmp_path / "data" / "predictions" / "bes_predictions.parquet")
    d = _last_month_dates(preds, k=1)[0]
    _force_selection(monkeypatch, lambda i, codes: [codes[1]])  # riskli fon (PPF değil)
    original = st.latest_fund_master

    def fm_missing():
        frame = original().copy()
        frame.loc[frame["fund_code"] == "E01", "founder_code"] = None
        frame.loc[frame["fund_code"] == "E01", "founder"] = ""
        return frame

    monkeypatch.setattr(st, "latest_fund_master", fm_missing)
    res = bes_plan(st, cfg, d.strftime("%Y-%m-%d"), root=tmp_path)
    assert res["status"] == "no_change"
    assert st.con.execute("SELECT COUNT(*) FROM bes_plan").fetchone()[0] == 0


def test_bes_plan_cli_forced_change_end_to_end(tmp_path, monkeypatch):
    """CLI yolu: gerçek değişimde bir change_id; aynı gün tekrar koşu append-only ve idempotent."""
    cfg = _bes_cfg()
    st = Store(tmp_path / "t.duckdb")
    ingest_tefas(st, FakeTefasClient(n=12, days=400), cfg, mode="initial", snapshot_date="2026-09-22")
    monkeypatch.setattr(cli, "_store", lambda dry_run=False, read_only=False: st)
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(feat_macro, "macro_features", lambda _st, cal: _make_macro(cal))
    runner = CliRunner()
    assert runner.invoke(cli.app, ["bes", "build"]).exit_code == 0
    preds = pd.read_parquet(tmp_path / "data" / "predictions" / "bes_predictions.parquet")
    d = str(pd.to_datetime(preds["decision_at"]).max().date())
    _force_selection(monkeypatch, lambda i, codes: [codes[1]])
    r1 = runner.invoke(cli.app, ["bes", "plan", "--date", d])
    assert r1.exit_code == 0, r1.output
    assert st.con.execute("SELECT COUNT(DISTINCT change_id) FROM bes_plan").fetchone()[0] == 1
    n_rows = st.con.execute("SELECT COUNT(*) FROM bes_plan").fetchone()[0]
    r2 = runner.invoke(cli.app, ["bes", "plan", "--date", d])
    assert r2.exit_code == 0, r2.output
    assert "değişiklik yok" in r2.output
    assert st.con.execute("SELECT COUNT(DISTINCT change_id) FROM bes_plan").fetchone()[0] == 1
    assert st.con.execute("SELECT COUNT(*) FROM bes_plan").fetchone()[0] == n_rows
