"""janus CLI: doctor | notify-test | ingest | quality | universe | nightly | morning."""

from __future__ import annotations

import importlib
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import typer
from loguru import logger

from janus.bes.build import bes_build
from janus.bes.plan import bes_plan as bes_plan_impl
from janus.config import ROOT, get_settings, load_config, store_path

app = typer.Typer(help="JANUS — kişisel EOD portföy karar sistemi", no_args_is_help=True)

CORE = ["pandas", "numpy", "duckdb", "pyarrow", "yaml", "pydantic_settings", "requests", "loguru"]
ML = ["lightgbm", "sklearn", "scipy", "statsmodels", "mlflow", "borsapy", "pandera"]
OPTIONAL = ["torch", "gymnasium", "catboost", "hmmlearn", "jumpmodels", "mapie", "yfinance", "alpaca"]


def _check(mod: str) -> str:
    try:
        importlib.import_module(mod)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"eksik ({type(e).__name__})"


def _parquet_root(cfg: dict, dry_run: bool = False) -> Path:
    """Curated Parquet root; dry_run ayrı `<ad>_dryrun` root kullanır."""
    rel = (cfg.get("store") or {}).get("parquet_dir", "data/curated")
    p = Path(rel)
    if not p.is_absolute():
        p = ROOT / rel
    return p.with_name(p.name + "_dryrun") if dry_run else p


def _parquet_dir(cfg: dict, dry_run: bool = False) -> Path:
    """Resolve the immutable Parquet generation selected by its atomic current pointer."""
    root = _parquet_root(cfg, dry_run)
    pointer = root.with_name(f".{root.name}.current")
    if not pointer.is_file():
        return root
    relative = Path(pointer.read_text(encoding="utf-8").strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Parquet generation pointer geçersiz")
    return root.parent / relative


def _export_snapshot(st, cfg: dict, dry_run: bool, status: str) -> None:
    """Başarılı ingest/nightly sonrası değişmez parquet anlık görüntüsü (S3b-0b); dry_run ayrı depoya."""
    if status == "failed":
        return
    try:
        st.export_parquet_atomic(_parquet_root(cfg, dry_run))
    except Exception as e:  # noqa: BLE001 — export düşerse koşu sonucu bozulmasın
        logger.warning("parquet export atlandı: {} ({})", type(e).__name__, e)


@app.command("export-snapshot")
def export_snapshot_cmd() -> None:
    """DuckDB'den dört Parquet görünümünü yenile; başarısız/kısmi export'ta nonzero dön."""
    from janus.data.store import parquet_snapshot_complete  # noqa: PLC0415

    cfg = load_config()
    st = _store()
    try:
        out = _parquet_root(cfg)
        paths = st.export_parquet_atomic(out)
        expected = {"fund_master", "fund_nav", "macro", "runs"}
        active_generation = paths[0].parent if paths else Path()
        if {path.stem for path in paths} != expected or not parquet_snapshot_complete(active_generation):
            raise RuntimeError("dört tablo Parquet snapshot'ı tamamlanmadı")
        typer.echo(f"SNAPSHOT EXPORT: {len(paths)}/4 tablo güncellendi")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"SNAPSHOT EXPORT FAILED: {type(exc).__name__}: {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        st.close()


def _store(dry_run: bool = False, read_only: bool = False):
    """dry_run=True → sahte veriler ayrı dosyaya (data/janus_dryrun.duckdb); gerçek depo kirlenmez.

    read_only=True → cfg parquet dizininde tam anlık görüntü varsa Store.from_parquet (S3b-0b);
    yoksa/yarımsetse uyarı + mevcut depo DuckDB salt-okunur açılır.
    """
    from janus.data.store import Store, parquet_snapshot_complete  # noqa: PLC0415

    cfg = load_config()
    p = store_path(cfg)
    if dry_run:
        return Store(p.with_name(p.stem + "_dryrun" + p.suffix), read_only=read_only)
    if read_only and parquet_snapshot_complete(_parquet_dir(cfg)):
        return Store.from_parquet(_parquet_dir(cfg))
    if read_only:
        logger.warning("parquet anlık görüntüsü yok/yarım → DuckDB salt-okunur ({})", _parquet_dir(cfg))
    return Store(p, read_only=read_only)


def _client(dry_run: bool):
    from janus.data.tefas_client import BorsapyClient, FakeTefasClient  # noqa: PLC0415

    if dry_run:
        return FakeTefasClient()
    icfg = load_config().get("ingest", {})
    return BorsapyClient(sleep=icfg.get("sleep_seconds", 0.25), retries=icfg.get("retries", 3))


@app.command()
def doctor() -> None:
    """Ortam, paket, MPS, sır ve dizin kontrolü. Sır değerleri asla yazdırılmaz."""
    ok = True
    typer.echo(f"Python {sys.version.split()[0]} on {platform.machine()} ({platform.system()})")
    if sys.version_info < (3, 12):
        typer.echo("  ! Python >= 3.12 gerekli")
        ok = False
    for group, mods in (("core", CORE), ("ml", ML), ("optional", OPTIONAL)):
        for m in mods:
            status = _check(m)
            typer.echo(f"  [{group}] {m:18s} {status}")
            if group == "core" and status != "ok":
                ok = False
    try:
        import torch  # noqa: PLC0415

        typer.echo(f"  torch {torch.__version__} | MPS available: {torch.backends.mps.is_available()}")
    except Exception:  # noqa: BLE001
        typer.echo("  torch yok (S4'e kadar opsiyonel)")
    env = ROOT / ".env"
    typer.echo(f"  .env: {'var' if env.exists() else 'YOK (.env.example kopyalayın)'}")
    for k, v in get_settings().secrets_present().items():
        typer.echo(f"  secret {k:8s}: {'tanımlı' if v else 'boş'}")
    for d in ("data", "reports", "mlruns", "sample_data"):
        (ROOT / d).mkdir(exist_ok=True)
        typer.echo(f"  dir {d:12s}: ok")
    try:
        cfg = load_config()
        typer.echo(
            f"  config: ok (sprint {cfg['project']['sprint']}, store {cfg.get('store', {}).get('path', 'data/janus.duckdb')})"
        )
    except Exception as e:  # noqa: BLE001
        typer.echo(f"  ! config okunamadı: {type(e).__name__}")
        ok = False
    typer.echo("DOCTOR: " + ("YEŞİL" if ok else "KIRMIZI"))
    raise typer.Exit(code=0 if ok else 1)


@app.command("notify-test")
def notify_test() -> None:
    """Telegram'a test mesajı gönderir."""
    from janus.report.telegram import send_message  # noqa: PLC0415

    sent = send_message(f"JANUS merhaba — {datetime.now():%Y-%m-%d %H:%M} — S1 kurulumu.")
    typer.echo("Telegram: " + ("gönderildi" if sent else "GÖNDERİLEMEDİ (.env ve ağ kontrol)"))
    raise typer.Exit(code=0 if sent else 1)


def _macro_fetcher(dry_run: bool):
    """Birincil borsapy, yedek EVDS3 REST (aynı anahtar, header'da)."""
    from janus.data.ingest_evds import BorsapyMacro, EvdsRestMacro, FakeMacro  # noqa: PLC0415

    if dry_run:
        return FakeMacro()
    key = get_settings().evds_api_key
    policy_code = load_config().get("macro", {}).get("policy_rate_evds_code", "TP.APIFON4")
    fetchers = [BorsapyMacro(key)]
    if key:
        fetchers.append(EvdsRestMacro(key, policy_code=policy_code))
    return fetchers


@app.command()
def ingest(
    source: str = typer.Argument("tefas", help="tefas | evds"),
    mode: str = typer.Option("incremental", help="tefas: initial (5y) | incremental (1mo)"),
    limit: int | None = typer.Option(None, help="İlk N fon (duman testi)"),
    codes: str | None = typer.Option(None, help="Virgülle ayrılmış fon kodları (yalnızca bunlar)"),
    only_failed: bool = typer.Option(False, "--only-failed", help="Son koşuda hata veren fonları yeniden dene"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Sahte istemci; ağ yok"),
) -> None:
    """Veri ingest → DuckDB. tefas: fund_master snapshot + fund_nav; evds: macro tablosu."""
    st = _store(dry_run)
    cfg = load_config()
    if source == "evds":
        from janus.data.ingest_evds import ingest_macro  # noqa: PLC0415

        res = ingest_macro(st, _macro_fetcher(dry_run), cfg)
        typer.echo(
            f"MACRO {res['status']}: {res['n_rows']} satır {res['counts']} hatalar={res['failures']} ({res['duration_s']}s)"
        )
        _export_snapshot(st, cfg, dry_run, res["status"])
        raise typer.Exit(code=0 if res["status"] != "failed" else 1)
    if source != "tefas":
        raise typer.BadParameter("kaynak: tefas | evds")
    from janus.data.ingest_tefas import ingest_tefas  # noqa: PLC0415

    code_list = [c.strip() for c in codes.split(",")] if codes else None
    res = ingest_tefas(st, _client(dry_run), cfg, mode=mode, limit=limit, codes=code_list, only_failed=only_failed)
    typer.echo(
        f"INGEST {res['status']}: {res['n_processed']} fon, info_ok={res['n_info_ok']}, hist_ok={res['n_hist_ok']}, "
        f"nav_rows={res['n_nav_rows']}, failures={res['n_failures']} ({res['duration_s']}s)"
    )
    _export_snapshot(st, cfg, dry_run, res["status"])
    raise typer.Exit(code=0 if res["status"] != "failed" else 1)


@app.command("backfill-profile-success")
def backfill_profile_success_cmd(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Geriye uyumlu idempotent onarım: başarılı eski profillerin eksik last_success_at alanı."""
    st = _store()
    try:
        count = st.backfill_profile_success_at(dry_run=dry_run)
        action = "aday satır" if dry_run else "satır güncellendi"
        typer.echo(
            f"PROFILE SUCCESS BACKFILL: {count} {action}; kaynak=proxy_ingest_time; vekil zaman, PIT kanıtı değil."
        )
    finally:
        st.close()


@app.command("evds-search")
def evds_search(term: str) -> None:
    """EVDS seri kodu arama (kod doğrulamak için), örn. janus evds-search "politika faizi"."""
    import borsapy as bp  # noqa: PLC0415

    bp.set_evds_key(get_settings().evds_api_key)
    df = bp.evds_search(term)
    typer.echo(df.head(25).to_string() if hasattr(df, "to_string") else str(df))


@app.command()
def quality(send: bool = typer.Option(False, help="Özeti Telegram'a gönder")) -> None:
    """Veri kalitesi özeti (askı, tazelik, bozuk hücre) ve Faz-1 evren sayımı."""
    from janus.data.quality import flag_data_stale, quality_summary, trade_status_masks  # noqa: PLC0415
    from janus.portfolio.universe import phase1_mask  # noqa: PLC0415
    from janus.report.telegram import format_quality, send_message  # noqa: PLC0415

    cfg = load_config()
    st = _store(read_only=True)
    wide = st.nav_wide()
    fm = st.latest_fund_master()
    q = quality_summary(wide, fm, cfg, eligible=None)
    if not wide.empty and not fm.empty:
        u = cfg["legs"]["tefas"]["universe"]
        susp = flag_data_stale(wide.tail(30), u.get("max_stale_days", 2))
        can_buy, _ = trade_status_masks(fm)
        m = phase1_mask(fm, cfg, suspended=susp, can_buy=can_buy)
        q["n_in_universe"] = int(m.sum())
    for k, v in q.items():
        typer.echo(f"  {k}: {v}")
    if send:
        send_message(
            format_quality(q) + (f"\nFaz-1 evren: {q.get('n_in_universe', '?')} fon" if "n_in_universe" in q else "")
        )


@app.command()
def universe(asof: str | None = typer.Option(None)) -> None:
    """Faz-1 evrenindeki fon kodlarını listeler."""
    from janus.data.quality import flag_data_stale, trade_status_masks  # noqa: PLC0415
    from janus.portfolio.universe import phase1_mask  # noqa: PLC0415

    cfg = load_config()
    st = _store(read_only=True)
    fm = st.latest_fund_master()
    if fm.empty:
        typer.echo("fund_master boş — önce `janus ingest tefas --mode initial`")
        raise typer.Exit(code=1)
    wide = st.nav_wide()
    u = cfg["legs"]["tefas"]["universe"]
    susp = flag_data_stale(wide.tail(30), u.get("max_stale_days", 2)) if not wide.empty else None
    can_buy, _ = trade_status_masks(fm)
    m = phase1_mask(fm, cfg, asof=asof, suspended=susp, can_buy=can_buy)
    typer.echo(f"Faz-1 evren: {int(m.sum())} / {len(m)} fon")
    typer.echo(", ".join(m[m].index.tolist()))


@app.command()
def backtest(
    start: str | None = typer.Option(None, help="Başlangıç tarihi (YYYY-MM-DD); varsayılan tüm geçmiş"),
    end: str | None = typer.Option(None),
    top_n: int = typer.Option(10, help="B1/B3 için fon sayısı"),
    no_report: bool = typer.Option(False, "--no-report"),
) -> None:
    """Baseline zinciri (B0 nakit, B1 top-N momentum, B3 kural kapısı) — vergi/maliyet sonrası; rapor reports/."""
    from janus.backtest.runner import run_baselines  # noqa: PLC0415

    cfg = load_config()
    st = _store(read_only=True)
    rows, path = run_baselines(
        st, cfg, start=start, end=end, top_n=top_n, out_dir=None if no_report else ROOT / "reports"
    )
    for r in rows:
        typer.echo(
            f"{r['strategy']:<20s} CAGR %{r['cagr'] * 100:6.1f} | nakit %{r['cash_cagr'] * 100:6.1f} | Sharpe {r['sharpe']:5.2f} "
            f"| MDD %{r['mdd'] * 100:5.1f} | turnover/yıl {r['turnover_per_year']:4.2f} | vergi %{r['taxes_pct_of_final'] * 100:4.1f} | DD tetik {r['dd_triggers']}"
        )
    if path:
        typer.echo(f"rapor: {path}")


@app.command("backtest-suite")
def backtest_suite(
    top_n: int = typer.Option(10), quick: bool = typer.Option(False, "--quick", help="Küçük tarama/senaryo (hızlı)")
) -> None:
    """S2b paketi: zincir + yıllık + sürtünme + fon türleri + 12 başlangıç + stres + askı + tarama (PBO, deflated Sharpe) + MLflow."""
    from janus.backtest.runner import run_suite  # noqa: PLC0415

    md, path = run_suite(
        _store(read_only=True), load_config(), top_n=top_n, out_dir=ROOT / "reports", root=ROOT, quick=quick
    )
    typer.echo(md.split("\n## Yıllık")[0])
    typer.echo(f"rapor: {path}")


@app.command("gate-suite")
def gate_suite(top_n: int = typer.Option(10), quick: bool = typer.Option(False, "--quick")) -> None:
    """S3a: B2b üstünde kapı karşılaştırması (yok / R0 / R1 jump / R1 HMM / R2 makas), 12 başlangıç, B2b taraması."""
    from janus.backtest.gate_suite import run_gate_suite  # noqa: PLC0415

    md, path = run_gate_suite(_store(read_only=True), load_config(), top_n=top_n, out_dir=ROOT / "reports", quick=quick)
    typer.echo(md.split("\n## B2b")[0])
    typer.echo(f"rapor: {path}")


@app.command("coverage-diagnose")
def coverage_diagnose(
    select_end: str = typer.Option("2025-06-30", help="Seçim/dış test ayrımı (YYYY-MM-DD)"),
    top_n: int = typer.Option(10, help="Seçilen top-N seri boyutu"),
    seed: int = typer.Option(0, help="Rastgele-10 serisi tohumu (sabit)"),
) -> None:
    """S3b-5-1: kapsama açığı ayrıştırımı (ACI vs sabit-α; seçim vs aday havuzu) → reports/coverage_diag_<ts>.md."""
    import pandas as pd  # noqa: PLC0415

    from janus.models.conformal import calibrate_predictions  # noqa: PLC0415
    from janus.models.coverage_diag import build_report, coverage_diag_tables  # noqa: PLC0415

    pred_path = ROOT / "data" / "predictions" / "predictions.parquet"
    cal_path = ROOT / "data" / "predictions" / "calibrated_target_020.parquet"
    feat_path = _features_parquet()
    for p in (pred_path, cal_path):
        if not p.exists():
            typer.echo(f"{p.name} yok — önce `janus predictions build` / `calibrate`")
            raise typer.Exit(code=1)
    preds = pd.read_parquet(pred_path)
    feats = pd.read_parquet(feat_path)
    cal = pd.read_parquet(cal_path)
    cfg = load_config().get("conformal", {})
    fixed = calibrate_predictions(
        preds,
        feats,
        miscoverage_target=0.20,  # calibrated_target_020 hedefi; α sabit = hedef (ACI kapalı)
        gamma=0.0,  # ACI kapalı: α sabit = hedef
        alpha_min=float(cfg.get("alpha_min", 0.02)),
        alpha_max=float(cfg.get("alpha_max", 0.5)),
        n_min=int(cfg.get("n_min", 200)),
        calib_window=int(cfg.get("calib_window", 126)),
    )
    sel_end = pd.Timestamp(select_end)
    t_aci, s_aci = coverage_diag_tables(cal, sel_end, top_n=top_n, seed=seed)
    t_fix, s_fix = coverage_diag_tables(fixed, sel_end, top_n=top_n, seed=seed)
    ts = datetime.now()
    exp_id = f"coverage_diag_{ts:%Y%m%d_%H%M}"
    asof = pd.Timestamp(cal_path.stat().st_mtime, unit="s", tz="Europe/Istanbul")
    md = build_report(
        t_aci, s_aci, t_fix, s_fix, exp_id, f"veri as-of: {asof:%Y-%m-%d %H:%M} (calibrated parquet mtime)", sel_end
    )
    out = ROOT / "reports" / f"{exp_id}.md"
    out.write_text(md, encoding="utf-8")
    typer.echo(f"COVERAGE-DIAG: {exp_id} → {out}")


@app.command("select-suite")
def select_suite(
    top_n: int = typer.Option(10),
    select_end: str = typer.Option("2025-06-30", help="Seçim dönemi sonu (YYYY-MM-DD); dış test bundan panel sonuna"),
    quick: bool = typer.Option(False, "--quick", help="12 yerine 4 başlangıç"),
) -> None:
    """S3b-4: B0 sepet · B2b · B2c-q · B2c-m · B2c-m+ceza; kapı yok (ADR-18). Uzun koşu — kullanıcı çalıştırır."""
    from janus.backtest.select_suite import run_select_suite_from_store  # noqa: PLC0415

    md, path = run_select_suite_from_store(
        _store(read_only=True), load_config(), top_n=top_n, select_end=select_end, out_dir=ROOT / "reports", quick=quick
    )
    typer.echo(md.split("\n## Kullanılabilirlik")[0])
    typer.echo(f"rapor: {path}")


features_app = typer.Typer(help="Fon-gün özellik deposu (S3b-1)")
app.add_typer(features_app, name="features")
predictions_app = typer.Typer(help="Walk-forward OOS tahminler (S3b-2)")
app.add_typer(predictions_app, name="predictions")


def _features_parquet() -> Path:
    p = ROOT / "data" / "features" / "fund_features.parquet"
    if not p.exists():
        typer.echo("fund_features.parquet yok — önce `janus features build`")
        raise typer.Exit(code=1)
    return p


def _overlap_start(cal_index: pd.DatetimeIndex, anchor: pd.Timestamp, n_bdays: int = 45) -> pd.Timestamp:
    """Artımlı birleştirme için çakışma penceresinin başlangıcı (anchor'dan n_bdays geri)."""
    cal_index = pd.DatetimeIndex(cal_index).sort_values()
    if anchor in cal_index:
        pos = int(cal_index.get_loc(anchor))
        return cal_index[max(0, pos - n_bdays)]
    return anchor - pd.Timedelta(days=90)


def _asof_summary(asof):
    """asof'u summary JSON'ı için normalize et (str/date/Timestamp/None)."""
    if asof is None:
        return None
    if isinstance(asof, str):
        return asof
    if hasattr(asof, "date"):
        return str(asof.date())
    return str(asof)


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame, keys: list[str]) -> bool:
    """İki parquet çerçevesini anahtar bazında bit düzeyinde karşılaştır (S5-6e/F17).

    Kolon kümesi/tip/satır kümesi farkı → False. Amaç: erken çıkışın sessizce
    içerik değişikliğini atlamasını engellemek.
    """
    if a is None or b is None:
        return a is b
    if set(a.columns) != set(b.columns):
        return False
    cols = list(a.columns)
    aa = a[cols].sort_values(keys, kind="stable").reset_index(drop=True)
    bb = b[cols].sort_values(keys, kind="stable").reset_index(drop=True)
    if len(aa) != len(bb):
        return False
    try:
        pd.testing.assert_frame_equal(aa, bb, check_exact=True, check_dtype=True)
    except AssertionError:
        return False
    return True


def _y_prefix_ok(prefix_existing: pd.DataFrame, prefix_recomputed: pd.DataFrame, keys: list[str]) -> bool:
    """Historical `y` öneki için tek yönlü etiket olgunlaşması istisnası (ADR-0029).

    İzinli tek durum: existing `y` NaN ve recomputed `y` dolu (yeni bilgi). Aksi hâlde
    (dolu→farklı, dolu→NaN, dtype farkı) korunum ihlali → False.
    """
    e = prefix_existing.set_index(keys)["y"].sort_index()
    r = prefix_recomputed.set_index(keys)["y"].sort_index()
    if not e.index.equals(r.index) or e.dtype != r.dtype:
        return False
    e_na = e.isna().to_numpy()
    r_na = r.isna().to_numpy()
    matured = e_na & ~r_na
    underived = ~matured
    if not underived.any():
        return True
    ev = e.to_numpy()[underived]
    rv = r.to_numpy()[underived]
    return bool(np.array_equal(ev, rv, equal_nan=True))


def _time_prefix(frame: pd.DataFrame, time_col: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """`time_col < cutoff` satırları (prefix korunumu/revizyon karşılaştırması için)."""
    return frame.loc[pd.to_datetime(frame[time_col]) < cutoff]


def _assert_panel_business_days(dates, cfg: dict | None, context: str) -> None:
    """Panel tarihlerini configured iş günü eksenine karşı fail-closed doğrula (ADR-0029/5).

    Normalize edilmiş tekil tarihler `is_business_day` ile denetlenir: hafta sonu + mevcut
    `config/janus.yaml:calendar.holidays`. `cfg` yoksa/`calendar` yoksa yalnız hafta sonu
    kontrol edilir (mevcut `is_business_day` davranışı). İş günü olmayan tarih varsa hiçbir
    yazım yapılmadan sayı bildirilerek `RuntimeError` yükseltilir. Eksik gün yönü gevşetilmez;
    terminal karar günü istisnası `next_business_day(..., cfg)` çıktısının iş günü olmasıyla korunur.
    """
    from janus.data.quality import is_business_day  # noqa: PLC0415

    idx = pd.to_datetime(pd.Series(dates)).dropna().dt.normalize().unique()
    if len(idx) == 0:
        return
    bad = sorted(pd.Timestamp(d) for d in idx if not is_business_day(d, cfg))
    if bad:
        sample = ", ".join(str(d.date()) for d in bad[:5])
        raise RuntimeError(
            f"{context}: panel iş günü olmayan tarih içeriyor ({len(bad)} adet; örn. {sample}); fail-closed (ADR-0029/5)"
        )


def _train_max_t_ok(preds: pd.DataFrame, features: pd.DataFrame) -> tuple[bool, int]:
    """TEMPORAL §9.3: her prediction için `train_max_t <= decision_at − 23` işlem günü.

    İşlem günü takvimi: `feature_asof` ∪ `decision_at`. Takvimde bulunmayan tarih veya
    yetersiz geçmiş `ok=False` sayılır (sessiz geçiştirme yok).
    """
    if preds is None or len(preds) == 0:
        return True, 0
    fa = pd.to_datetime(features["feature_asof"]).dropna().to_numpy()
    da = pd.to_datetime(preds["decision_at"]).dropna().to_numpy()
    if len(fa) == 0 or len(da) == 0:
        return False, int(len(preds))
    cal = pd.DatetimeIndex(np.unique(np.concatenate([fa, da]))).sort_values()
    pos = pd.Series(np.arange(len(cal)), index=cal)
    pos_t = pd.to_datetime(preds["train_max_t"]).map(pos)
    pos_d = pd.to_datetime(preds["decision_at"]).map(pos)
    bad = pos_t.isna() | pos_d.isna() | (pos_t > pos_d - 23)
    n_bad = int(bad.sum())
    return n_bad == 0, n_bad


def _features_build_impl(
    nav: pd.DataFrame,
    fm: pd.DataFrame,
    cfg: dict,
    st,
    out: Path,
    asof: str | pd.Timestamp | None = None,
    incremental: bool = False,
    leg: str = "tefas",
) -> dict:
    """Fon-gün özellik deposu; incremental=True ise mevcut parquet'ten devam eder.

    `leg`: "tefas" | "bes" — BES'te EMK evreni, stopaj 0, PPF sepeti (fallback yok).
    """

    from janus.backtest.data import cash_proxy_codes, cash_proxy_returns, equity_index  # noqa: PLC0415
    from janus.features.fund_features import build_features  # noqa: PLC0415
    from janus.features.macro import macro_features as macro_pit  # noqa: PLC0415
    from janus.features.market import market_features  # noqa: PLC0415

    t0 = time.perf_counter()
    if "fund_class" not in fm.columns:
        raise ValueError("fund_master.fund_class zorunlu; TEFAS/BES evreni ayırt edilemiyor")
    wanted_class = "EMK" if leg == "bes" else "YAT"
    fm = fm[fm["fund_class"].fillna("").astype(str).str.upper().eq(wanted_class)].copy()
    if leg == "bes":
        # F13: tek EMK kapsam politikası (devlet katkı + kapsam desenleri + TEFAS kara listesi).
        from janus.portfolio.universe import bes_scope_mask  # noqa: PLC0415

        scope = bes_scope_mask(fm, cfg)
        fm = fm.loc[fm["fund_code"].astype(str).map(scope).fillna(False).to_numpy()].copy()
    nav = nav.loc[:, nav.columns.intersection(fm["fund_code"].astype(str))]
    if fm.empty or nav.empty:
        raise ValueError(f"{wanted_class} özellikleri için fon/NAV yok")
    # ADR-0029/5: panel takvimi fail-closed; existing okunmadan/guard ve yazımdan ÖNCE (her iki leg).
    _assert_panel_business_days(nav.index, cfg, f"{wanted_class} özellik paneli")
    existing = None
    existing_max = None
    if incremental and out.exists():
        existing = pd.read_parquet(out)
        if existing.empty:
            existing = None
        else:
            existing_max = pd.to_datetime(existing["feature_asof"]).max()
    nav_max = pd.to_datetime(nav.index).max().normalize()
    if incremental and existing is not None:
        if nav_max < existing_max:
            # Girdi, maddeleşmiş dosyanın gerisinde: yeni sürüm doğrulanamaz → hattı durdur (stale).
            elapsed_seconds = time.perf_counter() - t0
            return {
                "status": "stale",
                "n_rows": int(len(existing)),
                "n_funds": int(existing["fund_code"].nunique()),
                "feature_asof_min": str(existing["feature_asof"].min().date()),
                "feature_asof_max": str(existing["feature_asof"].max().date()),
                "asof": _asof_summary(asof),
                "snapshot_asof": pd.Timestamp(st.snapshot_asof).isoformat()
                if getattr(st, "snapshot_asof", None)
                else None,
                "label_ready_ratio": float(existing["label_ready"].mean()),
                "eligible_at_decision_ratio": float(existing["eligible_at_decision"].mean()),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "incremental": True,
                "new_dates": 0,
                "revision_detected": False,
                "input_regressed": True,
            }
        asof = nav_max
    cash_codes = cash_proxy_codes(nav, fm, cfg, leg=leg)
    if leg == "bes" and not cash_codes:
        raise RuntimeError("BES nakit vekili yok: EMK para piyasası fonu sepeti bulunamadı")
    cash_returns = cash_proxy_returns(nav, cash_codes)
    mf = macro_pit(st, pd.DatetimeIndex(nav.index))
    fm_build = fm
    if leg == "bes":
        # BES'te stopaj yok (S5-5)
        fm_build = fm.copy()
        fm_build["withholding_rate"] = 0.0
    eq_pattern = "Hisse"
    eq_cols = fm_build[fm_build["umbrella_type"].fillna("").str.contains(eq_pattern, case=False)]["fund_code"].tolist()
    mkt = market_features(nav, equity_index(nav, fm_build, leg=leg), eq_cols)
    df = build_features(nav, fm_build, cash_returns, mf, mkt, cfg=cfg, asof=asof, leg=leg)
    new_dates = int(len(df))
    revision_detected = False
    if incremental and existing is not None:
        keys = ["feature_asof", "fund_code"]
        overlap = _overlap_start(nav.index, existing_max, n_bdays=45)
        # Prefix (overlap öncesi) korunur; içeriği değişmişse "revizyon/yeni sürüm" bayrağı.
        revision_detected = not _frames_equal(
            _time_prefix(existing, "feature_asof", overlap),
            _time_prefix(df, "feature_asof", overlap),
            keys,
        )
        keep = _time_prefix(existing, "feature_asof", overlap)
        if revision_detected:
            # PO kararı (S5-6e, seçenek A): overlap öncesi tarihsel revizyon → fail-closed.
            raise RuntimeError(
                "özellik öneki revize (tarihsel NAV revizyonu veya yeni fon geçmişi): "
                "fail-closed; geçmiş PIT öneki yeniden yazılmaz"
            )
        df = pd.concat([keep, df[pd.to_datetime(df["feature_asof"]) >= overlap]], ignore_index=True)
        df = df.sort_values(keys, kind="stable").reset_index(drop=True)
        new_dates = int((pd.to_datetime(df["feature_asof"]) > existing_max).sum())
        if new_dates == 0 and _frames_equal(existing, df, keys):
            elapsed_seconds = time.perf_counter() - t0
            return {
                "status": "ok",
                "n_rows": int(len(df)),
                "n_funds": int(df["fund_code"].nunique()),
                "feature_asof_min": str(df["feature_asof"].min().date()),
                "feature_asof_max": str(df["feature_asof"].max().date()),
                "asof": _asof_summary(asof),
                "snapshot_asof": pd.Timestamp(st.snapshot_asof).isoformat()
                if getattr(st, "snapshot_asof", None)
                else None,
                "label_ready_ratio": float(df["label_ready"].mean()),
                "eligible_at_decision_ratio": float(df["eligible_at_decision"].mean()),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "incremental": True,
                "new_dates": 0,
                "revision_detected": False,
            }
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    elapsed_seconds = time.perf_counter() - t0
    return {
        "status": "ok",
        "n_rows": int(len(df)),
        "n_funds": int(df["fund_code"].nunique()),
        "feature_asof_min": str(df["feature_asof"].min().date()),
        "feature_asof_max": str(df["feature_asof"].max().date()),
        "asof": _asof_summary(asof),
        "snapshot_asof": pd.Timestamp(st.snapshot_asof).isoformat() if getattr(st, "snapshot_asof", None) else None,
        "label_ready_ratio": float(df["label_ready"].mean()),
        "eligible_at_decision_ratio": float(df["eligible_at_decision"].mean()),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "incremental": incremental,
        "new_dates": new_dates,
        "revision_detected": revision_detected,
    }


def _predictions_build_impl(
    features: pd.DataFrame,
    out: Path,
    model_kwargs: dict[str, Any] | None,
    incremental: bool = False,
    cfg: dict | None = None,
) -> dict:
    """Walk-forward OOS tahminler; incremental=True ise mevcut tahminlerden devam eder.

    `cfg` verilirse feature `feature_asof` ve prediction `decision_at` tarihleri aynı configured
    iş günü eksenine karşı fail-closed doğrulanır (ADR-0029/5); terminal `next_business_day`
    üretimi iş günü olduğundan geçerli kalır. `cfg=None` → takvim doğrulaması atlanır.
    """

    from janus.models.walkforward import run_walkforward  # noqa: PLC0415

    t0 = time.perf_counter()
    if cfg is not None:
        _assert_panel_business_days(features["feature_asof"], cfg, "tahmin feature_asof paneli")
    existing = None
    existing_max = None
    if incremental and out.exists():
        existing = pd.read_parquet(out)
        if existing.empty:
            existing = None
        else:
            existing_max = pd.to_datetime(existing["decision_at"]).max()
    preds = run_walkforward(features, model_kwargs=model_kwargs)
    if cfg is not None:
        _assert_panel_business_days(preds["decision_at"], cfg, "tahmin decision_at paneli")
    new_dates = int(len(preds))
    revision_detected = False
    if incremental and existing is not None:
        keys = ["decision_at", "fund_code"]
        cal_index = pd.DatetimeIndex(np.sort(pd.to_datetime(features["feature_asof"]).dropna().unique()))
        overlap = _overlap_start(cal_index, existing_max - pd.Timedelta(days=1), n_bdays=45)
        revision_detected = not _frames_equal(
            _time_prefix(existing, "decision_at", overlap),
            _time_prefix(preds, "decision_at", overlap),
            keys,
        )
        keep = _time_prefix(existing, "decision_at", overlap)
        if revision_detected:
            # PO kararı (S5-6e, seçenek A): overlap öncesi tarihsel revizyon → fail-closed.
            raise RuntimeError(
                "tahmin öneki revize (özellik/tarihsel NAV revizyonu veya yeni fon geçmişi): "
                "fail-closed; geçmiş PIT öneki yeniden yazılmaz"
            )
        preds = pd.concat([keep, preds[pd.to_datetime(preds["decision_at"]) >= overlap]], ignore_index=True)
        preds = preds.sort_values(keys, kind="stable").reset_index(drop=True)
        new_dates = int((pd.to_datetime(preds["decision_at"]) > existing_max).sum())
        if new_dates == 0 and _frames_equal(existing, preds, keys):
            ok, n_bad = _train_max_t_ok(existing, features)
            if not ok:
                raise RuntimeError(f"train_max_t D−23 sözleşme ihlali: {n_bad} satır")
            elapsed = time.perf_counter() - t0
            return {
                "status": "ok",
                "n_rows": int(len(existing)),
                "n_pred_rows": int(len(existing)),
                "n_models": int(existing["model_id"].nunique()) if len(existing) else 0,
                "decision_at_min": str(existing["decision_at"].min().date()) if len(existing) else None,
                "decision_at_max": str(existing["decision_at"].max().date()) if len(existing) else None,
                "train_max_t_ok": ok,
                "train_max_t_violations": n_bad,
                "elapsed_seconds": round(elapsed, 3),
                "incremental": True,
                "new_dates": 0,
                "revision_detected": False,
            }
    # ADR-0029/2: terminal D−23 doğrulaması canonical yazımdan ÖNCE; ihlalde dosya değişmez.
    ok, n_bad = _train_max_t_ok(preds, features)
    if not ok:
        raise RuntimeError(f"train_max_t D−23 sözleşme ihlali: {n_bad} satır")
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    elapsed = time.perf_counter() - t0
    return {
        "status": "ok",
        "n_rows": int(len(preds)),
        "n_pred_rows": int(len(preds)),
        "n_models": int(preds["model_id"].nunique()) if len(preds) else 0,
        "decision_at_min": str(preds["decision_at"].min().date()) if len(preds) else None,
        "decision_at_max": str(preds["decision_at"].max().date()) if len(preds) else None,
        "train_max_t_ok": ok,
        "train_max_t_violations": n_bad,
        "elapsed_seconds": round(elapsed, 3),
        "incremental": incremental,
        "new_dates": new_dates,
        "revision_detected": revision_detected,
    }


def _predictions_calibrate_impl(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    target: float,
    out: Path,
    cfg: dict,
    incremental: bool = False,
) -> dict:
    """CQR + ACI-tarzı günlük global α; incremental=True ise mevcut kalibrasyondan devam eder."""

    from janus.models.conformal import calibrate_predictions, coverage_report  # noqa: PLC0415

    t0 = time.perf_counter()
    existing = None
    existing_max = None
    if incremental and out.exists():
        existing = pd.read_parquet(out)
        if existing.empty:
            existing = None
        else:
            existing_max = pd.to_datetime(existing["decision_at"]).max()
    cal = calibrate_predictions(
        predictions,
        features,
        miscoverage_target=target,
        gamma=float(cfg.get("gamma", 0.05)),
        alpha_min=float(cfg.get("alpha_min", 0.02)),
        alpha_max=float(cfg.get("alpha_max", 0.5)),
        n_min=int(cfg.get("n_min", 200)),
        calib_window=int(cfg.get("calib_window", 126)),
    )
    new_dates = int(len(cal))
    revision_detected = False
    if incremental and existing is not None:
        keys = ["decision_at", "fund_code"]
        append_cols = ["alpha_D", "lower", "upper", "n_calib", "quality_flag"]
        strict_cols = ["q10", "q50", "q90", "model_id", "train_max_t"]
        canonical = [*keys, *strict_cols, *append_cols, "y"]
        in_prefix = pd.to_datetime(cal["decision_at"]) <= existing_max
        prefix_recomputed = cal.loc[in_prefix].copy()
        prefix_existing = existing.loc[pd.to_datetime(existing["decision_at"]) <= existing_max].copy()
        # ADR-0028/0029: historical canonical prefix immutable. Kolon kümesi, α-kaynaklı append-only
        # kolonlar ve q10/q50/q90/model_id/train_max_t değer+dtype birebir korunur; `y` yalnız tek
        # yönlü olgunlaşabilir. Herhangi bir ihlalde yazmadan fail-closed.
        if set(prefix_existing.columns) != set(prefix_recomputed.columns) or not set(canonical).issubset(
            prefix_existing.columns
        ):
            revision_detected = True
        elif not _frames_equal(prefix_existing[[*keys, *append_cols]], prefix_recomputed[[*keys, *append_cols]], keys):
            revision_detected = True
        elif not _frames_equal(prefix_existing[[*keys, *strict_cols]], prefix_recomputed[[*keys, *strict_cols]], keys):
            revision_detected = True
        else:
            revision_detected = not _y_prefix_ok(prefix_existing, prefix_recomputed, keys)
        if revision_detected:
            raise RuntimeError(
                "kalibrasyon canonical öneki revize: append-only/canonical ihlali; fail-closed; "
                "geçmiş PIT/alpha öneki yeniden yazılmaz (S5-R2/ADR-0029)"
            )
        # Olgunlaşan `y` gibi pasif kolonlar FULL ile tazelenir; α-kaynaklı kolonlar existing'den korunur.
        old = prefix_existing.set_index(keys)[append_cols]
        rec = prefix_recomputed.set_index(keys)
        present = rec.index.isin(old.index)
        for col in append_cols:
            rec.loc[present, col] = old.loc[rec.index[present], col].to_numpy()
        prefix_out = rec.reset_index()
        new_part = cal.loc[~in_prefix]
        cal = pd.concat([prefix_out, new_part], ignore_index=True)
        cal = cal.sort_values(keys, kind="stable").reset_index(drop=True)
        new_dates = int((pd.to_datetime(cal["decision_at"]) > existing_max).sum())
        if new_dates == 0 and _frames_equal(existing, cal, keys):
            cov = coverage_report(existing)
            elapsed = time.perf_counter() - t0
            return {
                "status": "ok",
                "n_rows": int(len(existing)),
                "miscoverage_target": target,
                "n_no_interval": int(existing["lower"].isna().sum()),
                "n_crossing": int(existing["quality_flag"].str.contains("crossing").sum()),
                "alpha_last": float(existing["alpha_D"].dropna().iloc[-1])
                if existing["alpha_D"].notna().any()
                else None,
                "coverage_last_month": float(cov["coverage"].iloc[-1]) if len(cov) else None,
                "elapsed_seconds": round(elapsed, 3),
                "incremental": True,
                "new_dates": 0,
                "revision_detected": False,
                "note": "ACI-tarzı günlük global α; özgün ACI garantisi iddia edilmez (TEMPORAL §6)",
            }
    out.parent.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(out, index=False)
    cov = coverage_report(cal)
    elapsed = time.perf_counter() - t0
    return {
        "status": "ok",
        "n_rows": int(len(cal)),
        "miscoverage_target": target,
        "n_no_interval": int(cal["lower"].isna().sum()),
        "n_crossing": int(cal["quality_flag"].str.contains("crossing").sum()),
        "alpha_last": float(cal["alpha_D"].dropna().iloc[-1]) if cal["alpha_D"].notna().any() else None,
        "coverage_last_month": float(cov["coverage"].iloc[-1]) if len(cov) else None,
        "elapsed_seconds": round(elapsed, 3),
        "incremental": incremental,
        "new_dates": new_dates,
        "revision_detected": revision_detected,
        "note": "ACI-tarzı günlük global α; özgün ACI garantisi iddia edilmez (TEMPORAL §6)",
    }


def _select_impl(
    date: str,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    calibrated: pd.DataFrame,
    out: Path,
    cfg: dict,
) -> dict:
    """B2c-fdr seçimi: BH q20 (kanonik) + q10/q30 gölge kolonları."""
    from janus.models.conformal_selection import bh_select, conformal_pvalues  # noqa: PLC0415

    t0 = time.perf_counter()
    d = pd.Timestamp(date).normalize()
    day_preds = predictions[pd.to_datetime(predictions["decision_at"]) == d]
    if day_preds.empty:
        raise ValueError(f"decision_at={date} için tahmin yok")
    day_cal = calibrated[pd.to_datetime(calibrated["decision_at"]) == d]
    pvals = conformal_pvalues(predictions, features, calib_window=int(cfg.get("calib_window", 126)))
    day_pv = pvals[pd.to_datetime(pvals["decision_at"]) == d].copy()
    day_pv = day_pv.merge(day_cal[["fund_code", "lower"]], on="fund_code", how="left")
    q_grid = cfg.get("fdr_q_grid", [0.10, 0.20, 0.30])
    for q in q_grid:
        selected = bh_select(day_pv.set_index("fund_code")["p_value"], q)
        day_pv[f"selected_q{int(round(q * 100)):02d}"] = day_pv["fund_code"].isin(selected)
    cols = ["decision_at", "fund_code", "p_value"]
    cols += [f"selected_q{int(round(q * 100)):02d}" for q in sorted(q_grid)]
    cols += ["lower", "q50"]
    day_pv = day_pv[cols]
    out.parent.mkdir(parents=True, exist_ok=True)
    day_pv.to_parquet(out, index=False)
    return {
        "status": "ok",
        "n_rows": int(len(day_pv)),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "date": date,
        "n_funds": int(len(day_pv)),
        "n_selected_q20": int(day_pv["selected_q20"].sum()) if "selected_q20" in day_pv.columns else 0,
        "q_grid": [float(q) for q in q_grid],
    }


@predictions_app.command("build")
def predictions_build(
    num_leaves: int = typer.Option(31),
    min_data_in_leaf: int = typer.Option(1000),
    num_boost_round: int = typer.Option(300),
    incremental: bool = typer.Option(False, "--incremental", help="Mevcut tahminlerden devam et"),
) -> None:
    """Aylık refit walk-forward OOS tahminler → data/predictions/predictions.parquet."""
    df = pd.read_parquet(_features_parquet())
    out = ROOT / "data" / "predictions" / "predictions.parquet"
    summary = _predictions_build_impl(
        df,
        out,
        model_kwargs={
            "num_leaves": num_leaves,
            "min_data_in_leaf": min_data_in_leaf,
            "num_boost_round": num_boost_round,
        },
        incremental=incremental,
        cfg=load_config(),
    )
    (out.parent / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    typer.echo(
        f"PREDICTIONS: {summary['n_pred_rows']} satır, {summary['n_models']} model, "
        f"incremental={summary['incremental']}, new_dates={summary['new_dates']} → {out}"
    )


@predictions_app.command("calibrate")
def predictions_calibrate(
    target: float = typer.Option(0.20, help="miscoverage_target: 0.20 seçim / 0.10 risk bandı"),
    diagnose: bool = typer.Option(False, "--diagnose", help="Salt-okunur agregat diagnostik; hiçbir dosya yazmaz"),
    incremental: bool = typer.Option(False, "--incremental", help="Mevcut kalibrasyondan devam et"),
) -> None:
    """CQR + ACI-tarzı günlük global α (S3b-3) → calibrated_target_<tag>.parquet (target'a göre ayrık)."""
    pred_path = ROOT / "data" / "predictions" / "predictions.parquet"
    if not pred_path.exists():
        typer.echo("predictions.parquet yok — önce `janus predictions build`")
        raise typer.Exit(code=1)
    cfg = load_config().get("conformal", {})
    preds = pd.read_parquet(pred_path)
    feats = pd.read_parquet(_features_parquet())
    if diagnose:
        from janus.models.conformal import calibration_diagnostics  # noqa: PLC0415

        diag = calibration_diagnostics(
            preds,
            feats,
            miscoverage_target=target,
            **{
                k: float(cfg[k]) if k in ("gamma", "alpha_min", "alpha_max") else int(cfg[k])
                for k in ("gamma", "alpha_min", "alpha_max", "n_min", "calib_window")
                if k in cfg
            },
        )
        for k, v in diag.items():
            typer.echo(f"  {k}: {v}")
        raise typer.Exit(code=0)
    tag = str(int(round(target * 100))).zfill(3)
    out = ROOT / "data" / "predictions" / f"calibrated_target_{tag}.parquet"
    summary = _predictions_calibrate_impl(preds, feats, target, out, cfg, incremental=incremental)
    (out.parent / f"calibration_summary_target_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    typer.echo(
        f"CALIBRATE: {summary['n_rows']} satır, alpha_last={summary['alpha_last']}, "
        f"incremental={summary['incremental']}, new_dates={summary['new_dates']} → {out}"
    )


@predictions_app.command("cpcv")
def predictions_cpcv(
    num_boost_round: int = typer.Option(300),
) -> None:
    """CPCV (6 blok / 2 test) hiperparametre seçimi → data/predictions/cpcv_summary.json (yalnız seçim; kanıt değil)."""
    import itertools  # noqa: PLC0415

    from janus.models.walkforward import cpcv_select  # noqa: PLC0415

    df = pd.read_parquet(_features_parquet())
    grid = [{"num_leaves": n, "min_data_in_leaf": m} for n, m in itertools.product((15, 31, 63), (200, 500, 1000))]
    res = cpcv_select(df, grid, model_kwargs={"num_boost_round": num_boost_round})
    best = res.sort_values("pinball_mean").head(1)
    out_dir = ROOT / "data" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_combos": int(len(res)),
        "best_params": {
            k: (int(best[k].iloc[0]) if k in ("num_leaves", "min_data_in_leaf") else None)
            for k in ("num_leaves", "min_data_in_leaf")
        }
        if len(best)
        else None,
        "best_pinball_mean": float(best["pinball_mean"].iloc[0]) if len(best) else None,
        "note": "yalnızca hiperparametre seçimi; strateji başarısı veya bağımsız OOS kanıtı değildir (VALIDATION §1)",
    }
    (out_dir / "cpcv_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    typer.echo(f"CPCV: {summary['n_combos']} kombinasyon; en iyi: {summary['best_params']}")


@features_app.command("build")
def features_build(
    asof: str | None = typer.Option(None, help="Yalnızca feature_asof <= bu tarih (YYYY-MM-DD)"),
    incremental: bool = typer.Option(False, "--incremental", help="Mevcut feature deposundan devam et"),
) -> None:
    """Fon-gün özellik deposu (S3b-1): snapshot'tan okur → data/features/fund_features.parquet."""
    cfg = load_config()
    st = _store(read_only=True)
    fm = st.latest_fund_master()
    nav = st.nav_wide()
    if nav.empty or fm.empty:
        typer.echo("veri yok — önce `janus ingest tefas --mode initial`")
        raise typer.Exit(code=1)
    out = ROOT / "data" / "features" / "fund_features.parquet"
    summary = _features_build_impl(nav, fm, cfg, st, out, asof=asof, incremental=incremental)
    started = datetime.now()
    try:
        st.log_run(f"features-{started:%Y%m%d-%H%M%S}", "features", started, "ok", summary)
    except RuntimeError:
        # salt-okunur snapshot Store'a yazılamaz (S3b-0b) → özet parquet'in yanına JSON olarak
        (out.parent / "build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        logger.warning("runs kaydı yazılamadı (salt-okunur) → {}", out.parent / "build_summary.json")
    typer.echo(
        f"FEATURES: {summary['n_rows']} satır, {summary['n_funds']} fon, "
        f"incremental={summary['incremental']}, new_dates={summary['new_dates']} → {out}"
    )


@app.command()
def select(
    date: str = typer.Option(..., "--date", help="Karar günü (YYYY-MM-DD)"),
) -> None:
    """B2c-fdr: conformal p-değerleri + BH q20/q10/q30 → data/predictions/selection_<date>.parquet."""
    pred_path = ROOT / "data" / "predictions" / "predictions.parquet"
    cal_path = ROOT / "data" / "predictions" / "calibrated_target_020.parquet"
    for p, name in ((pred_path, "predictions"), (cal_path, "calibrated_target_020")):
        if not p.exists():
            typer.echo(f"{name}.parquet yok — önce `janus predictions build/calibrate`")
            raise typer.Exit(code=1)
    cfg = load_config().get("conformal", {})
    preds = pd.read_parquet(pred_path)
    feats = pd.read_parquet(_features_parquet())
    cal = pd.read_parquet(cal_path)
    out = ROOT / "data" / "predictions" / f"selection_{date}.parquet"
    try:
        summary = _select_impl(date, preds, feats, cal, out, cfg)
    except ValueError as e:
        typer.echo(f"{e}")
        raise typer.Exit(code=1) from e
    typer.echo(f"SELECT {summary['date']}: {summary['n_funds']} fon, q20={summary['n_selected_q20']} seçildi → {out}")


paper_app = typer.Typer(help="Kağıt-ticaret defteri ve önerileri (S5-1)")
app.add_typer(paper_app, name="paper")

bes_app = typer.Typer(help="BES ayağı — EMK evreni, gölge plan (S5-5)")
app.add_typer(bes_app, name="bes")


def _b0_cash_basket(store, cfg: dict[str, Any], date: str | None = None) -> tuple[pd.Timestamp, list[dict[str, Any]]]:
    """B0 sepetini D'ye kadarki NAV ve D'den önceki son fund_master snapshot'ıyla hesapla."""
    from janus.backtest.data import _execution_profiles_asof, cash_proxy_codes  # noqa: PLC0415
    from janus.strategies.hrp import constrain_founder_targets  # noqa: PLC0415

    nav = store.nav_wide()
    if nav.empty:
        raise ValueError("NAV verisi yok; B0 sepeti hesaplanamadı")
    if date:
        d = pd.Timestamp(date).normalize()
    else:
        latest_snapshot = store.con.execute("SELECT max(snapshot_date) FROM fund_master").fetchone()[0]
        d = (
            max(pd.Timestamp(nav.index.max()).normalize(), pd.Timestamp(latest_snapshot).normalize())
            if latest_snapshot
            else pd.Timestamp(nav.index.max()).normalize()
        )
    nav = nav.loc[pd.to_datetime(nav.index).normalize() <= d]
    if nav.empty:
        raise ValueError(f"{d.date()} veya öncesinde NAV verisi yok")
    master_rows = store.con.execute("SELECT * FROM fund_master").df()
    master_rows = _execution_profiles_asof(
        master_rows, d, cfg.get("project", {}).get("runs", {}).get("morning", "09:15")
    )
    if master_rows.empty:
        raise ValueError(f"{d.date()} veya öncesinde fund_master snapshot'ı yok")
    fm = master_rows.drop_duplicates("fund_code", keep="first").copy()
    counts = nav.notna().sum()
    fm["n_nav"] = fm["fund_code"].astype(str).map(counts).fillna(0).astype(int)
    codes = cash_proxy_codes(
        nav,
        fm,
        cfg,
        require_execution_profile=True,
        profile_asof=d,
    )
    if not codes:
        raise ValueError(
            "B0 canlı/PIT sepetinde seçilebilir fon yok: founder_code, info_ok profili ve buy_valor/sell_valor "
            "zorunludur; valörler 2026-09-22 öncesine taşınmaz"
        )
    meta = fm.drop_duplicates("fund_code").set_index("fund_code")
    rows = []
    founder_codes = meta.get("founder_code", pd.Series(dtype="string"))
    equal_targets = pd.Series(1.0 / len(codes), index=codes)
    capped_targets = constrain_founder_targets(
        equal_targets,
        pd.Series(dtype=float),
        founder_codes,
        max_weight=float(cfg["legs"]["tefas"]["constraints"].get("max_weight_per_founder", 0.30)),
        max_funds=int(cfg["legs"]["tefas"]["constraints"].get("max_funds_per_founder", 3)),
    )
    for code in codes:
        item = meta.loc[code]
        series = nav[code].dropna()
        rows.append(
            {
                "fund_code": code,
                "name": item.get("name"),
                "founder": item.get("founder"),
                "weight_pct": float(capped_targets.get(code, 0.0)) * 100.0,
                "buy_valor": item.get("buy_valor"),
                "sell_valor": item.get("sell_valor"),
                "last_nav_date": pd.Timestamp(series.index[-1]).date().isoformat() if not series.empty else None,
                "withholding_rate": item.get("withholding_rate"),
                "last_success_source": item.get("last_success_source"),
            }
        )
    return d, rows


def _has_b2c_evidence(date: str | None) -> bool:
    if not date:
        return False
    selection = ROOT / "data" / "predictions" / f"selection_{pd.Timestamp(date):%Y-%m-%d}.parquet"
    if not selection.exists():
        return False
    selected = pd.read_parquet(selection)
    return bool(selected["selected_q20"].fillna(False).any()) if "selected_q20" in selected else False


def _display_or_unknown(value: Any, formatter=lambda x: str(x)) -> str:
    if value is None or pd.isna(value):
        return "—"
    return formatter(value)


@paper_app.command("basket")
def paper_basket_cmd(
    date: str | None = typer.Option(None, "--date", help="Sepet değerlendirme tarihi (YYYY-MM-DD); varsayılan son NAV"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """B0 nakit sepeti üyelerini D as-of bilgisiyle tablo olarak göster."""
    st = _store(dry_run)
    cfg = load_config()
    try:
        d, rows = _b0_cash_basket(st, cfg, date)
    except ValueError as exc:
        typer.echo(f"PAPER BASKET: {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"B0 NAKİT SEPETİ — {d.date()}")
    if not rows:
        typer.echo("Sepet üyesi bulunamadı (veri/uygun fon yok).")
        return
    headers = [
        "Fon kodu",
        "Ad",
        "PYŞ",
        "B0-only hedef %",
        "Alış valörü",
        "Satış valörü",
        "Son NAV tarihi",
        "Stopaj oranı %",
    ]
    typer.echo("| " + " | ".join(headers) + " |")
    typer.echo("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [
            row["fund_code"],
            _display_or_unknown(row["name"]),
            _display_or_unknown(row["founder"]),
            f"{row['weight_pct']:.2f}",
            _display_or_unknown(row["buy_valor"], lambda x: f"T+{int(x)}"),
            _display_or_unknown(row["sell_valor"], lambda x: f"T+{int(x)}"),
            _display_or_unknown(row["last_nav_date"]),
            _display_or_unknown(row["withholding_rate"], lambda x: f"{float(x) * 100:.2f}"),
        ]
        typer.echo("| " + " | ".join(cells) + " |")
    typer.echo(
        "Not: ağırlıklar B0-only hedefidir; mevcut riskli lotlar ortak PYŞ kapasitesini azaltır. Kalan pay nakittir."
    )
    if any(row.get("last_success_source") == "proxy_ingest_time" for row in rows):
        typer.echo("Uyarı: vekil zaman, PIT kanıtı değil (proxy_ingest_time).")


@paper_app.command("init")
def paper_init(
    capital: float = typer.Option(100.0, help="Başlangıç sermayesi (birim, yüzde raporlanır)"),
    date: str | None = typer.Option(None, "--date", help="Karar günü (YYYY-MM-DD); varsayılan Istanbul tarihi"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Kağıt defteri başlat: sermaye serbest nakittir; ilk B0 alımı proposal/fill ile olur."""
    from janus.paper.core import PaperLedger  # noqa: PLC0415

    st = _store(dry_run)
    cfg = load_config()
    tz = ZoneInfo(cfg.get("project", {}).get("timezone", "Europe/Istanbul"))
    decision_date = pd.Timestamp(date).normalize() if date else pd.Timestamp(datetime.now(tz).date())
    try:
        paper = PaperLedger(st, cfg, capital=capital, asof=decision_date)
        paper.init_capital(capital)
        st.con.execute(
            "CREATE TABLE IF NOT EXISTS janus_pilot_epoch "
            "(epoch_id INTEGER PRIMARY KEY, started_at TIMESTAMP, initialized_at TIMESTAMP, "
            "published_at TIMESTAMP, available_from TIMESTAMP)"
        )
        st.con.execute(
            "INSERT INTO janus_pilot_epoch "
            "SELECT 1, now(), now(), now(), now() WHERE NOT EXISTS "
            "(SELECT 1 FROM janus_pilot_epoch WHERE epoch_id=1)"
        )
        st.con.execute("UPDATE janus_pilot_epoch SET initialized_at=coalesce(initialized_at, now()) WHERE epoch_id=1")
    except ValueError as exc:
        typer.echo(f"PAPER INIT: {exc}")
        raise typer.Exit(code=1) from exc
    candidate_warning = (
        f" | UYARI: {len(paper._b0_codes)} B0 adayı; 3–5 hedefinin altında" if len(paper._b0_codes) < 3 else ""
    )
    typer.echo(
        f"PAPER INIT: capital={capital} cash={paper.ledger.cash:.2f} pozisyon=0 | "
        f"decision_asof={decision_date.date()} meta_asof={paper.meta_asof.date()} nav_asof={paper.nav_asof.date()}"
        f"{candidate_warning}"
    )


@paper_app.command("reset")
def paper_reset_cmd(
    archive_v1: bool = typer.Option(False, "--archive-v1", help="Eski paper_* tablolarını _v1 olarak arşivle"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Yeni pilot epoch'u başlatmak için kağıt tablolarını atomik olarak arşivle."""
    if not archive_v1:
        raise typer.BadParameter("yeni pilot için açıkça --archive-v1 gerekir")
    from janus.paper.archive import archive_paper_v1  # noqa: PLC0415
    from janus.paper.core import _ensure_paper_tables  # noqa: PLC0415

    st = _store(dry_run)
    try:
        archived = archive_paper_v1(st, ensure_schema=_ensure_paper_tables)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"PAPER RESET: başarısız — {exc}")
        raise typer.Exit(code=1) from exc
    typer.echo(f"PAPER RESET: epoch başlatıldı | arşivlenen tablo={len(archived)}")


@paper_app.command("propose")
def paper_propose_cmd(
    date: str = typer.Option(..., "--date", help="Karar günü (YYYY-MM-DD)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    send: bool = typer.Option(False, "--send", help="Telegram'a gönder"),
) -> None:
    """Karar günü için B2c-fdr öneri listesi; kaynak eski/selection yoksa 'tut'."""
    from janus.paper.core import paper_propose  # noqa: PLC0415
    from janus.report.telegram import send_message  # noqa: PLC0415

    st = _store(dry_run)
    cfg = load_config()
    res = paper_propose(st, cfg, date, root=ROOT)
    if res["status"] == "hold":
        typer.echo(f"PAPER PROPOSE {res['date']}: tut — {'; '.join(res['warnings'])}")
    else:
        typer.echo(
            f"PAPER PROPOSE {res['date']}: {res['proposal_id']} | "
            f"{res['n_orders']} emir | risky %{res['risky_weight'] * 100:.1f} | "
            f"meta_asof={res['meta_asof']} nav_asof={res['nav_asof']} | md={res['md_path']}"
        )
    if send:
        send_message(res["telegram"])


@paper_app.command("fill")
def paper_fill_cmd(
    proposal_id: str = typer.Argument(..., help="Öneri kimliği (YYYYMMDD-NN)"),
    pct: float = typer.Option(1.0, "--pct", help="Gerçekleşen oran (0–1)"),
    price_date: str | None = typer.Option(None, "--price-date", help="Dolum fiyat tarihi (YYYY-MM-DD); varsayılan D+1"),
    skip: bool = typer.Option(False, "--skip", help="Öneriyi işlemeden atla"),
    auto: bool = typer.Option(False, "--auto", help="D+1 fiyatıyla kağıt simülasyonu"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Öneriyi deftere işle; 12:00 sonrası veya önceden işlenmişse reddet."""
    from janus.paper.fill import paper_fill  # noqa: PLC0415

    st = _store(dry_run)
    cfg = load_config()
    res = paper_fill(st, cfg, proposal_id, pct=pct, price_date=price_date, skip=skip, auto=auto)
    if res["status"] == "error":
        typer.echo(f"PAPER FILL {proposal_id}: HATA — {res['message']}")
        raise typer.Exit(code=1)
    if res["status"] == "expired":
        typer.echo(f"PAPER FILL {proposal_id}: süresi doldu")
        raise typer.Exit(code=0)
    typer.echo(
        f"PAPER FILL {proposal_id}: {res['status']} | {res.get('n_fills', 0)} fill | tarih={res.get('fill_date', '-')}"
    )


@paper_app.command("reconcile")
def paper_reconcile_cmd(
    date: str = typer.Option(..., "--date", help="Rapor tarihi (YYYY-MM-DD)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Günlük mutabakat: hedef vs gerçek, özdeşlik, liquidation_value."""
    from janus.paper.reconcile import paper_reconcile  # noqa: PLC0415

    st = _store(dry_run)
    cfg = load_config()
    res = paper_reconcile(st, cfg, date, out_dir=ROOT / cfg.get("reporting", {}).get("orders_dir", "reports"))
    typer.echo(
        f"PAPER RECONCILE {res['date']}: identity={res['identity_ok']} | "
        f"max_diff_pp={res['max_weight_diff'] * 100:.1f} | expired={res['n_expired']} | "
        f"md={res['md_path']}"
    )


@paper_app.command("weekly")
def paper_weekly_cmd(
    date: str | None = typer.Option(None, "--date", help="Pazar günü (YYYY-MM-DD); varsayılan bugün"),
    send: bool = typer.Option(False, "--send", help="Telegram'a gönder"),
) -> None:
    """Haftalık gölge raporu ve ADR-20 sayacı."""
    from janus.paper.weekly import paper_weekly  # noqa: PLC0415
    from janus.report.telegram import send_message  # noqa: PLC0415

    st = _store()
    cfg = load_config()
    res = paper_weekly(st, cfg, date=date, out_dir=ROOT / cfg.get("reporting", {}).get("orders_dir", "reports"))
    typer.echo(
        f"PAPER WEEKLY {res['date']}: {len(res['rows'])} portföy | "
        f"kanıt günleri={res['proof_days']} | ADR21={res['adr20']['adr21_ready']} | md={res['md_path']}"
    )
    if send:
        send_message(res["telegram"])


@paper_app.command("kpi")
def paper_kpi_cmd(
    date: str = typer.Option(..., "--date", help="Rapor tarihi (YYYY-MM-DD)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Operasyon KPI raporu: gece/sabah sağlığı, gecikme, mutabakat."""
    from janus.paper.kpi import paper_kpi  # noqa: PLC0415

    st = _store(dry_run)
    cfg = load_config()
    res = paper_kpi(st, cfg, date=date, out_dir=ROOT / cfg.get("reporting", {}).get("orders_dir", "reports"))
    kpis = res["kpis"]
    gate = "sağlandı" if kpis["gate_met"] else ("sağlanmadı" if kpis["gate_met"] is False else "veri yok")
    typer.echo(
        f"PAPER KPI {res['date']}: gate={gate} | "
        f"nightly_ok={kpis['nightly_success_ok']} | morning_ok={kpis['morning_on_time']} | "
        f"reconcile_ok={kpis['reconcile_ok']} | md={res['md_path']}"
    )


def _model_kwargs_from_cpcv() -> dict[str, Any]:
    cpcv = ROOT / "data" / "predictions" / "cpcv_summary.json"
    if cpcv.exists():
        try:
            d = json.loads(cpcv.read_text(encoding="utf-8"))
            bp = d.get("best_params") or {}
            if bp.get("num_leaves") and bp.get("min_data_in_leaf"):
                return {
                    "num_leaves": int(bp["num_leaves"]),
                    "min_data_in_leaf": int(bp["min_data_in_leaf"]),
                    "num_boost_round": 300,
                }
        except Exception:  # noqa: BLE001
            pass
    return {}


def _expected_decision_date(asof: datetime | pd.Timestamp, cfg: dict) -> pd.Timestamp:
    """Beklenen karar günü: beklenen NAV etiketi sonrasındaki ilk configured iş günü."""
    from janus.data.quality import expected_last_nav_date, next_business_day  # noqa: PLC0415

    latest_nav = expected_last_nav_date(asof=asof, cfg=cfg)
    return next_business_day(latest_nav, cfg)


@app.command()
def nightly(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Gece koşusu: ingest → ... → select → paper propose → fill --auto → reconcile → shadow MTM."""
    from janus.data.ingest_evds import ingest_macro  # noqa: PLC0415
    from janus.data.ingest_tefas import ingest_tefas  # noqa: PLC0415
    from janus.data.quality import quality_summary  # noqa: PLC0415
    from janus.paper.core import paper_propose  # noqa: PLC0415
    from janus.paper.fill import paper_fill_auto  # noqa: PLC0415
    from janus.paper.reconcile import paper_reconcile  # noqa: PLC0415
    from janus.paper.shadows import run_shadows, snapshot_shadow_equity  # noqa: PLC0415

    st = _store(dry_run)
    cfg = load_config()
    started = datetime.now()
    steps: dict[str, dict] = {}

    def run_step(name: str, fn, *args, n_key: str = "n_rows", **kwargs) -> dict:
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            if not isinstance(result, dict):
                raise TypeError(f"{name} impl sonuç sözlüğü döndürmedi")
            result = dict(result)
            result.setdefault("status", "ok")
            raw_status = result["status"]
            business_statuses = {
                "proposed": "ok",
                "hold": "ok",
                "filled": "ok",
                "no_proposal": "ok",
                "partially_filled": "partial",
                "pending": "partial",
                "expired": "ok",
                "skipped": "ok",
                "error": "failed",
                "rejected": "failed",
            }
            if raw_status in business_statuses:
                result["business_status"] = raw_status
                result["status"] = business_statuses[raw_status]
            if result["status"] not in {"ok", "partial", "failed", "skipped"}:
                raise ValueError(f"{name} bilinmeyen status={raw_status}")
            result.setdefault("n_rows", result.get(n_key, result.get("n_nav_rows", 0)))
            result.setdefault("elapsed_seconds", round(time.perf_counter() - t0, 3))
            return result
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "failed",
                "n_rows": 0,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
                "error": str(exc),
            }

    res = run_step("ingest", ingest_tefas, st, _client(dry_run), cfg, mode="incremental", n_key="n_nav_rows")
    steps["ingest"] = res
    mres = run_step("macro", ingest_macro, st, _macro_fetcher(dry_run), cfg)
    steps["macro"] = mres
    wide = st.nav_wide()
    fm = st.latest_fund_master()
    q = quality_summary(wide, fm, cfg, eligible=None)
    steps["quality"] = {"status": "ok", "n_rows": 1, **q}
    _export_snapshot(st, cfg, dry_run, "ok" if res["status"] != "failed" and mres["status"] != "failed" else "failed")

    feat_path = ROOT / "data" / "features" / "fund_features.parquet"
    upstream_ok = res["status"] != "failed" and mres["status"] != "failed" and not wide.empty and not fm.empty
    feat_summary = (
        run_step("features", _features_build_impl, wide, fm, cfg, st, feat_path, incremental=True)
        if upstream_ok
        else {"status": "skipped", "n_rows": 0, "elapsed_seconds": 0.0}
    )
    steps["features"] = feat_summary
    pred_path = ROOT / "data" / "predictions" / "predictions.parquet"
    pred_summary = (
        run_step(
            "predictions",
            _predictions_build_impl,
            pd.read_parquet(feat_path),
            pred_path,
            model_kwargs=_model_kwargs_from_cpcv(),
            incremental=True,
            cfg=cfg,
        )
        if feat_summary["status"] in {"ok", "partial"} and feat_path.exists()
        else {"status": "skipped", "n_rows": 0, "elapsed_seconds": 0.0}
    )
    steps["predictions"] = pred_summary

    cal_cfg = cfg.get("conformal", {})
    cal_020_path = ROOT / "data" / "predictions" / "calibrated_target_020.parquet"
    cal_010_path = ROOT / "data" / "predictions" / "calibrated_target_010.parquet"
    preds = pd.read_parquet(pred_path) if pred_summary["status"] in {"ok", "partial"} and pred_path.exists() else None
    feats = pd.read_parquet(feat_path) if preds is not None else None
    for target, label, path in ((0.20, "calibrate_020", cal_020_path), (0.10, "calibrate_010", cal_010_path)):
        summary = (
            run_step(label, _predictions_calibrate_impl, preds, feats, target, path, cal_cfg, incremental=True)
            if preds is not None
            else {"status": "skipped", "n_rows": 0, "elapsed_seconds": 0.0}
        )
        steps[label] = summary

    d = _expected_decision_date(started, cfg)
    sel_path = None
    cal_020 = None
    if steps["calibrate_020"]["status"] in {"ok", "partial"} and cal_020_path.exists():
        cal_020 = pd.read_parquet(cal_020_path)
        sel_path = ROOT / "data" / "predictions" / f"selection_{d:%Y-%m-%d}.parquet"
        available = all(
            isinstance(frame, pd.DataFrame)
            and "decision_at" in frame
            and pd.to_datetime(frame["decision_at"]).dt.normalize().eq(d).any()
            for frame in (preds, feats, cal_020)
        )
        if available:
            sel_summary = run_step("select", _select_impl, str(d.date()), preds, feats, cal_020, sel_path, cal_cfg)
        else:
            sel_summary = {
                "status": "skipped",
                "n_rows": 0,
                "elapsed_seconds": 0.0,
                "expected_decision_at": str(d.date()),
                "reason": "beklenen karar günü için güncel tahmin/özellik/kalibrasyon yok",
            }
    else:
        sel_summary = {
            "status": "skipped",
            "n_rows": 0,
            "elapsed_seconds": 0.0,
            "expected_decision_at": str(d.date()),
            "reason": "beklenen karar günü için kalibrasyon yok",
        }
    steps["select"] = sel_summary
    paper_summary = (
        run_step("paper_propose", paper_propose, st, cfg, str(d.date()), root=ROOT)
        if sel_summary["status"] in {"ok", "partial"}
        else {"status": "skipped", "n_rows": 0, "elapsed_seconds": 0.0}
    )
    steps["paper_propose"] = paper_summary

    # Fill mutabakatı önceki günün önerilerine dayanır; sinyal zincirinden bağımsızdır.
    steps["paper_fill_auto"] = run_step("paper_fill_auto", paper_fill_auto, st, cfg)
    if paper_summary["status"] in {"ok", "partial"}:
        steps["paper_reconcile"] = run_step(
            "paper_reconcile",
            paper_reconcile,
            st,
            cfg,
            str(d.date()),
            out_dir=ROOT / cfg.get("reporting", {}).get("orders_dir", "reports"),
        )
        shadow_result = run_step(
            "shadow_mtm",
            snapshot_shadow_equity,
            st,
            run_shadows(st, cfg),
            pd.Timestamp(wide.index.max()),
        )
    else:
        steps["paper_reconcile"] = {"status": "skipped", "n_rows": 0, "elapsed_seconds": 0.0}
        shadow_result = {"status": "skipped", "n_rows": 0, "elapsed_seconds": 0.0}
    steps["shadow_mtm"] = shadow_result
    from janus.paper.kpi import paper_kpi  # noqa: PLC0415

    kpi_date = str(d.date())
    kpi_res = run_step(
        "paper_kpi",
        paper_kpi,
        st,
        cfg,
        date=kpi_date,
        out_dir=ROOT / cfg.get("reporting", {}).get("orders_dir", "reports"),
        asof=str((started - pd.Timedelta(days=1)).date()),
        decision_asof=kpi_date,
    )
    gate_met = kpi_res.get("kpis", {}).get("gate_met")
    kpi_res.setdefault("gate", "met" if gate_met is True else ("not_met" if gate_met is False else "no_data"))
    steps["paper_kpi"] = kpi_res

    statuses = [step.get("status", "failed") for step in steps.values()]
    overall = "failed" if "failed" in statuses else ("partial" if "partial" in statuses else "ok")
    st.log_run(
        f"nightly-{started:%Y%m%d-%H%M%S}",
        "nightly",
        started,
        overall,
        steps,
    )
    typer.echo(
        f"NIGHTLY {overall}: ingest={steps['ingest']['status']} macro={steps['macro']['status']} "
        f"features={steps['features'].get('status')} predictions={steps['predictions'].get('status')} "
        f"select={steps['select'].get('status')} paper={steps['paper_propose'].get('status')} "
        f"fill={steps['paper_fill_auto'].get('status')} reconcile={steps['paper_reconcile'].get('status')} "
        f"shadows={steps['shadow_mtm'].get('status')} kpi={steps['paper_kpi'].get('status')} "
        f"steps={json.dumps(steps, ensure_ascii=False, separators=(',', ':'))}"
    )


@app.command()
def morning() -> None:
    """Sabah koşusu: son gece koşusunun kalite + kağıt öneri özetini Telegram'a gönderir."""
    from janus.data.quality import is_business_day  # noqa: PLC0415
    from janus.report.telegram import format_quality, send_message  # noqa: PLC0415

    st = _store()
    started = datetime.now()
    cfg = load_config()
    if not is_business_day(started, cfg):
        st.log_run(
            f"morning-{started:%Y%m%d-%H%M%S}",
            "morning",
            started,
            "ok",
            {"business_status": "no_new_data", "sent_ok": None, "sent_at": None},
        )
        typer.echo("MORNING ok: no_new_data (iş günü değil)")
        st.close()
        return
    last = st.last_run("nightly")
    if not last:
        sent_ok = send_message("JANUS sabah — gece koşusu bulunamadı (runs boş).")
        sent_at = datetime.now()
        st.log_run(
            f"morning-{started:%Y%m%d-%H%M%S}",
            "morning",
            started,
            "ok" if sent_ok else "failed",
            {"error": "no nightly run", "sent_ok": bool(sent_ok), "sent_at": sent_at.isoformat()},
        )
        raise typer.Exit(code=1)
    q = dict(last["summary"].get("quality", {}))
    q["macro_status"] = last["summary"].get("macro", {}).get("status")
    pp = last["summary"].get("paper_propose", {})
    expected_date = _expected_decision_date(started, cfg)
    proposal_is_current = (
        pp.get("status") in {"ok", "partial"} and pd.Timestamp(pp.get("date")).normalize() == expected_date
        if pp.get("date")
        else False
    )
    orders_text = (
        pp.get("telegram", "Emir: yok (gece önerisi üretilmedi).")
        if proposal_is_current
        else "Emir: yok (beklenen karar günü için güncel öneri yok)."
    )
    basket_date = str(expected_date.date())
    if not _has_b2c_evidence(basket_date):
        try:
            _, basket_rows = _b0_cash_basket(st, cfg, basket_date)
            basket_codes = ", ".join(row["fund_code"] for row in basket_rows) or "sepet üyesi bulunamadı"
        except ValueError:
            basket_codes = "sepet verisi yok"
        orders_text += f"\nkanıt yok → sepet: {basket_codes}"
    head = (
        f"JANUS sabah — {datetime.now():%Y-%m-%d %H:%M} | gece koşusu {last['status']} ({last['finished_at']:%H:%M})\n"
    )
    sent_ok = send_message(head + format_quality(q) + "\n" + orders_text)
    sent_at = datetime.now()
    st.log_run(
        f"morning-{started:%Y%m%d-%H%M%S}",
        "morning",
        started,
        "ok" if sent_ok else "failed",
        {
            "nightly_run_id": last["run_id"],
            "nightly_status": last["status"],
            "sent_ok": bool(sent_ok),
            "sent_at": sent_at.isoformat(),
        },
    )
    st.close()


@bes_app.command("build")
def bes_build_cmd(
    incremental: bool = typer.Option(False, "--incremental", help="Mevcut BES parquet'lerinden devam et"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Sahte istemci modu değil; BES build'i test depoda çalıştır"),
) -> None:
    """BES feature / prediction / calibration boru hattı (EMK evreni)."""
    st = _store(dry_run)
    cfg = load_config()
    try:
        summary = bes_build(st, cfg, ROOT, incremental=incremental)
    except Exception as e:  # noqa: BLE001
        typer.echo(f"BES BUILD HATA: {e}")
        raise typer.Exit(code=1) from e
    typer.echo(
        f"BES BUILD: features={summary['features']['n_rows']} satır, "
        f"predictions={summary['predictions']['n_pred_rows']} satır, "
        f"calibrate={summary['calibrate']['n_rows']} satır"
    )


@bes_app.command("plan")
def bes_plan_cmd(
    date: str = typer.Option(..., "--date", help="Plan tarihi (YYYY-MM-DD)"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """BES aylık plan: B2c-fdr seçimi + değişiklik sayacı; gölge rapor."""
    st = _store(dry_run)
    cfg = load_config()
    res = bes_plan_impl(st, cfg, date, root=ROOT)
    if res["status"] == "error":
        typer.echo(f"BES PLAN HATA: {res['message']}")
        raise typer.Exit(code=1)
    if res["status"] == "limit_reached":
        typer.echo(f"BES PLAN {date}: {res['message']}")
        raise typer.Exit(code=0)
    if res["status"] == "no_change":
        typer.echo(
            f"BES PLAN {res['date']}: değişiklik yok (hak harcanmadı) | "
            f"seçilen={res['n_selected']} / {res['n_available']} | md={res['md_path']}"
        )
        raise typer.Exit(code=0)
    typer.echo(
        f"BES PLAN {res['date']}: change_id={res['change_id']} | "
        f"seçilen={res['n_selected']} / {res['n_available']} | md={res['md_path']}"
    )


if __name__ == "__main__":
    app()
