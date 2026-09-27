# TECH_RULES.md — teknik standartlar (M4 Pro / Python 3.13) — sürüm 1.1 (23.09.2026)

## Donanım ve çerçeve
- Apple M4 Pro, 24 GB. PyTorch **MPS** (`PYTORCH_ENABLE_MPS_FALLBACK=1`); **CUDA yok**; `tensorflow-gpu` asla.
- Model boyutu sınırlı tutulur; parametre sayısı koddan üretilir (`sum(p.numel())`), belgeye elle yazılmaz.

## Kod
- **Büyük tablolarda** satır döngüsü yok (`iterrows`, `apply(axis=1)` üzerinden hesap): pandas/numpy vektörize veya DuckDB SQL.
- **İzinli sıralı döngüler:** zamana bağlı durum makineleri (defter, olay işleme, rejim DP, ACI) gün üzerinde döngü kurabilir; fon boyutu vektörizedir. Performans bütçesi: tam panel backtest ≤ 5 s, özellik deposu ≤ 60 s.
- Hesap çekirdeği (özellikler, defter, modeller) **saf**: ağ, dosya, MLflow, Telegram erişimi yalnızca adaptör katmanlarında (`data/tefas_client`, `data/store`, `backtest/mlflow_utils`, `report/`). Test edilen çekirdek ağa/dosyaya dokunmaz.
- Tip ipuçları; ruff satır 120; her modül için test; sayısal iddia test/rapordan.
- Konfig `config/janus.yaml` + `.env`; sabit kod içinde değil.

## Zaman ve veri (docs/TEMPORAL_PROTOCOL.md kazanır)
- As-of: karar `D` yalnızca `source_available_at ≤ D` gözlemleri görür (TEFAS NAV: `observed_date ≤ D−1`). Gecikmeler işlem günü cinsinden.
- X ve y maskeleri ayrı; etiket olgunlaşması (`label_available_at`) eğitim/kalibrasyon/kill-switch sınırıdır.
- Uydurma yok: NaN/0 NAV → bayrak + o gün evren dışı; forward-fill yalnızca değerleme için (son bilinen NAV) ve ADR ile.
- Tarihçesi olmayan meta alanlar tarihsel özellik olmaz; yarı-sabit alanlar "A3 sabit varsayımı" etiketiyle.
- Sızıntı testleri: beyaz liste + gelecek perturbasyonu + model sürümü kaydı (TEMPORAL §9). Negatif kontrol kanıt sayılmaz.

## Defter (docs/LEDGER_EXECUTION_SPEC.md kazanır)
- `can_buy` / `can_sell` / `data_stale` / `suspended` ayrı bayraklar; haircut yalnızca stres/tasfiye değeri.
- Vergi satışta bir kez, lot oranı alım tarihine göre; nakit vekili defter pozisyonu.
- Raporlanan her metrik vergi ve maliyet sonrası; farklar **pp**; deney kimliği zorunlu.

## Doğrulama (docs/VALIDATION_PROTOCOL.md kazanır)
- Seçim dönemi ≠ dış test; 12 başlangıç duyarlılıktır; tohum yalnızca stokastik modüllerde; PBO/DSR taramalarda; CPCV hiperparametre seçiminde.

## Git ve doküman
- Dallar `main` (korumalı), `feat/*`. pre-commit (gitleaks, detect-secrets, ruff) zorunlu.
- Karar → `docs/decisions/ADR-xxxx.md`; sözleşme → `docs/*_PROTOCOL.md` / `docs/*_SPEC.md`.
