# janus-quant-research

[English](README.md) | **Türkçe**

Araştırma odaklı, gün sonu (EOD) çalışan nicel bir portföy motoru. Referans
kurulum Türkiye yatırım fonlarını (TEFAS) ve emeklilik fonlarını (BES)
hedefler; defter, icrayı backtest ile kağıt-ticarette aynı biçimde modeller.

Bu, iç bir araştırma sisteminin public kullanım için temizlenmiş anlık
görüntüsüdür. **Piyasa verisi**, **kişisel konfigürasyon** ve **operasyon
kaydı** içermez. Yerel bir sürüm adayıdır; hiçbir uzak depoya gönderilmemiştir.

> Araştırma yazılımıdır; yatırım tavsiyesi değildir. Motor yalnızca günlük emir
> **listesi** üretir; emirler bir insan tarafından manuel girilir.

## Kapsam

- `src/janus/` — motor: veri alımı, özellik deposu, kuantil modeller,
  conformal kalibrasyon/seçim, HRP tahsisi, defter-hizalı backtest,
  kağıt-ticaret ve raporlama.
- `tests/` — deterministik sentetik üreteçlerle çalışan birim/entegrasyon
  test paketi.
- `config/janus.yaml` — örnek konfigürasyon (temizlenmiş; kurgusal şirket
  adları).
- `docs/` — metodoloji sözleşmeleri ve seçili mimari karar kayıtları (ADR).

## Metod öne çıkanları (bu anlık görüntüde uygulanmış)

- **Point-in-time (as-of) veri modeli.** Kaynak-erişilebilirlik ekseni, `D`
  günündeki kararın neyi görebileceğini belirler; özellik (`X`) ve etiket
  (`y`) maskeleri ayrıdır; sızıntı bir özellik beyaz listesi, gelecek-
  perturbasyon testi ve tahmin başına model sürümü kaydıyla kontrol edilir.
- **Conformal seçim.** CQR aralıkları ve ACI-tarzı günlük `alpha` güncellemesi;
  conformal p-değerleri ile Benjamini-Hochberg FDR adımı (`q = 0,20`).
  Kalibrasyon ve risk bantları ayrıdır.
- **Hierarchical Risk Parity (HRP).** Ledoit-Wolf kovaryans, fon / kurucu /
  küme tavanları ve vergi-farkındalıklı rebalancing sapma eşikleri.
- **Defter-hizalı backtest.** FIFO lotları, tarihe göre stopaj, alış/satış
  valörü ve settlement; ayrı `can_buy` / `can_sell`, `data_stale` ve
  `suspended` bayrakları.
- **Doğrulama.** Seçim dönemi ile dış test ayrımı, duyarlılık için 12 başlangıç,
  purge/embargo walk-forward ve CPCV, parametre taramalarında PBO/CSCV ve
  deflated Sharpe, stokastik modüllerde 5 tohum.
- **Modeller.** Global LightGBM kuantil modelleri ve jump/HMM rejim modülü.
- **Mühendislik.** Typer CLI, DuckDB/Parquet depolama, MLflow deney takibi,
  pytest + ruff + pre-commit.

## Kapsam dışı (gelecek çalışma)

Uyarlanabilir bir maruziyet katmanı — olası bir pekiştirmeli öğrenme
denetleyicisi dahil — gelecek araştırmasıdır. Bu anlık görüntüde hiçbir RL
ortamı veya ajanı uygulanmamıştır.

## Kurulum

```bash
conda env create -f environment.yml
conda activate janus
pip install -e ".[dev]"
cp .env.example .env   # kendi anahtarlarınızı doldurun; yerel dosyayı asla commit etmeyin
pre-commit install
janus doctor                  # ortam / paket / MPS / sır kontrolü
```

## Kullanım

```bash
janus doctor
janus ingest --help
janus backtest
janus backtest-suite
janus gate-suite
```

## Test ve kalite kapıları

```bash
pytest -q
ruff check src tests
pre-commit run --all-files
```

Tüm testler sentetik veriyle çalışır; piyasa verisi veya ağ erişimi gerektirmez.

## Veri ve gizlilik

- Piyasa verisi, defter, kişisel path, sır veya operasyon kaydı içermez.
- Konfigürasyon kurgusal şirket adları kullanır.
- Yalnızca `tests/` ve `sample_data/` altındaki sentetik örnekler kullanılır.

## Lisans

MIT Lisansı ile yayımlanır; bkz. `LICENSE`. Üçüncü taraf bağımlılık lisansları
`THIRD_PARTY_NOTICES.md` içinde listelenmiştir.
