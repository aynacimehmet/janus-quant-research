# ADR-0029: Canonical PIT prefix bütünlüğü, terminal doğrulama sırası ve panel takvim fail-closed
- Durum: Kabul (sistem sahibi, 26.09.2026)
- Tarih: 2026-09-26
- Sahip persona: Quant Scientist + Data Scientist

## Bağlam
S5-R2 bağımsız incelemesi ADR-0028’in
“overlap-öncesi tarihsel prefix değişirse fail-closed; geçmiş PIT/alpha öneki yeniden yazılmaz” kararının
üç kod yolunda delindiğini gösterdi:
- **A:** `_predictions_build_impl` `_train_max_t_ok`’u `to_parquet`’tan **sonra** çalıştırıyor; D−23 ret olsa
  da canonical tahmin dosyası önce değişiyor (append + full).
- **B:** `_predictions_calibrate_impl` guard’ı yalnız `alpha_D/lower/upper/n_calib/quality_flag` önekini
  karşılaştırıyor; `q10/q50/q90/y/train_max_t/model_id` değişip α kolonları sabit kalırsa eski önek sessizce
  eziliyor.
- **B1:** BES’te kapsam dışı fon satırları `bes/build.py` ve `_features_build_impl` içinde guard’lardan
  **önce** yerinde budanıp yazılıyor.
- **B2:** Panel tarihleri configured iş günü eksenine karşı doğrulanmıyor; panel’e giren işlem günü olmayan
  sahte tarih D−23 guard’ını zayıflatıyor (`_train_max_t_ok` takvimi `feature_asof ∪ decision_at`).

Sistem sahibi 26.09.2026’da aşağıdaki beş kararı kabul etti. Bu ADR yeni bir takvim, veri kaynağı veya
migration politikası **icat etmez**; yalnız mevcut sözleşmeyi (ADR-0028, TEMPORAL §1/§4/§9.3) kapatır.

## Karar
1. **Historical canonical PIT prefix immutable:** overlap-öncesi tarihsel canonical prefix tüm kolon, değer
   ve dtype bakımından korunur. Maddeleşmiş bir değer (materyalize `y` dahil) değişirse ret; kolon kümesi veya
   dtype değişirse ret.
2. **Prediction terminal validation write’tan önce:** `_train_max_t_ok` canonical `to_parquet`’tan **önce**
   tamamlanır. Append ve full ret yollarında eski canonical dosya değişmeden kalır.
3. **Calibration guard tam canonical prefix:** kalibrasyon guard’ı `q10/q50/q90/y/train_max_t/model_id` dahil
   tüm canonical öneki karşılaştırır; `alpha_D/lower/upper/n_calib/quality_flag` ADR-0028 gereği append-only
   olarak existing’ten korunur ve değişirse ret. Değer/kolon/dtype farkında yazmadan fail-closed.
4. **BES kapsam değişimi canonical’ı guard öncesi budayamaz:** kapsam pruning/write işlemi guard öncesinden
   kaldırılır. Kapsam değişiminde canonical geçmiş yerinde rewrite edilmez; fail-closed durur. Migration
   gerekiyorsa ADR-0028’in ayrı-lineage/promotion kapısına devredilir; bu ADR yeni BES politikası belirlemez.
5. **Panel takvimi fail-closed:** panel tarihleri mevcut configured iş günü eksenine (`is_business_day`,
   `config/janus.yaml:calendar`) karşı fail-closed doğrulanır. Eksik gün yönü **gevşetilmez**; D−23 semantiği
   **değiştirilmez**; terminal karar günü istisnası mevcut sözleşmeyle uyumlu korunur (terminal
   `next_business_day(cal[-1], cfg)` configured iş günüdür).

## Uygulama (kod, test)
- `src/janus/cli.py`:
  - `_predictions_build_impl`: `_train_max_t_ok` `to_parquet` öncesine taşınır (append + full).
  - `_predictions_calibrate_impl`: guard tam canonical prefix’e genişletilir; `y` için yalnız tek yönlü
    etiket olgunlaşması (existing NaN → recomputed değer) izinlidir, maddeleşmiş değer değişimi rettir.
    Ret mesajı ADR-0028’e uygun güvenli dur/eskalasyon olur (yerinde full rebuild önerisi kaldırılır).
  - `_features_build_impl`: BES kapsam budama/yazımı guard öncesinden kaldırılır; panel tarih doğrulaması
    stale/no-op/canonical yazımından önce eklenir (her iki leg).
  - `_predictions_build_impl`: isteğe bağlı `cfg` ile feature/karar tarihleri aynı configured takvime karşı
    doğrulanır; CLI ve BES caller’ları tam cfg geçirir.
- `src/janus/bes/build.py`: predictions/calibration’ı guard’lardan önce budayan bloklar kaldırılır.
- Testler: A (append+full D−23 ret → canonical byte/değer+dtype sabit), B (q50-only ve train_max_t-only
  prefix perturbasyonu → ret + canonical sabit), B1 (BES scope change → ret + canonical sabit), B2
  (hafta sonu/takvim-dışı sahte gün → RED→GREEN; D−23 sınırı ve terminal istisnası korunur).

## Sonuçlar / risk
- **Liveness:** `y` olgunlaşması tek yönlü tamamlanma olarak yazılır; aksi halde her gece bir etiket
  olgunlaştığı için artımlı zincir sürekli fail-closed olurdu. Materyalize `y` değişimi yine rettir.
- **Kapsam dışı borç:** Başarılı **non-incremental** koşunun mevcut canonical’ı yerinde ezmesi ADR-0028’e
  aykırıdır ve bu ADR onu izinli saymaz; ayrı controlled-rebuild/lineage kararı borcudur.
- **Doğrulanamayan:** Configured takvim gerçek NAV seanslarıyla eşdeğerliği kanıtlanmış değildir (tatil
  listesi eksik olabilir). B2 kirlilik yönünü kapatır; fiilî eşdeğerlik kullanıcının salt-okunur takvim
  dumanıyla ayrıca doğrulanır.
- **Kabul:** Sistem sahibi 26.09.2026’da onayladı; durum “Kabul”. S5 kapanışı bu ADR ile açılmaz; bağımsız
  VALIDATION ve gerçek-veri kapıları ayrıdır.
