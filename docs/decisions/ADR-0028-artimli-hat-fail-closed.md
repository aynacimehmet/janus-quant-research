# ADR-0028: Artımlı hatta overlap-öncesi revizyonda fail-closed ve PIT öneki dokunulmazlığı
- Durum: Kabul (sistem sahibi, 26.09.2026)
- Tarih: 2026-09-26
- Sahip persona: Quant Scientist

## Bağlam
S5-0b artımlı hattı (`features/predictions/calibrate --incremental`) günlük append için tasarlandı; ancak
overlap-öncesi (eski) tarihsel prefix değiştiğinde ne olacağı sözleşmede tanımsızdı. Bu değişimin nedeni tek
bir sınıfa indirgenemez: NAV/veri revizyonu, PIT kanıtı olmayan yeni fon/makro geçmişi (PIT'siz yeni geçmiş)
veya implementation/schema/model sürümü değişimi olabilir. **Tetikleyici, nedenden bağımsız olarak "prefix
değişti mi" sorusudur**; neden sınıfı yalnız denetim (audit) kaydıdır, fail-closed kapısının koşulu değildir
ve bir neden diğerine zorlanmaz. TEMPORAL §9.2 yalnız "geçmiş A3 değerleri yeniden yazılmaz" der; değişim
halini tanımlamaz. S5-6e (F17) bu boşlukta iki hedefi çakıştırdı: (i) incremental=full bit-eşitlik, (ii) ACI
`alpha_D` öneki append-only. Eski kod yalnız max-date'e bakıp erken çıkıyor, değişimi sessizce atlıyordu.
Sistem sahibi seçenek A'yı seçti; bu ADR onu kalıcılaştırır ve aynı tarihte kabul edilmiştir.

## Seçenekler (artı / eksi)
| Seçenek | Artı | Eksi |
|---|---|---|
| **A. Fail-closed + dokunulmaz önek (seçilen)** | PIT kaydı bozulmaz; sessiz veri kaybı yok; hata operatör görür; incremental=full temiz append'te korunur | Revizyonlu gecede canlı zincir durur; operatör müdahalesi gerekir |
| B. Frozen prefix + sessiz devam | Zincir kesintisiz sürer | Yeniden hesaplanan yeni günler revize geçmişle karışır; sessiz PIT tutarsızlığı; hata görünmez |
| C. Otomatik tam rebuild | Zincir sürer, çıktı self-consistent | Kaydedilmiş `alpha_D` ve PIT öneki yeniden yazılır (append-only ihlali); geri dönüşsüz |

## Karar
Overlap-öncesi tarihsel prefix'te — nedeni ne olursa olsun — değişim tespit edilirse **incremental/canlı
zincir fail-closed durur**; **geçmiş PIT öneki asla yeniden yazılmaz**. Değişimin giderilmesi operatörün
bilinçli kararıyla **ayrı, kontrollü bir full rebuild** yolundan yapılır; otomatik/sessiz rebuild yoktur.
Kontrollü full rebuild bile **mevcut kanonik PIT/alpha geçmişini yerinde (in-place) overwrite edemez**:
yeniden üretim **ayrı bir lineage/version** olarak yazılır ve kanonik geçmiş dokunulmadan kalır. Yeni hattın
üretime alınması bu ADR kapsamında değildir; ayrı ve açık bir **promotion/migration kararı** gerektirir.
Seçenek B ve otomatik/sessiz overwrite reddedilmiştir. Neden sınıfı (veri revizyonu, PIT'siz yeni geçmiş,
implementation/schema/sürüm değişimi) karar mantığını değiştirmez; yalnız audit kaydında ayrıştırılır.

## Sonuçlar (kod, konfig, test, risk)
- **Uygulama:** `src/janus/cli.py` — `_frames_equal`, `_time_prefix`, `_train_max_t_ok`; feature/prediction
  `revision_detected` → `RuntimeError`; kalibrasyonda `alpha_D/lower/upper/n_calib/quality_flag` öneki
  existing'ten korunur, değişirse `RuntimeError`; `nav_max < existing_max` → `status="stale"`.
- **İlişki:** Temiz append'te prefix değişmez → `incremental == full` (bit düzeyi) ve `alpha_D` öneki
  değişmez (append-only) birlikte sağlanır. Prefix değişiminde ikisi çelişir; A, full-eşitliği askıya alıp PIT
  dokunulmazlığını üstün tutar. Yeni karar günlerinin α'sı yalnız prefix değişmediği doğrulandıktan sonra
  eklenir.
- **Lineage ve neden sınıfı:** Değişimin nedeni (NAV/veri revizyonu, PIT'siz yeni geçmiş, implementation/
  schema/sürüm değişimi) tetikleyiciyi değiştirmez; hepsi aynı fail-closed kapısına girer. Kontrollü full
  rebuild çıktısı kanonik geçmişi yerinde değiştirmez, ayrı lineage/version üretir; üretime geçiş ayrı bir
  promotion/migration kararıdır.
- **Test:** `tests/test_cli_incremental.py` F17 regresyonları (10 test). Sentetik ve salt-okunur duman testlerinde fail-closed yolu doğrulandı; canonical prefix byte düzeyinde korundu.
- **Konfig:** Yeni anahtar yok. **Risk:** Prefix değişimi olan gecelerde `select`/`paper propose` durur;
  `check_dtype=True` schema kaymasını da aynı kapıya sokar.
- **Borçlar:**
  - `train_max_t` işlem-günü takvim eşdeğerliği — S5-R2 zorunlu kontrol kapsamı **KAPALI (26.09.2026):** sentetik ve salt-okunur duman testleri geçti. Bu duman yalnız takvim-dışı NAV günü bulunmadığını gösterir; **tam takvim eşdeğerliği doğrulanamadı** (teknik kapanışı bloklamaz) ve eksik-gün yönü muhafazakâr kalır.
  - Lineage/version saklama yeri/şeması ve promotion/rollback yolu bu ADR'de verilmez; kontrollü full rebuild üretime alınmadan önce **ayrı bir karar** (ADR) gerektirir — **AÇIK**.
- **Kabul:** Sistem sahibi 26.09.2026'da onayladı; durum "Kabul".
