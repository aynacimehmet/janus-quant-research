# ADR-0023: A3 yürütme alanlarının ilk PIT öncesine taşınması
- Durum: kabul
- Tarih: 2026-09-25 · Sahip persona: PO / sistem sahibi

## Bağlam
S5-6c B0 karşılaştırması, 2026-09-22 tarihli ilk gerçek PIT snapshot öncesinde geçmiş icra edilebilirliğini tanımlamayı gerektiriyor. Bu alanların geçmiş kaydı yoktur; bugünkü profilin geçmişe taşınması PIT kanıtı veya model girdisi sayılamaz. TEMPORAL_PROTOCOL §8'deki genel tarihçesiz meta yasağı için yalnız dar bir yürütme istisnası karara bağlanmalıdır.

## Seçenekler (artı / eksi)
| Seçenek | Artı | Eksi |
|---|---|---|
| A. Yürütme alanlarını ilk PIT öncesine A3 varsayımıyla taşımak; kapsamı ve etiketi sınırlamak | Geçmiş B0 icra simülasyonuna açık varsayım sağlar; A3 ile PIT kanıtını ayırır | Geçmiş sonuçlar gerçek tarihsel icra verisi değildir; varsayım yanlılığı ve hayatta kalan evren riski sürer |
| B. Yürütme alanlarını geçmişe taşımamak | Tarihsel icra bilgisi uydurulmaz; PIT sınırı yalın kalır | İlk PIT öncesi icra karşılaştırması yapılamaz veya yürütme alanları eksik kalır |

## Karar
Seçenek A kabul edilmiştir; istisna yalnızca yürütme alanlarını kapsar: alış/satış valörü, TEFAS işlem durumu metni ve komisyon. Yalnızca 2026-09-22 ilk PIT snapshot öncesindeki tarihler için mevcut profil A3 varsayımıyla geriye taşınabilir; bu tarihlerde status “işlem görüyor” varsayılır. Sonuç satırları açıkça **“A3 varsayımlı, PIT kanıtı değil”** diye etiketlenir.

Komisyon alanına ilişkin bu hüküm, ADR-0027 ile supersede edilmiştir: TEFAS/BES'te komisyon yürütme maliyeti değildir ve A3 varsayımı/provenance'ı üretilmez.

Canlı/PIT seçim ve icrada işlem yönlerinin tek doğruluk kaynağı `tefas_status` metnidir (`quality.trade_status_masks` + bilinmeyen status için strict tanınma kapısı). Saklanan `can_buy`/`can_sell` alanları uyumsuzluk uyarısı/audit bilgisidir, işlemi tek başına bloke etmez. Tanınmayan/boş status iki yönde fail-closed kalır; tanınan tek yönlü kapalı status yalnız o yönü kapatır.

Bu istisna bu alanları `FEATURE_COLUMNS`'a, tarihsel X'e, model eğitimine/çıkarımına veya kanıt iddiasına eklemez; yalnız yürütme/işlem yapılabilirlik simülasyonunda kullanılabilir. Bugünkü PYŞ kara listesi ayrı bir **bugünkü politika** etiketidir; tarihsel PIT bilgisi değildir.

2026-09-22 ve sonrasında karar günü D 09:15 itibarıyla en son erişilebilir fon snapshot'ı, `snapshot_date ≤ D` ve `source_published_at ≤ D 09:15` olan snapshot'lar arasından seçilir. D 09:15 sonrasında yayımlanan yeni snapshot as-of dışındadır; bu durumda önceki erişilebilir snapshot en son erişilebilir snapshot olarak kalır. Snapshot seçildikten sonra profilin `last_success_at ≤ D 09:15`, D 09:15 itibarıyla en fazla 7 takvim günü yaş ve gerekli alanları (alış/satış valörü, tanınan `tefas_status`, `tax_category`) taşıdığı doğrulanır. Saklanan yön bayrakları metin maskesiyle uyuşmazsa uyarı kaydedilir; maskeyi değiştirmez. Valör/status/tax eksik veya profil eskiyse ilgili fon için işlem yapılmaz; gerekçe fon bazında kaydedilir ve değerleme son mevcut NAV ile sürer. Daha eski snapshot'a fallback yapılmaz. Yayın timestamp'i yoksa date-only varsayımı açıkça etiketlenir ve PIT kanıtı sayılmaz. Gelecek profil/meta kullanılamaz.

ADR-27, bu ADR'nin yalnızca komisyon/eksik-fee hükmünü supersede eder. TEFAS/BES execution fee deterministik sıfırdır; ham `entry_fee`/`exit_fee` execution maliyeti değildir; fee eksikliği/geçersizliği yürütme engeli veya provenance üretmez. Diğer profil/as-of/valör/status/tax hükümleri aynen sürer.

Eski/yeni B0 karşılaştırmasında **“satış valörü +1” stres satırı zorunludur**. Bu ADR performans sonucu veya tarihsel icra kanıtı ileri sürmez.

## Sonuçlar (kod, konfig, test, risk)
- Kod/konfig: bu ADR yalnız dokümantasyon kararıdır; uygulama S5-6c kapsamında yapılır. TEMPORAL_PROTOCOL §8 ve VALIDATION_PROTOCOL §4 bu sınırı yansıtır.
- Test: uygulamada ilk PIT öncesi A3 etiketinin/X dışılığının, post-PIT snapshot ve 7 takvim günü tazeliğinin, eksik/eski profilde fon-bazında fail-closed davranışının, son NAV değerlemesinin ve zorunlu B0 satış-valörü +1 stres satırının test edilmesi gerekir.
- Risk: A3 varsayımı gerçek geçmiş koşulları kanıtlamaz; sonuçlar bu etiketle raporlanmalı, bugünkü PYŞ kara listesi ayrı tutulmalı ve stres satırı atlanmamalıdır.
- Kabul tarihi/id: **2026-09-25, ADR-0023**.
