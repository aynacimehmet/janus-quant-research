# ADR-0030: Canonical PIT immutability istisnası: y için tek yönlü NaN→değer label maturation
- Durum: Kabul (sistem sahibi, 26.09.2026)
- Tarih: 2026-09-26
- Sahip persona: Quant Scientist + Data Scientist

## Bağlam
ADR-0028 tarihsel canonical PIT (point-in-time) önekinin dokunulmazlığını ve değişimde fail-closed durmayı
kalıcılaştırdı. ADR-0029 m.1 bu kuralı genişletti: overlap-öncesi tarihsel canonical prefix tüm kolon, değer ve
dtype bakımından korunur; maddeleşmiş bir değer (`y` dahil) değişirse ret; kolon kümesi veya dtype değişirse
ret. ADR-0029 m.3 aynı zamanda `y` kolonunda **yalnız tek yönlü olgunlaşma** (existing `NaN` → recomputed
değer) iznini tanımladı; materyalize `y` değerinin değişimi yine rettir.

Bu istisna kalıcı bir politikadır ve tek başına bir ADR'de sabitlenmelidir. Liveness gerekçesi: etiketler
(`y`) zamanla olgunlaşır — karar gününde henüz oluşmamış ileri getiri, takip eden koşularda NaN'dan değere
döner. Bu izin olmasaydı her gece bir etiket olgunlaştıkça artımlı zincir sürekli fail-closed olur ve canlı
hat hiç ilerleyemezdi. Aynı anda ADR-0028'in PIT dokunulmazlığı çekirdeği (`y` dışındaki tüm canonical
kolonlar) korunur. Bu ADR yeni bir veri kaynağı, migration veya label-revision politikası **icat etmez**;
yalnız ADR-0029 m.3'teki istisnayı kalıcılaştırır.

## Seçenekler (artı / eksi)
| Seçenek | Artı | Eksi |
|---|---|---|
| **A. `y` için tek yönlü NaN→değer maturation (koşullu, seçilen)** | Olgunlaşan etiketlerle artımlı zincir canlı kalır; PIT dokunulmazlığı `y` dışında korunur; değer değişimi hâlâ fail-closed | Yalnız `label_available_at` mevcutken güvenli; meta alan eksikse maturation durur |
| B. Tüm canonical prefix'i mutlak immutable tut | En katı PIT kaydı | Olgunlaşan `y` her gece fail-closed yaptırır; canlı hat pratikte ilerleyemez |
| C. `y` için serbest rewrite | Zincir kesintisiz sürer | Maddeleşmiş etiket değişimi/silme sessizce PIT tutarsızlığı yaratır; fail-closed kaybı |

## Karar
1. **Genel immutable:** Historical canonical PIT prefix genel olarak immutable kalır (ADR-0029 m.1 ile
   uyumlu).
2. **Tek istisna:** YALNIZ `y` (etiket) kolonu için `NaN → değer` yönünde label maturation (etiket
   olgunlaşması) izinlidir. Bu, ADR-0029'daki "`y` yalnız tek yönlü olgunlaşma" istisnasının
   kalıcılaştırılmasıdır.
3. **Koşul:** İzin ancak `label_available_at` (temporal sözleşme meta alanı) erişilebilir/mevcut olduğunda
   geçerlidir; `label_available_at` yoksa maturation izni yoktur, **fail-closed**.
4. **Fail-closed olanlar:** `değer → farklı değer`, `değer → NaN`, dtype değişimi, schema/kolon kümesi
   değişimi. Bunların tümü rettir.
5. **Diğer kolonlar:** `y` dışındaki historical canonical kolonlar ADR-0029 kapsamında immutable kalır.
6. **Kapsam dışı:** Controlled rebuild / backfill / label revision politikası bu ADR'de **tanımlanmaz**;
   ADR-0028'in ayrı-lineage/promotion kapısına devredilir.

## Sonuçlar (kod, konfig, test, risk)
- **Uygulama:** ADR-0029 m.3'te tanımlanan `y` yönlü karşılaştırma bu ADR ile kalıcı sözleşme olur; kod
  yolu ADR-0029 uygulamasındaki gibidir (bu ADR kod değişikliği gerektirmez).
- **Liveness:** `y` olgunlaşması tek yönlü tamamlanma olarak yazılır; aksi halde her gece bir etiket
  olgunlaştığı için artımlı zincir sürekli fail-closed olurdu. Materyalize `y` değişimi yine rettir.
- **Koşulluluk:** `label_available_at` mevcut değilse maturation izni verilmez; fail-closed korunur.
- **Kapsam dışı borç:** Controlled rebuild / backfill / label revision politikası bu ADR'de tanımlanmaz;
  ADR-0028'in ayrı-lineage/promotion kapısına devredilir.
- **Doğrulanamayan:** `label_available_at` meta alanının historical satırlarda geriye dönük doluluğu
  kanıtlanmış değildir; bu ayrı bir borçtur ve uydurma veri ile kapatılmaz.
- **Kabul:** Sistem sahibi 26.09.2026'da onayladı; durum "Kabul". S5 kapanışı bu ADR ile açılmaz.
