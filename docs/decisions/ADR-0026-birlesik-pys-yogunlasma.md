# ADR-0026: Birleşik PYŞ yoğunlaşma kısıtı — B0 + riskli
- Durum: Kabul (kullanıcı, 25.09.2026)
- Tarih: 2026-09-25
- Sahip persona: Product Owner
- İlgili: ADR-14, ADR-20, ADR-22, ADR-23, S5-6c, S5-6c2

## Bağlam
ADR-14 kurucu yoğunlaşma riskini PYŞ başına en fazla %30 ağırlık ve en fazla 3 fon ile sınırlar. S5-6c kağıt B0 sepeti için buna ek olarak “PYŞ başına en fazla 1 B0 fonu” kuralı getirmiştir.
B0 artık sanal nakit değil, kağıt/canlı yolda gerçek fon lotlarından oluşmaktadır. Bu nedenle kurucu riski yalnız risky sepet içinde veya B0 içinde ayrı ayrı ölçmek, aynı PYŞ'nin B0 ve risky pozisyonlar üzerinden toplam maruziyetini gizleyebilir.

## Seçenekler
### A — B0 ve risky için ayrı PYŞ kısıtları
Artı: mevcut kod daha az değişir. Eksi: aynı PYŞ'nin iki alt-sepetteki toplam ekonomik maruziyeti %30'u aşabilir.
### B — PYŞ kısıtını toplam TEFAS portföyünde uygula
Artı: ADR-14'ün yoğunlaşma-risk amacını doğrudan ölçer; B0/risky ayrımından kaynaklanan kural arbitrajını kaldırır. Eksi: B0 ağırlıklandırması risky pozisyonlarla ortak kapasite hesabı gerektirir; bazı durumlarda yatırım yapılamayan bakiye nakitte kalabilir.

## Karar
Seçenek B.
1. PYŞ/kurucu yoğunlaşması B0 gerçek fon pozisyonları ve riskli fon pozisyonları birlikte olmak üzere toplam TEFAS portföyü üzerinde hesaplanır.
2. Her `founder_code` için toplam ağırlık toplam portföyün en fazla %30'udur.
3. Her `founder_code` için pozitif ağırlıklı fon sayısı toplam portföyde en fazla 3'tür.
4. S5-6c'deki “B0 sepetinde PYŞ başına en fazla 1 fon” kuralı kaldırılır. Aynı PYŞ'den birden fazla B0 fonu tutulabilir; birleşik %30 / 3-fon sınırı geçerlidir.
5. Riskli HRP fon başına %25 tavanı, ADR-20 pilot risky toplam %30 tavanı, seçim-kümesi kısıtları ve ADR-22 min-hold kuralı değişmez.
6. Satılamayan veya min-hold altındaki mevcut pozisyonlar PYŞ maruziyetine dahildir. Mevcut ihlal yeni alımla büyütülemez; diğer sözleşmeleri ihlal eden zorunlu satış yapılmaz.
7. Kısıtlar nedeniyle dağıtılamayan bakiye başka bir PYŞ'ye limit aşacak biçimde aktarılmaz; uygulanabilir hedef yoksa serbest nakitte kalabilir.
8. %100 fon yatırımı hedefinde %30 PYŞ tavanı matematiksel olarak en az dört PYŞ gerektirir. Bu, kısıtı gevşetme gerekçesi değildir.

## LEDGER_EXECUTION_SPEC §7 delta
Mevcut riskli-sepet HRP kurallarına aşağıdaki portföy-geneli kural eklenir:
> **Kurucu/PYŞ kısıtı portföy-genelidir.** B0 gerçek fon pozisyonları ile riskli pozisyonlar birlikte değerlendirilir. Her `founder_code` için toplam hedef/gerçekleşebilir ağırlık toplam portföyün en fazla %30'u; pozitif ağırlıklı fon sayısı en fazla 3'tür. B0 için PYŞ başına tek-fon zorunluluğu yoktur. Kilitli/satılamayan mevcut pozisyonlar limite dahildir; limit ihlali yeni alımla kötüleştirilemez. Uygulanabilir alıcı bulunmayan bakiye nakitte kalır.
Riskli HRP için fon %25 tavanı ve seçim-kümesi kısıtları değişmez.

## Sonuçlar / test gereksinimi
- S5-6c B0 founder-dedup kuralı kaldırılır.
- B0+risky ortak founder exposure hesabı tek çekirdekten uygulanır.
- Testler en az şu durumları kapsar: aynı PYŞ'den 2–3 fon kabul; 4. fon reddi; B0+risky birleşik %30 sınırı; locked/min-hold pozisyon; kapasite yoksa bakiye nakit; normal çok-PYŞ senaryosunda regresyon yok.
- Para yolu değişikliği nedeniyle kısa gerçek-veri smoke zorunludur.
- Tam suite/strateji kabulü bu ADR'nin kabulü değildir; VALIDATION kapıları ayrıdır.

## Risk
Kurucu kimliği `founder_code` PIT/meta kalitesine bağlıdır. Eksik veya yanlış kurucu eşlemesi yoğunlaşmayı olduğundan düşük gösterebilir. ADR-23 as-of ve fail-closed kuralları değiştirilmez.
