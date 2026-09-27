# ADR-0016 (taslak): Sapma eşikleri ve vergi erteleme heuristiği
- Durum: önerildi · Tarih: 2026-09-23 · Sahip: Quant Scientist
## Bağlam
Sistem sahibi kararı: sapma %3, vergi doğuran satışta %6. Uygulamada üç ek heuristik belgelendi: (1) "vergi doğuran" = satılacak lotların değer bazında ≥ %50'si kârda; (2) yeni pozisyon eşikten muaf; (3) kapı/DD kaynaklı azaltma eşikten muaf.
## Seçenekler
| Seçenek | Artı | Eksi |
|---|---|---|
| A. Üç heuristik olduğu gibi (belgelenmiş) | Turnover düşük; risk azaltma gecikmez | %50 eşiği keyfi |
| B. Lot bazında: yalnızca kârlı lotlar için %6, zararlı lotlar %3 | Daha hassas | Uygulama karmaşık; kısmi satış mantığı |
| C. Tek eşik %3, vergi dikkate alınmaz | Basit | Vergi sürtünmesi artar |
## Karar (öneri)
A, ADR olarak kabul edilir; B S5 sonrası deney. Test: eşiğin iki yanındaki örnek işlemler ve kapı kaynaklı zorunlu azaltma.
