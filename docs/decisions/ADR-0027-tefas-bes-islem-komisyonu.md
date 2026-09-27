# ADR-0027: TEFAS/BES işlem komisyonu semantiği
- Durum: Kabul (kullanıcı, 26.09.2026)
- Tarih: 2026-09-26
- Sahip persona: Product Owner
- İlgili: ADR-23, LEDGER_EXECUTION_SPEC, DATA_DICTIONARY, S5-6c2
## Bağlam
Mevcut TEFAS yolunda `entry_fee`/`exit_fee` eksikse `fee_assumed_zero` ve “A3-komisyon” olarak sıfıra çekilmektedir. Doğrulanan iş kuralı: TEFAS ve BES fonlarında alım/satım işlem komisyonu yoktur; ABD ayağında mevcut 1,5 USD/emir maliyeti vardır. TEFAS/BES için sıfır komisyon eksik-veri varsayımı değil deterministik ayak kuralıdır.
## Karar
1. TEFAS işlem komisyonu = 0; BES işlem komisyonu = 0.
2. TEFAS/BES `entry_fee`/`exit_fee` yürütme maliyeti değildir; gerekirse ham kaynak alanı olarak korunur.
3. TEFAS/BES için `fee_assumed_zero` / “A3-komisyon” uyarısı üretilmez.
4. ABD 1,5 USD/emir kuralı değişmez.
5. TEFAS/BES sürtünmesinde işlem komisyonu yoktur; vergi, valör/settlement ve diğer geçerli etkiler korunur.
6. ADR-27, ADR-23'ün yalnız komisyon/eksik-fee hükmünü supersede eder; diğer PIT/status/valör/fail-closed kararları değişmez.
7. ADR-27 öncesi gerçek-veri sonuçları, efektif TEFAS fee değerlerinin tamamının sıfır olduğu kanıtlanmadıkça güncel kabul kanıtı değildir.
## Test / doğrulama
- TEFAS/BES eksik veya non-zero kaynak fee → execution fee yine 0.
- TEFAS/BES paper/backtest fee maliyeti üretmez.
- ABD 1,5 USD/emir regresyon testi korunur.
- TEFAS/BES `fee_assumed_zero` warning/provenance kalkar.
- Düzeltme sonrası kısa gerçek-veri smoke yeniden çalıştırılır.
