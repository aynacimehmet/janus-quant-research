# ADR-0022: Kanıt ufku tutma kuralı
- Durum: Kabul · Tarih: 2026-09-25 · Sahip: Product Owner · İlgili: ADR-20, S5-R (F12), S5-6a
## Karar
Conformal selection (BH) kanıtıyla alınan riskli fon, ilk dolum tarihinden itibaren en az 21 iş günü (işlem takvimi; config paper.min_hold_days) tutulur — alım kararı 21 günlük hedef ufku için verilmiştir. Süre dolmadan kanıt kaybı → "tut". Süre dolunca BH kümesinde değilse satılır, hedef nakit sepeti. İstisnalar: DD kuralı (DD > %12 → maruziyet ≤ risk.dd_exposure), can_sell=False/askı. Kural B0/sepet fonlarına uygulanmaz. Mesajdaki kanıt sayısı = BH kümesi boyutu.
