# ADR-0020: B2c-fdr (conformal selection, BH q=0,20) üretim adayı; S5 kağıt-ticaret kuralı; kill-switch v2 red
- Durum: Kabul (kullanıcı onayı 24.09.2026) · Sahip: Product Owner · İlgili: ADR-04, ADR-18, ADR-19
## Bağlam
- B2c-fdr (Jin & Candès conformal selection, H0: y ≤ 0, BH q = 0,20, clipped, n_min 200, calib 126): tam dönemde nakit sepetine göre küçük pozitif fazla getiri, düşük MDD ve yüksek nakitte kalma oranı; 12 başlangıçta medyan pozitif ve tüm başlangıçlar ≥ 0; dış testte hiç seçim yapmadı (B0 ile eşdeğer). Seçilen kümedeki gerçekleşen y > 0 oranı yüksek, ancak satır sayısı istatistiksel iddia için yetersiz.
- q ızgarası: q = 0,10 hiç seçmiyor (= B0); q = 0,30 gevşek ve daha yüksek riskli; q = 0,20 önceden kaydedilmiş varsayılan.
- Kill-switch v2 (nakit referanslı) ters tepti: reddedildi; v1 gölgelerde.
## Karar
1. B2c-fdr **üretim adayı**. ADR-18 ölçütünün "dış test ≥ +1 pp" kısmı seçim olmadığı için değerlendirilemedi; "12 başlangıçta medyan ≥ 0 ve hiçbir başlangıçta negatif değil" sağlandı.
2. **S5 kağıt-ticaret:** canlı emir listesi = B2c-fdr çıktısı (çoğu gün B0 sepeti; seçim yaptığında öneri). Sabah mesajında "kanıt: yok / var (N fon)". İlk sinyaller kağıt üzerinde izlenir; en az 3 seçim bölümünde olgunlaşmış nakit-fazlası ≥ 2/3 pozitif → gerçek sermaye, riskli pay tavanı %30 (ADR-21'e konu). Gölge portföyler: B2b, B2c-m, B2c-fdr q10/q30.
3. Kill-switch v2 red; α tavanı ve λ = 0 iptal (ADR-19).
4. Varsayımlar korunur: A4 valör yönü, A5 zarar mahsupu açık kabul engeli; aynı gün fonlar bağımlı — BH'nin PRDS altında geçerliliği iddia edilmez, ileriye dönük seçim kalitesi (y > 0 oranı) izlenir.
## Sonuçlar
S4 araştırması (uyarlanabilir maruziyet) sürüyor; gelecekteki bir maruziyet ajanının durum vektörüne "kanıt var/yok" ve seçilen küme boyutu eklenebilir.
