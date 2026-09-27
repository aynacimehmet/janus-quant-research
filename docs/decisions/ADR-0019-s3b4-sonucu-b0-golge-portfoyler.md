# ADR-0019: Seçim sonucu — alt sınıra göre seçim kanıt değil; S5 = B0 + gölge portföyler; seçim-ayarlı conformal araştırması
- Durum: Kabul (kullanıcı onayı 24.09.2026) · Tarih: 2026-09-24 · Sahip: Product Owner · İlgili: ADR-04, ADR-18
## Bağlam
- Dış testte hiçbir conformal seçim varyantı B0 nakit sepetini geçemedi; 12 başlangıç medyanları negatif kaldı. ADR-18 ölçütü (dış test ≥ +1 pp ve medyan ≥ 0) geçilmedi.
- Seçilen-fon kapsaması hedefin altında kaldı: fon bazında geçerli aralık, evrendeki en yüksek alt sınırı seçince geçerliliğini yitiriyor (kazananın laneti).
- Dış testte aday sayısı arttı ve ACI geçmiş kapsamaya bakıp aralıkları daralttı; rejim kırılmasında yanlış pozitif çoğaldı. Kill-switch tetiklendi ancak model ve baseline birlikte bozulduğu için fark görünmedi. `tax_penalty` sıralamayı değiştirmedi.
- Seçim dönemi ile dış test arasındaki fark VALIDATION §1 ayrımının gerekliliğini gösterdi.
## Karar
1. "Conformal alt sınırı > 0 → kanıt" (ADR-04a, ADR-18) **seçim adımıyla birlikte geçersizdir**; alt sınıra göre sıralama üretim seçim mekanizması olamaz.
2. **S5 kağıt-ticaret B0 nakit sepetiyle başlar** (operasyon testi: emir listesi, `proposal_id`/`fill` mutabakatı, Telegram, gece/sabah koşuları). B2b, B2c-m ve S3b-5'te üretilecek B2c-fdr **gölge portföy** olarak her gün hesaplanır, yüzde bazında raporlanır, işlem yapılmaz — ileriye dönük kanıt sermayesiz birikir.
3. Araştırma ajandası (S3b-5): (a) seçilen-fon kapsama açığının kök neden ayrıştırımı (aralık darlığı vs seçim etkisi; α sabit karşı-olgusal); (b) **seçim-ayarlı conformal**: conformal p-değerleri + Benjamini–Hochberg (FDR kontrolü; Jin & Candès 2023 "conformal selection") → B2c-fdr; (c) kill-switch referansı nakit (seçilen fonların olgunlaşmış 63g nakit-fazlası < 0 → düş); (d) α tavanı hedef + 0,05; (e) λ = 0 varyantı. γ/α geriye dönük ayarlanmaz.
4. Üretim adayı ancak B2c-fdr (veya sonrası) dış testte ADR-18 ölçütünü geçerse değişir; aksi hâlde B0 kalır. S4 (DRQN) araştırma olarak sürer; beklenti "nakde yakın kal".
## Sonuçlar
S5 spesifikasyonu B0 + gölge portföyler olarak belirlendi; seçim mekanizması araştırma aşamasında, üretim tabanı B0.
