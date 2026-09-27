# ADR-0018: Nakit tabanı + kanıtla risk — B3 kağıt-ticaret adayı geri çekildi; kapı çalışması S4'e
- Durum: Kabul · Tarih: 2026-09-23 · Sahip: Product Owner · Değiştirir: ADR-15 (kağıt-ticaret adayı ve S3 önceliğinin içeriği)
## Bağlam
Nakit vekili defter içi para piyasası sepeti oldu (bozumda vergi, tarihe göre oran). Yeniden koşularda hiçbir kural tabanlı maruziyet kapısı nakit sepetini anlamlı biçimde geçemedi; parametre taramasında aşırı uyum riski yüksek çıktı. Beş yıllık dönemde tek rejim geçişi olduğundan kapı modelleri istatistiksel olarak ayırt edilemiyor.
## Karar
1. Üretim çerçevesi: **varsayılan pozisyon nakit sepeti (B0)**; riskli varlık yalnızca **kanıt** varsa: conformal alt sınırı > 0 olan fonlar (ADR-04a, S3b-3/4). Kanıt yoksa portföy nakitte kalır.
2. ADR-15'teki "kağıt-ticaret adayı B3" geri çekildi; S5 kağıt-ticaret adayı S3b select-suite sonucuna göre belirlenir (B0 dahil).
3. HRP yumuşatma parametreleri (λ 0,5, tazeleme 3) tasarım varsayılanıdır; veriden optimize edilmez.
4. Maruziyet kapısı (R0/R1/R2) üretimde kullanılmaz; S4'te DRQN girdisi/ablasyonu olarak kalır. Kullanıcının DD > %12 → orta (0,65) kuralı korunur; DD hedefi kapı seviyelerinden ayrı konfig anahtarı (`risk.dd_exposure`) olur — ikili deneyin karıştırdığı nokta.
5. Kabul ölçütü VALIDATION_PROTOCOL'e göre: dış testte B0 sepetine karşı `excess_cagr_pp ≥ +1`, 12 başlangıçta medyan ≥ 0.
## Sonuçlar
Değerlendirme suite'ine "B0 sepet" satırı ve "nakitte kalma oranı" kolonu eklendi.
