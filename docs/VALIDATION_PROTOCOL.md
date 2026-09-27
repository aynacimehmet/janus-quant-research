# VALIDATION_PROTOCOL.md — "Başarılı" neye dayanır?

Sürüm 1.0 · 23.09.2026 · Sahip: QA / Validation

## 1. Katmanlar ve amaçlar
| Katman | Soru | Yöntem | Ana metrik |
|---|---|---|---|
| Tahmin (GBDT quantile) | Tahmin naive'den iyi mi? | Rolling-origin WF (aylık refit, purge = etiket olgunlaşması), **CPCV** (6 blok / 2 test) yalnızca hiperparametre seçimi için | Pinball loss (3 α) vs (0-tahmin, 63g ortalama→21g, tarihsel kuantil) aynı satırlarda; rank IC |
| Kalibrasyon (CQR+ACI) | Aralıklar güvenilir mi? | Tarih bloğu bazında kapsama (aylık), **seçilen fonlarda** kapsama ayrı; aralık genişliği; aday sayısı; nakitte kalma oranı | Kapsama − hedef (pp), blok bazında |
| Strateji | Nakdi vergi sonrası geçiyor mu? | Anchored WF; **seçim dönemi** (2022-10 → 2025-06) ve **dış test** (2025-07 → 2026-09) ayrı; 12 başlangıç = duyarlılık (bağımsızlık değil) | Önceden sabit: `excess_cagr_pp` (vergi sonrası, **B0 sepet defteri** karşısında — günlük vergi yaklaşımı değil), ikincil MDD ve **nakitte kalma oranı**; kabul marjı ≥ +1 pp dış testte ve 12 başlangıçta medyan ≥ 0 (ADR-18) |
| Tarama | Seçilen konfig şans mı? | CSCV/PBO (8 blok / 70 kombinasyon) strateji taramaları için; DSR (deneme sayısı = tarama boyutu) | PBO ≤ 0,5 ve DSR raporda |
| Stokastik bileşen (DRQN, HMM init) | Tohuma bağlı mı? | 5 tohum yalnızca stokastik modüllerde; deterministik stratejide tekrarlanabilirlik testi | Tohumlar arası std |
| Pilot | Operasyon çalışıyor mu? | 4 hafta kağıt-ticaret: koşu başarı oranı, öneri→gerçekleşme mutabakatı, fiyat/valör sapması | Operasyon KPI'ları — **alfa kanıtı değildir** |
| Canlı sermaye | Ne zaman gerçek para? | Pilot ✓ **ve** dış test kabul marjı ✓ **ve** valör/vergi varsayımları doğrulanmış | Ayrı karar (ADR) |

## 2. Deney kaydı
Her sonuç satırı: `run_id`, git SHA, konfig hash, veri as-of (parquet zamanı), model_id'ler, seçim/dış test dönemi, karşılaştırma serisi (B0 defteri), metrik tanımı ve birim (**yüzde puan = pp**). MLflow + `docs/reports/` özeti (veri içermez).

## 3. Birimler
- Getiri farkları **pp** (CAGR farkı); oranlar `%`; kapsama `pp` sapma. "Fazla getiri" daima "B0 defteri karşısında, vergi sonrası".

## 4. Bilinen sınırlar (raporlarda tekrar edilir)
- Evren bugünün sağ kalanları ve bugünkü kara liste (V05): "bugünkü politika" ve "o günkü bilgi" koşuları ayrı raporlanır; ikincisi kara listesizdir.
- Beş yılda tek rejim geçişi: kapı modelleri arasında istatistiksel ayrım beklenmez; 23.09 yeniden koşusunda kademeli ve ikili kapılar nakdi geçmedi → kapı üretimde yok (ADR-18).
- Günlük vergili nakit vekili kullanan erken sonuçlar iyimserdi; eski tablolar yalnızca tarihçe olarak tutulur, karşılaştırmada kullanılmaz.
- Fon-gün satır sayısı bağımsız örneklem gücü değildir (aynı gün fonlar ve örtüşen ufuklar bağımlı).
- S5-6c PO kararı ve fix2 (ADR-0023; 2026-09-25): ilk PIT öncesi alış/satış valörü ve status metni A3 yürütme varsayımıyla taşınabilir; status “işlem görüyor” kabul edilir ve dönem “A3 varsayımlı, PIT kanıtı değil” diye etiketlenir. Bu alanlar tarihsel X/FEATURE_COLUMNS/model girdisi değildir; bugünkü PYŞ kara listesi ayrı politika etiketidir. 2026-09-22 ve sonrasında en yeni erişilebilir profil seçilir; `last_success_at ≤ D 09:15`, yaş ≤7 takvim günü, alış/satış valörü, tanınan `tefas_status` ve `tax_category` gereklidir. İşlem yönleri status metninden türetilir; saklanan `can_buy`/`can_sell` uyuşmazlığı uyarı/audit alanıdır. Bilinmeyen status iki yönü, tek yönlü kapalı status ilgili yönü kapatır. Komisyon hükmü ADR-27 ile supersede edilmiştir: TEFAS/BES execution fee deterministik 0; kaynak fee eksik/non-zero/geçersizliği block, `fee_assumed_zero` veya A3-komisyon warning/provenance oluşturmaz. Eksik/eski/geç yayımlanmış profil için eski snapshot'a fallback yok; neden kaydı ve son NAV değerlemesi korunur. Date-only yayın varsayımı PIT kanıtı değildir; gelecek meta kullanılmaz.
- Eski/yeni B0 karşılaştırmasında ayrıca “satış valörü +1” stres satırı raporlanması zorunludur (ADR-0023); bu satır B0 stresidir, başka strateji için üretilen valör stresi onun yerine geçmez.
