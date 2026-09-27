# TEMPORAL_PROTOCOL.md — Janus zaman sözleşmesi (tüm katmanlar için tek kaynak)

Sürüm 1.0 · 23.09.2026 · Sahip: Quant Scientist · Değişiklik ADR gerektirir.

## 1. Takvim ve indeks
- **İşlem günü indeksi `i`**: TEFAS NAV panelinin tarih ekseni (BIST iş günleri). Tüm gecikmeler *işlem günü* cinsindendir; takvim günü kullanılmaz.
- Zaman dilimi Europe/Istanbul. ABD ayağı (Faz 2) kendi piyasa takvimini ve DST'yi ayrı tutar.

## 2. TEFAS NAV'ın yaşam döngüsü (varsayım A1 — kullanıcı teyidi 22.09.2026)
| Olay | Zaman | Alan |
|---|---|---|
| Gün `i` kapanış NAV'ı hesaplanır | `i` akşamı | `observed_date = cal[i]` |
| NAV ilan edilir | `i+1` sabahı (~09:00) | `source_available_at = cal[i+1] 09:00` (tarihsel yeniden kurmada **varsayılan**; canlı kayıtta gerçek zaman) |
| Depoya yazılır | koşu zamanı | `ingested_at`, `run_id` |
Kural: karar anı `D` için görülebilir NAV'lar `source_available_at ≤ D 09:15` olanlardır → pratikte `observed_date ≤ D−1`. Tarihsel testte `ingested_at` **filtre olarak kullanılmaz** (2026'da yüklenen 2022 verisi 2022 testinde görünmelidir); yalnızca revizyon izi için tutulur.


> **A1 revizyonu (23.09.2026, gözlem):** TEFAS fiyat etiketi **ilan günü**dür: 22.09 kapanışından hesaplanan fiyat panelde 23.09 etiketiyle ve 23.09 sabahı görünür. Dolayısıyla karar günü D, etiket D'yi görür; sabah emri etiket **D+1**'de dolar. Backtest motorundaki "sinyal ≤ D−1, dolum D" kuralı etiket eksenine göre bir gün kayar; sinyal–dolum gecikmesi (1 etiket-günü) ve P&L aynı kalır. Canlı akış: sabah tazeleme, günün etiketi yayımlandıktan sonra (`calendar.nav_publish_time`) koşmalı; yayın saati ilk gerçek koşularla kalibre edilecek; §2 tablosu bu revizyona göre güncellenecek.

## 3. Karar–emir–dolum zinciri (TEFAS)
| Adım | Zaman | Fiyat / durum |
|---|---|---|
| Gece koşusu (ağır) | `D−1` 23:30 | NAV ≤ `D−2` görür (NAV[D−1] henüz ilan edilmedi) |
| Sabah tazeleme + emir listesi | `D` 09:15 | NAV ≤ `D−1` görür (`feature_asof = D−1`) |
| Emir | `D` 09:30–12:00 | `decision_at = D` |
| Dolum | NAV[`D`] (ilan `D+1`) | `fill_price_date = D` |
| Pay satılabilir | `D + buy_valor` | lot.available_idx |
| Satış nakdi | satış günü + `sell_valor` | alacak |
Yatırımcının **ilk kazanabileceği getiri** NAV[D] → NAV[D+1]'dir; NAV[D−1] → NAV[D] getirisi ona ait değildir.

## 4. Denetimli tahmin hedefi (H13)
Özellik satırı `t` (NAV ≤ t ile hesaplanır) → karar `D = t+1` → dolum NAV[t+1] → 21 işlem günü elde tutma:
`y_t = log(NAV[t+22] / NAV[t+1]) − Σ_{h=t+2..t+22} log(1 + cash_h)`

- Nakit terimi **log uzayında** birikimdir: NAV terimiyle aynı uzayda toplanır; `cash_h` = nakit vekili sepetin (P02, `cash_proxy_returns`) h. gün brüt basit getirisi, tek kaynak.
- `label_end = t+22`, `label_available_at = cal[t+23] 09:00` (NAV[t+22] ilanı).
- Eğitim seti karar günü `D` için: `label_available_at ≤ D` ⇔ `t ≤ D − 23`. Bu, "purge"ün tanımıdır; ayrıca takvim günü hesabı yapılmaz.
- Çıkarım: her `t = D−1` satırı için X vardır, y yoktur (`label_ready=False`); **X ve y maskeleri ayrıdır** (H03).

## 5. Yeniden eğitim ve tahmin sürümü (H04)
- Yeniden eğitim ayın ilk işlem günü `R`: model `M_R`, eğitim satırları `t ≤ R − 23`. `R` ile bir sonraki `R'` arasındaki karar günlerinde `M_R` kullanılır; her tahmin kaydı `model_id = R` taşır.
- Eğitim penceresi anchored (başlangıçtan). Tarihsel yeniden kurmada model tarihleri gerçek takvimle aynı.

## 6. Conformal kalibrasyon ve ACI (H05, H06, H07)
- Kalibrasyon seti karar günü `D` için: OOS tahmin satırları (`t ≥ D − calib_window − 23`, `label_available_at ≤ D`). Skor `s = max(q_lo − y, y − q_hi)`.
- Kuantil: sıralı istatistik `s_(k)`, `k = ceil((n+1)(1−α_D))`; `k > n` → sonsuz aralık (tahmin verilmez, fon aday olmaz); `n < n_min` (=200 fon-gün) → tahmin verilmez.
- **Günde bir α güncellemesi:** `D` günü olgunlaşan satırlar (`t = D − 23`) üzerinden `err_D = ağırlıklı ortalama 1[y ∉ aralık]` (ağırlık: eşit; alternatif: yalnızca o gün seçilmiş fonlar — ayrı deney). `α_{D+1} = clip(α_D + γ(miscoverage_target − err_D), α_min, α_max)`. Fon sırasından bağımsızdır; bu panel uyarlaması özgün ACI garantisini **iddia etmez**, "ACI-tarzı günlük global α" olarak adlandırılır.
- `miscoverage_target` (0,2 seçim / 0,1 risk bandı) ile `coverage_target` (0,8 / 0,9) ayrı isimlerdir; API `miscoverage_target` alır.
- Quantile crossing: `q_lo > q_hi` ise ikisi sıralanır ve satır `quality_flag=crossing` ile işaretlenir.

## 7. Kill-switch (H05)
Karar günü `D`'de son 126 işlem günü içinde `label_available_at ≤ D` olan OOS satırlarda pinball loss (model) ≥ pinball loss (aynı satırlarda kuantil baseline) ise seçim skoru momentum'a düşer. Baseline'lar aynı satır kümesinde, aynı ufuk ve α ile hesaplanır (H09).

## 8. Makro ve fon meta verisi (D03, H02)
- Makro: `available_from` = gerçek yayın zamanı biliniyorsa o; bilinmiyorsa **varsayım** (TÜFE: dönem ayını izleyen ayın 3'ü 10:00 → tarihsel yeniden kurmada `date + 35 gün` kaba yaklaşımı; `assumption_id = A2`). Revize değerler ayrı `revision_id` ile; ilk yayın değeri yoksa mevcut değer kullanılır ve varsayım etiketlenir.
- Fon meta verisi: `fund_master` snapshot'ları `snapshot_date` taşır. **Tarihçesi olmayan alanlar** (`aum_now`, `investor_count`, `tefas_status`, valör, komisyon) tarihsel özellik olarak **kullanılmaz**. Dar yürütme istisnası ADR-0023'tür: yalnızca alış/satış valörü ve TEFAS status metni 2026-09-22 ilk PIT snapshot öncesindeki yürütme simülasyonunda mevcut profilden A3 varsayımıyla geriye taşınabilir; status “işlem görüyor” varsayılır. ADR-27 gereği TEFAS/BES komisyonu deterministik sıfırdır; kaynak fee eksikliği varsayım/provenance değildir. Sonuç “A3 varsayımlı, PIT kanıtı değil” diye işaretlenir. Bu alanlar `FEATURE_COLUMNS`, tarihsel X, model eğitimi/çıkarımı veya kanıt iddiasına **giremez**. Bugünkü PYŞ kara listesi ayrı “bugünkü politika” etiketidir.
- 2026-09-22 ve sonrasında karar günü D 09:15 itibarıyla en son erişilebilir fon snapshot'ı, `snapshot_date ≤ D` ve `source_published_at ≤ D 09:15` olanlar arasından seçilir. D 09:15 sonrasında yayımlanan snapshot as-of dışıdır; önceki erişilebilir snapshot en yenisi olarak kalır. Seçimden sonra `last_success_at ≤ D 09:15`, profil yaşı ≤ 7 takvim günü ve gerekli alanlar (alış/satış valörü, tanınan status metni, `tax_category`) doğrulanır. İşlem yönü metinden türetilir; saklanan `can_buy`/`can_sell` bayrakları uyuşmazlık uyarısıdır. Bilinmeyen status iki yönü kapatır; tek yönlü kapalı status yalnız o yönü kapatır. ADR-27 uyarınca TEFAS/BES execution fee deterministik 0'dır; ham kaynak fee eksikliği/geçersizliği block veya provenance oluşturmaz. Seçilen snapshot/profil eksik veya eskiyse fon bazında işlem yapılmaz, gerekçe kaydedilir ve değerleme son mevcut NAV ile sürer; daha eski snapshot'a fallback yapılmaz. Yayın timestamp'i yoksa date-only varsayımı açıkça etiketlenir, PIT kanıtı değildir. Gelecek snapshot/meta kullanılamaz. Eski/yeni B0 karşılaştırmasında satış valörü +1 stres satırı zorunludur (ADR-0023; VALIDATION_PROTOCOL §4).
- Yarı-sabit alanlar (`umbrella_type`, `category`, `tax_category`, `founder`) tarihsel deneyde "sabit varsayılır" (`assumption_id = A3`) ve bu varsayım raporlanır; bu kural yürütme istisnasını genişletmez.

## 9. Sızıntı testleri (H08)
1. **Beyaz liste:** özellik üretici yalnızca `FEATURE_COLUMNS` listesini çıkarır; `y`, `label_*`, `NAV[t+k]` türevleri listeye giremez; ihlal → hata.
2. **Gelecek perturbasyonu:** `t > t0` için NAV/makro/**tarihçesiz (PIT) meta** değerleri değiştirildiğinde `t ≤ t0` satırlarının X'i, tahminleri, α'sı, fallback bayrağı ve emir listesi bit düzeyinde aynı kalır. Not: A3 yarı-sabit alanlar (`umbrella_type`, `category`, `tax_category`, `founder` vb.) tarihsel deneyde sabit varsayılır; geçmiş A3 değerlerinin yeniden yazılması beklenmez (meta perturbasyonu = tarihçesiz alanlar).
3. **Model sürümü:** her tahmin kaydı `model_id` ve `train_max_t` taşır; `train_max_t ≤ D − 23` doğrulanır.
4. Negatif kontrol (isteğe bağlı): geleceği sızdıran kolon eklenince loss düşer — bu sızıntı *korumasının* kanıtı değildir.

## 10. Rejim modeli (V04)
E�itim penceresi ≤ `D−1` üzerinde durum dizisi çözülür (DP); yalnızca `D−1` durumu çevrimiçi kuralla çıkarılır ve `regime_signal[D]` olarak kaydedilir. Sonraki yeniden tahminler geçmiş kayıtlı sinyalleri **değiştirmez** (kayıt tablosu append-only). Model adı: "hard-label jump model (sparse değil)".
