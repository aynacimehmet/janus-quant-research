# LEDGER_EXECUTION_SPEC.md — Defter, icra, vergi ve performans sözleşmesi

Sürüm 1.0 · 23.09.2026 · Sahip: Quant Scientist · Kod: `src/janus/backtest/{ledger,engine,costs}.py`

## 1. Varlıklar ve durumlar
- Nakit (`cash`), alacaklar (`receivables[settle_idx]`), pozisyonlar (`units[i]`), lotlar (FIFO: `units, cost_per_unit, bought_idx, tax_rate, available_idx`).
- Fon durumu üç ayrı bayrak (D05, P05):
  - `can_buy[D]`: `tefas_status` "işlem görüyor" ve alım kapalı değil; askıda değil.
  - `can_sell[D]`: bozum kapalı değil; pay `available_idx ≤ D`. "Alıma kapalı, bozuma açık" fon **satılabilir**.
  - `data_stale[D]`: NAV son `max_stale_days` işlem gününde yok. Yalnızca **veri** durumudur: alım/satım yapılmaz, değerleme son bilinen NAV ile, **haircut uygulanmaz**, sabah mesajında uyarı.
  - `suspended[D]`: resmî askı = durum metni "alımına kapalı" **ve** "bozumuna kapalı" (SPK tasfiye seti) **veya** senaryo/kullanıcı maskesi. "TEFAS'ta işlem görmüyor" askı **değildir** (platformda satılmıyor; `can_buy=can_sell=False`, zaten evren dışı) — alım/satım yok, değerleme `NAV_last × (1 − haircut)` yalnızca **stres senaryosunda ve raporlamada "tasfiye değeri"** olarak; risk kapısı/DD için `official_value` (son resmî NAV) kullanılır, `stress_value` ayrı sütunda raporlanır.

## 2. Fiyatlama ve zamanlama (TEMPORAL_PROTOCOL §3)
TEFAS/BES alımı `NAV[D]` ile ve satışı `NAV[D]` ile yürütülür; işlem komisyonu deterministik 0'dır (ADR-27). `entry_fee`/`exit_fee` ham kaynak alanları execution maliyeti değildir. Satılabilirlik `D + buy_valor`, satış nakdi `D + sell_valor`. Valör alanları `execution.valor_mapping` ile eşlenir; A4 **doğrulandı** (kullanıcı, TEFAS fon sayfası, 24.09.2026): eşleme yatırımcı perspektifiyle uyumlu; kabul engeli kalktı. ABD'de 1,5 USD/emir maliyeti değişmez.

## 3. Vergi (ADR-01, P04)
- Stopaj satışta, lot bazında: `tax = rate(lot) × max(gerçekleşen kâr, 0)`; zarar mahsup edilmez (varsayım A5, doğrulanacak).
- `rate(lot)` = `withholding_rate(tax_category, purchase_date)` (borsapy tarih × kategori tablosu); kategori bilinmiyorsa %17,5 ve `tax_unknown=True`; tarihsel lotlarda kategori sabit varsayımı (A3) raporlanır.
- Vergi **yalnızca bir kez** düşülür: satış nakdinden. Değer `V` tahakkuk vergi içermez. Raporda ayrıca `liquidation_value = V − (tüm pozisyonlar bugün satılsa ödenecek vergi)` verilir.
- Nakit vekili (aşağıda) vergisini bozumda öder; günlük vergi yaklaşımı yalnızca hassasiyet deneyidir.

## 4. Performans tanımı (P01)
- Backtest'te dış nakit akışı yoktur; `V_t = cash + receivables + Σ units × NAV_last`. Getiri `log(V_t/V_{t−1})`.
- Canlıda katkı/çekim olursa zaman ağırlıklı getiri (TWR) ve dış akıştan arındırılmış `V`; para yatırılan gün getiri 0.
- Mutabakat denklemi her gün doğrulanır: `V_t = V_{t−1} + PnL_t − vergi_t − komisyon_t (+ dış akış_t)`. TEFAS/BES icrasında `komisyon_t = 0`; vergi ve settlement/alacak zamanlaması aynen uygulanır.

## 5. Nakit vekili — B0 (P02)
- **Backtest istisnası (PO onaylı, S5-6c):** Backtest B0, tarih boyunca kesintisiz geçmişli PPF'lerden eşit ağırlıklı **endeks vekilidir**; kağıt defterde B0, PIT profil/valör ve erişilebilir NAV koşullarını sağlayan gerçek fon lotlarıdır. Bu iki yol aynı defter değildir; eşit getiri/muhasebe iddiası kurulmaz, endeks vekili ile icra edilebilir sepet farkı ayrı raporlanır. Büyüklüğe göre seçim yapılmaz.
- Kağıt pilot INIT yalnız `cash=capital` açar; pozisyon/lot oluşturmaz. İlk karar D'de BH kümesi boşsa B0 hedefi, kod sıralı gerçek fon kodlarına eşit BUY hedefi üretir; emir ancak D+1 NAV ile dolar.
- Öneri bütçesi karar D'de yalnız serbest nakit ve D itibarıyla vadesi gelmiş alacaklardır. Önerilen satışlardan doğacak gelir bütçe değildir; risky BUY bütçesi yetmiyorsa o alım üretilmez, B0 BUY yalnız kalan kullanılabilir nakit kadardır. D+1 fill aynı gün yeni finansman/emir üretmez; yeni proposal sonraki karar gününde ve insan onayı 12:00'ye kadar geçerlidir.
- B0 üye adedi 3–5 hedef; 1–2 aday uyarıyla kabul; 0 aday fail-closed hold'dur. NAV ve metadata ayrı PIT sınırlarında (her ikisi `≤ D`) raporlanır; D−1 NAV + D meta geçerlidir. Stale NAV son erişilebilir değerle işaretlenir; stale fonla işlem yapılmaz.
- Backtest'teki ilk B0 sepeti gün 1'de endeks vekili başlangıcıdır; kağıt INIT'te ise pozisyon yoktur. Kanıtla alınmış risky lot 21 iş günü tutulur; DD > %12 istisnası uygulanır. B0 lotları min-hold'dan muaftır.
- Backtest strateji nakdi (risky dışı payı) da aynı endeks-sepet vekiliyle değerlenir. Kağıt defterde nakit NAV/getirisi uydurulmaz: pozisyon alımı yalnız gerçek fill sonrası oluşur, nakit serbest nakit olarak kalır; eşit muhasebe/eşit getiri iddia edilmez.
- Politika faizi serisi yalnızca `assumption_id = A6` etiketli yedek/hassasiyet serisidir.

## 6. Rebalance, sapma eşikleri ve kapı (P06, P08)
- Planlı: ayın ilk işlem günü (çeyreklik varyant: çeyrek ilk işlem günü). Tetikleyici: DD > `dd_trigger` (0,12) → `exposure ≤ medium`; DD < `dd_release` (0,06) **ve** planlı rebalance → serbest. "Her durumda en az orta" ifadesi geçersizdir; kural budur.
- Sapma eşiği: `|hedef − mevcut| < 0,03` → işlem yok; satışta, satılacak lotların **değer bazında** ≥ %50'si kârdaysa eşik 0,06 (vergi erteleme). Yeni pozisyon (mevcut 0) eşikten muaf. Kapı/DD kaynaklı azaltma eşikten **muaf** (risk azaltma öncelikli). Bu heuristik ADR-16 taslağı olarak sunulur.
- Kilitli pozisyon (askıda/satılamaz) hedefi aşarsa gün raporda açıklanır; hedef gerçekleştirilebilir kısma normalize edilir.

## 7. HRP ve kısıtlar (P07)
Ağırlık paydası = risky sepet (nakit vekili hariç). Tavanlar: fon %25, kurucu %30 ve ≤ 3 fon, küme ≤ 3 fon. Algoritma: tavana çarpan fon/grup dondurulur, fazla dondurulmamışlara oransal dağıtılır, alıcı kalmazsa kalan nakit vekiline gider. Test kümesi: az aday, tek kurucu, kesişen küme/kurucu, satılamayan varlık.

Kurucu/PYŞ kısıtı portföy-genelidir. B0 gerçek fon pozisyonları ile riskli pozisyonlar birlikte değerlendirilir. Her founder_code için toplam hedef/gerçekleşebilir ağırlık toplam portföyün en fazla %30'u; pozitif ağırlıklı fon sayısı en fazla 3'tür. B0 için PYŞ başına tek-fon zorunluluğu yoktur. Kilitli/satılamayan mevcut pozisyonlar limite dahildir; limit ihlali yeni alımla kötüleştirilemez. Uygulanabilir alıcı bulunmayan bakiye nakitte kalır.
Riskli HRP için fon %25 tavanı ve seçim-kümesi kısıtları değişmez.

## 8. Öneri ve gerçekleşme mutabakatı (P09, S5)
Her sabah emir listesi `proposal_id` (tarih+sıra), geçerlilik `D 12:00`. Kullanıcı gerçekleşmeleri (kod, yön, tutar/pay, tarih) `janus fill …` ile bildirir; bildirilmeyen öneri **süresi dolmuş** sayılır ve defterde görünmez. Aynı `proposal_id` iki kez işlenemez. Hedef portföy ile gerçek portföy farkı her sabah raporlanır.
