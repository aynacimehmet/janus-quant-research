# DATA_DICTIONARY.md — tablo, alan, birim, anahtar ve zaman sözleşmesi (sürüm 1.1, 23.09.2026)

Zaman alanlarının anlamı `docs/TEMPORAL_PROTOCOL.md` §2'de tanımlıdır. Birimler: oranlar **ondalık** (0,175), yüzdeler yalnızca raporda; getiriler **log** (aksi belirtilmedikçe); günler **işlem günü**.

## fund_master (PIT snapshot; anahtar: `snapshot_date, fund_code`)
| Kolon | Tip | Tanım / kaynak | Not |
|---|---|---|---|
| snapshot_date | DATE | Snapshot günü | Aynı gün yeniden koşu üzerine yazar; `run_id` ile izlenir (D02, P2) |
| run_id | VARCHAR | Yazan koşu | S3b-0c'de eklenir |
| fund_code, isin, name | VARCHAR | TEFAS kodu, ISIN, unvan | |
| fund_class | VARCHAR | YAT / EMK | |
| umbrella_type, category | VARCHAR | Şemsiye türü, kategori | Tarihsel deneyde sabit varsayılır (A3) |
| founder_code | VARCHAR | Kurucu kodu (`kurucuKod`) | **Kimlik** budur; `founder` addan türetilmiş, alias tablosu S3b-4 |
| founder, manager | VARCHAR | Kurucu adı (türetilmiş), yönetici (S3) | |
| buy_valor, sell_valor | INTEGER | İşlem günü; yön `execution.valor_mapping` | **A4 doğrulandı (24.09.2026)** |
| entry_fee, exit_fee | DOUBLE | Ham TEFAS kaynak ücret alanları; TEFAS/BES execution maliyeti değildir, execution fee deterministik 0 (ADR-27) | Ham metin `*_raw`, `parse_status` (D07, S3b-0c) |
| first_trading_time, last_trading_time | VARCHAR | Emir saatleri (metin) | |
| tax_category | VARCHAR | borsapy vergi sınıfı | |
| withholding_rate | DOUBLE | Snapshot günü oranı (ondalık) | Lot oranı alım tarihine göre ayrıca hesaplanır (P04) |
| applied_fee, expense_ratio | DOUBLE | Ondalık | |
| aum_now, investor_count, risk_value | DOUBLE/BIGINT/INTEGER | Anlık; **tarihçesi yok** → tarihsel özellik değil (H02) | |
| tefas_status | VARCHAR | TEFAS durum metni | → `can_buy`, `can_sell` türetilir (D05) |
| can_buy, can_sell | BOOLEAN | Status metninden türetilmiş cache/audit alanı; yürütme tek doğruluk kaynağı olarak `tefas_status` kullanır; mismatch warning | S5-6c-fix2 |
| first_nav_date, last_nav_date, n_nav | DATE, DATE, INTEGER | Depodaki gözlenen geçmiş | `observed_history_age`; **kuruluş tarihi değildir** (D09) |
| info_ok | BOOLEAN | Profil cevabında en az bir tanımlayıcı/kategori alanı var | Taşınmış bilgi yeni başarı sayılmaz |
| info_fetched | BOOLEAN | Profil bugün alındı | Taşınmış profil = False |
| last_success_at | TIMESTAMP | `fund_info` çağrısının son başarılı tamamlanma zamanı | Taşıma/skip zamanı güncellenmez; legacy NULL strict canlı B0'da reddedilir; D itibarıyla ≤7 takvim günü |
| last_success_source | VARCHAR | Başarı zamanı kaynağı: `fetch` gerçek başarılı çağrı tamamlanması; `proxy_ingest_time` legacy geri doldurma vekili | `proxy_ingest_time` dönemleri “vekil zaman, PIT kanıtı değil” diye raporlanır |
| source_published_at → **ingested_at** | TIMESTAMP | Snapshot/depoya giriş zamanı | Profilin başarılı alım zamanı değildir; eski adın dönüşümü D01 |

## Backtest execution metadata (`prepare` sonucu: `execution_meta_by_date`)
| Alan | Tip | Tanım / zaman sınırı |
|---|---|---|
| decision_date, fund_code | DATE, VARCHAR | MultiIndex anahtarı; karar günü × NAV evrenindeki fon |
| buy_valor, sell_valor | Sayısal | Yalnız yürütme girdisi; özellik/X/model girdisi değildir |
| entry_fee, exit_fee | DOUBLE | Ham `fund_master` fee değerleri kaynakta korunur. TEFAS/BES için execution meta alanları deterministik 0'dır; eksik/non-zero/geçersiz kaynak değer fee block veya `fee_assumed_zero` provenance üretmez (ADR-27). Özellik/X/model girdisi değildir |
| can_buy, can_sell, tefas_status | BOOLEAN, VARCHAR | Yön maskeleri `tefas_status` metninden; saklanan bayrak uyumsuzluğu `status_flag_mismatch` audit bilgisi. A3 öncesi status “İşlem Görüyor” varsayılır |
| tax_category | VARCHAR | Günlük yürütme meta alanı; post-PIT NULL/boş/boşluk o fonun BUY/SELL yönlerini `missing_tax_category` ile kapatır. Riskli lot `tax_rate`'i alış günü kategorisiyle hesaplanıp lotta sabitlenir; sonraki profil revizyonları açık lotu değiştirmez. A3 öncesi kategori provenance/varsayımı aynen korunur; FEATURE_COLUMNS/model girdisi değildir |
| execution_source, status | VARCHAR | `PIT` ya da `A3_CURRENT_PROFILE`; A3 etiketi “A3 varsayımlı, PIT kanıtı değil” |
| source_snapshot_date, last_success_at | DATE, TIMESTAMP | Kaynak snapshot ve başarı zamanı; önce karar/kesim anında yayımlanmış en yeni snapshot seçilir, sonra başarı ≤ karar günü `project.runs.morning` (Europe/Istanbul; varsayılan 09:15) ve yaşı ≤7 takvim günü doğrulanır. Seçilen en yeni snapshot eski/eksikse fon işlem dışıdır; daha eski snapshot'a geri dönülmez. |
| status_flag_mismatch | BOOLEAN | Status maskesi ile kayıtlı yön flag'i farklı veya NULL; audit bilgisi |
| buy_reason, sell_reason | VARCHAR | Fon bazında işlem yok gerekçesi; `status_buy_closed` / `status_sell_closed` ilgili yönü status metnine göre kapatır; bilinmeyen status iki yönü kapatır. `missing_tax_category` iki yönü kapatır. Eksik/gelecek/eski profil diğer fonları bloke etmez |
| published_at, available_from | TIMESTAMP, DATE | `published_at` kaynak `source_published_at` varsa yayın zamanı; `available_from` snapshot tarihi |
| publication_time_assumption | VARCHAR | `source_timestamp` veya yayın zamanı yoksa `date_only_pit_assumption`; timestamp'siz snapshot için yalnız tarih-as-of sınırı varsayılır |

Görünüm `Store.fund_master_history()` snapshot geçmişinden kurulur; post-PIT her karar için son uygun snapshot, karar sabahı cutoff'ına kadar yayımlanmış olmalıdır. Başarı timestamp'i de aynı cutoff'tan geçmeli; D günü cutoff sonrası profil D kararına değil sonraki kararlara adaydır. Yayın timestamp'i olmayan snapshot'lar tarih-only PIT varsayımıyla etiketlenir. Ön PIT A3 yürütme varsayımı en son profilden taşınabilir ancak PIT kanıtı değildir. Bu görünüm FEATURE_COLUMNS/label tablolarını değiştirmez.

## Backtest events — yürütme engeli
| Alan | Tip | Tanım |
|---|---|---|
| event | VARCHAR | `execution_blocked`: karar günü hedefi işlem gerektirirken o fonun execution profili işlemi kapatmıştır |
| code | VARCHAR | İşlem engeli olan fon kodu |
| action | VARCHAR | Engellenen yön (`BUY` / `SELL`) |
| reason | VARCHAR | `buy_reason` / `sell_reason` kaynağı; eksik satır için `missing_execution_profile` |

## fund_nav (anahtar: `fund_code, observed_date`)
| Kolon | Tanım |
|---|---|
| observed_date (eski: date) | NAV'ın ait olduğu gün |
| price | Birim fiyat (TL) |
| source_available_at | Karar için görülebilirlik zamanı: tarihsel yeniden kurmada `cal[i+1] 09:00` (A1); canlı kayıtta gerçek |
| ingested_at, run_id | Depoya giriş; **tarihsel testte filtre değildir** |
| revision_id | Aynı `observed_date` için sonraki farklı fiyat (revizyon) — S3b-0c |

## macro (anahtar: `series, date`)
| Kolon | Tanım |
|---|---|
| series | usdtry, eurtry, cpi_index, policy_rate |
| date | Gözlem dönemi (günlük: gün; aylık: ayın 1'i) |
| value | Kur TL; TÜFE endeks (2003=100); politika faizi **yüzde** (özellikte /100) |
| available_from | Görülebilirlik: gerçek yayın zamanı yoksa varsayım (A2: TÜFE dönem+35 gün ≈ sonraki ayın 3–6'sı) |
| assumption_id | A2 / gerçek | S3b-0c |
| ingested_at | Depoya giriş |

## fund_features (parquet; anahtar: `feature_asof, fund_code`) — S3b-1
| Kolon | Tanım |
|---|---|
| feature_asof | t (NAV ≤ t) |
| decision_at | Panelde bir sonraki NAV işlem günü; terminal t satırında `calendar.holidays` ile bulunan sonraki iş günü (terminal t+1 panelde olmasa da NaT değil) |
| label_end, label_available_at | t+22, cal[t+23]; panel sonundaki terminal satırda NaT; paneldeki son olgun label'ın availability'si terminal karar günüdür |
| feature_ready, label_ready | X tam mı (FEATURE_COLUMNS tümü dolu); y olgunlaşmış mı (label_end ≤ son NAV) — ayrı maskeler (H03) |
| eligible_at_decision | O günkü bilgiyle evren üyeliği: t'de NAV var, gözlenen geçmiş ≥ 252, data_stale değil |
| policy_today_excluded | Bugünkü politika dışlaması (kara liste / durum metni / can_buy); özellik DEĞİL, yalnızca "bugünkü politika" koşusu (V05) |
| y | log(NAV[t+22]/NAV[t+1]) − Σ_{h=t+2..t+22} log(1+cash_h); cash_h = nakit vekili sepetin brüt basit günlük getirisi (tek kaynak: `janus.backtest.data.cash_proxy_returns`, P02); çıkarım satırlarında NaN |

FEATURE_COLUMNS beyaz listesi (`src/janus/features/fund_features.py`; dışı → ValueError, TEMPORAL §9.1):
- `log_ret_{1,5,21,63,126,252}`: log(NAV_t/NAV_{t−k})
- `cash_excess_{21,63}`: log getiri − log1p(cash) birikim farkı (aynı pencere)
- `ewma_vol` (λ=0,94, yıllık), `vol_63` (63g std, yıllık)
- `dd_from_peak_126` = 1 − NAV / 126g kayan zirve (pencere içi max DD değil); `dist_peak_252` = NAV / 252g zirve − 1
- `pct_rank_{63,252}`: o gün eligible fonlar içinde yüzdelik sıra (gün bazında)
- `umbrella_type_code`, `tax_rate`: yarı-sabit meta, A3 varsayımı
- makro: `policy_rate, d_policy_63, cpi_yoy, real_rate, usdtry_ret63, usdtry_vol21` (PIT, available_from)
- piyasa: `eq_trend63, eq_vol21, breadth200`
- `aum_now`, `investor_count`, `tefas_status` tarihsel özellik değildir (H02)

## predictions (parquet; anahtar: `decision_at, fund_code, model_id`) — S3b-2/3
| q10, q50, q90 | Model kuantilleri |
| model_id, train_max_t | Refit tarihi; eğitim son t (≤ decision_at − 23) |
| lower, upper, alpha_D, n_calib, quality_flag | CQR+ACI çıktıları (S3b-3) |
| p_value, n_calib, quality_flag | Conformal p-değeri (S3b-5-2; H0: y ≤ 0; skor V = y − q50; V̂ = −q50; p = (1 + #{V_i < V̂})/(n+1); kalibrasyon TEMPORAL §6 olgunlaşma kuralı; yetersiz → NaN + `insufficient_calibration`) |
| selected (BH) | Benjamini–Hochberg adım-up seçimi, q = `conformal.fdr_q` (0,20; ızgara 0,10/0,30 rapor satırı); küme boş → nakit (ADR-18) |
| ks2 alanları | `ks2` (bayrak), `ks2_features` (cash_excess_21 çerçevesi), `ks2_days` (nakde düşülen rebalance gün sayısı); pencere: rolling(63).mean().shift(23) — olgunlaşmış 63 iş günü, label_available_at ≤ D (S3b-5-3; kill-switch v2 whipsaw nedeniyle red — MB-035) |

## selection_<D>.parquet (günlük B2c-fdr seçim; anahtar: `(decision_at, fund_code)`) — S5-0b/S5-6b2
| Kolon | Tip | Tanım |
|---|---|---|
| decision_at | DATE | Sinyal/öneri karar günü; paper proposal as-of doğrulaması |
| fund_code | VARCHAR | Fon kodu |
| p_value | DOUBLE | Conformal p-değeri (H0: y ≤ 0); yetersiz kalibrasyon/NaN q50 → NaN |
| selected_q10, selected_q20, selected_q30 | BOOLEAN | Benjamini–Hochberg adım-up seçimi; q20 kanonik, q10/q30 gölge |
| lower | DOUBLE | CQR alt sınır (target 0.20) o gün için |
| q50 | DOUBLE | Model ortanca tahmini o gün için |

Satır sayısı = o gün tahmini verilen fon sayısı; seçilen küme boşsa tüm `selected_*` False.

## BES predictions (`data/predictions/bes_*`) — S5-5
| Dosya | Tanım |
|---|---|
| `bes_features.parquet` | EMK evreni için fon-gün paneli; şema `fund_features.parquet` ile aynı; `tax_rate = 0` |
| `bes_predictions.parquet` | EMK evreni için walk-forward OOS tahminler; şema `predictions.parquet` ile aynı |
| `bes_calibrated_target_020.parquet` | EMK evreni için CQR+ACI; şema `calibrated_target_020.parquet` ile aynı |
| `bes_selection_<D>.parquet` | EMK evreni için B2c-fdr seçimi; yalnızca q20; şema `selection_<D>.parquet` ile aynı |

## bes_plan (DuckDB; anahtar: `change_id, fund_code`) — S5-5 / S5-6d
| Kolon | Tip | Tanım |
|---|---|---|
| change_id | VARCHAR | YYYY-NN formatında yıllık sıra (örn. 2026-01); her gerçek dağılım değişiminde bir kez |
| date | DATE | Plan tarihi |
| year | INTEGER | Plan yılı |
| month | INTEGER | Plan ayı |
| fund_code | VARCHAR | Hedef fondaki kod (seçilen riskli fon veya B0 EMK PPF sepet üyesi) |
| target_weight | DOUBLE | Hedef ağırlık (fon/PYŞ tavanları sonrası) |
| p_value | DOUBLE | Conformal p-değeri (q20 seçiminden; B0 PPF satırında NULL) |
| lower | DOUBLE | CQR alt sınır (B0 PPF satırında NULL) |
| q50 | DOUBLE | Model ortanca tahmini (B0 PPF satırında NULL) |
| note | VARCHAR | "seçim" (riskli) / "B0 nakit" (PPF sepeti) |

**Append-only (S5-6d / F14):** satırlar silinmez veya güncellenmez. Yalnız hedef fon→ağırlık vektörü en son kayıtlı dağılımdan farklıysa (fon/PYŞ tavanları ve B0 PPF ayrımı sonrası) yeni `change_id` ile satır eklenir; aynı dağılımda yazma yoktur ve değişiklik hakkı harcanmaz. Seçim boşsa ve kayıtlı plan yoksa varsayılan B0 korunur, hak harcanmaz. Hedef ağırlık toplamı < 1 ise fark raporlanan serbest nakittir (getiri atfedilmez). Aylık plan yılda en fazla `legs.bes.plan.planned + reserve` gerçek değişiklik üretir. Açık soru: hakkın hukuken hangi olayda tükendiği sistem sahibince teyit edilmelidir.

## BES config varsayımları — S5-5
| Anahtar | Değer | Açıklama |
|---|---|---|
| `legs.bes.execution.valor_assumption_id` | A8 | BEFAS valör bilinmiyor; varsayım alış T+1 / satış T+2 |
| `legs.bes.execution.default_buy_valor` | 1 | A8 varsayımı |
| `legs.bes.execution.default_sell_valor` | 2 | A8 varsayımı |
| `legs.bes.tax.switch_tax` | 0.0 | BES'te stopaj yok |
| `legs.bes.universe.exclude_state_contribution_patterns` | ["devlet katkı"] | Devlet katkısı fonları dışarı; rapora listelenir |

## paper_* (kağıt-ticaret defteri; S5-1)
Yalnızca `janus paper` komutlarından yazılır; Parquet export'a dahil edilmez.

### janus_pilot_epoch
| Kolon | Tip | Tanım |
|---|---|---|
| epoch_id | INTEGER | Tek aktif pilot epoch'un sabit anahtarı (1) |
| started_at | TIMESTAMP | `janus paper reset --archive-v1` sonrası pilot sayacının başlangıç anı |
| initialized_at | TIMESTAMP | Yeni defterin ilk başarılı `janus paper init` zamanı; sonraki init'ler üzerine yazmaz |
| published_at | TIMESTAMP | Epoch kaydının yayımlandığı/kaydedildiği an |
| available_from | TIMESTAMP | KPI hesaplarında bu epoch'un verisinin kullanılabilir olduğu an |

Reset yalnızca tam `paper_` önekiyle başlayan tüm canlı tabloları (`paperX...` dahil değil) `_v1` olarak atomik DDL rename ile arşivler; dolu tabloların yanı sıra boş tablolar da arşivlenir. Hedef çakışması veya manifest'siz `_v1` artığı varsa hiçbir tablo taşınmaz. Arşiv rename'leri, yeni şema, epoch ve manifest tek transaction içindedir; hata hepsini geri alır. `runs` ve model/veri tabloları korunur. KPI penceresinin başlangıcı, 28 günlük üst sınır ile epoch `started_at` tarihinin geç olanıdır; eski `runs` satırları sayaca girmez.

### janus_paper_archive_manifest
| Kolon | Tip | Tanım |
|---|---|---|
| archived_tables | VARCHAR (JSON listesi) | Tamamlanan arşivde `_v1` olarak taşınan tablo adları; idempotent reset doğrulaması için |
| archived_at | TIMESTAMP | Arşiv transaction'ının zamanı |
| published_at | TIMESTAMP | Manifest kaydının yayımlandığı an |
| available_from | TIMESTAMP | Manifest bilgisinin kullanılabilir olduğu an |

Manifest yalnız arşiv transaction'ı başarıyla tamamlanınca vardır. Idempotent retry, manifestteki tam arşiv kümesi mevcutsa, epoch varsa ve yeni `paper_` tabloları boşsa no-op/schema ensure yapar; manifest bulunmayan kısmi `_v1` kümesi kabul edilmez.

### paper_positions
| Kolon | Tip | Tanım |
|---|---|---|
| updated_at | TIMESTAMP | Yazım zamanı |
| fund_code | VARCHAR | Gerçek fon kodu; `CASH_PROXY` sanal pozisyon olarak oluşturulmaz (eski kayıt varsa fail-closed) |
| units | DOUBLE | Güncel birim |
| cost_basis | DOUBLE | Ağırlıklı ortalama maliyet |

### paper_b0_memberships
| Kolon | Tip | Tanım |
|---|---|---|
| membership_date | DATE | Bu aktif B0 üyeliğinin gözlendiği NAV/as-of tarihi |
| snapshot_date | DATE | B0 seçiminde kullanılan PIT `fund_master` snapshot tarihi |
| fund_code | VARCHAR | Seçilen gerçek B0 fon kodu; aynı fonun her üyelik snapshot'ı ayrı saklanır |
| founder_code | VARCHAR | Snapshot anındaki PYŞ kimliği; kurucu adı kullanılmaz |
| buy_valor, sell_valor | INTEGER | Snapshot anındaki bilinen alış/satış valörleri (iş günü) |

PK: `(membership_date, fund_code)`. Üyelik kayıtları append-only tutulur. Reload'da aktif B0 yeni üyelerden kurulur; eski üyeye ait lot/pozisyon satılana kadar B0 equity bileşeninde kalır ve hedefi sıfır olur. Tarihsel üyelik fonu güncel PIT master/meta veya as-of NAV evreninde yoksa reload fail-closed olur; hiçbir pozisyon sessizce atlanmaz.

### paper_lots
| Kolon | Tip | Tanım |
|---|---|---|
| updated_at | TIMESTAMP | Yazım zamanı |
| fund_code | VARCHAR | Fon kodu |
| units | DOUBLE | Lot birimi |
| cost_per_unit | DOUBLE | Giriş fiyatı (komisyon dahil) |
| bought_date | DATE | Alım tarihi |
| tax_rate | DOUBLE | Lotun stopaj oranı (P04) |
| available_date | DATE | Satılabilir tarih (buy_valor sonrası); takvim ekseni dışında NULL olabilir |
| bought_idx | BIGINT | Alımın TEFAS NAV işlem-günü indeksi (restart'ta lot yaşını korur) |
| available_idx | BIGINT | Mutlak satılabilir NAV indeks değeri; tarih NAV ekseninde yokken NULL'ı 0'a düşürmez |

### paper_cash
| Kolon | Tip | Tanım |
|---|---|---|
| updated_at | TIMESTAMP | Yazım zamanı |
| cash | DOUBLE | Serbest nakit bakiyesi |
| initial_capital | DOUBLE | Başlangıç sermayesi (yüzde tabanı) |

INIT yalnız `paper_cash.cash=capital` yazar; `paper_positions`, `paper_lots` ve alacak boştur. İlk karar D'de BH kümesi boşsa tam B0 hedefi gerçek B0 fon kodlarına eşit BUY olarak açılır; alım ilk kez D+1 NAV'de dolar. Paper init `--date D` (varsayılan config `project.timezone` saatindeki bugünkü tarih) karar günü olarak kullanır. NAV paneli `≤ D`, her fonun PIT `fund_master` profili `snapshot_date ≤ D` ile ayrıca seçilir; CLI `decision_asof`, `meta_asof`, `nav_asof` değerlerini gösterir. `meta_asof` kullanılan fon profillerinin en yeni snapshot tarihidir; üyelik tablosu her kodun kendi snapshot tarihini saklar. D−1 NAV / D meta farkı tek başına blok değildir.

Kağıt öneri kullanılabilir bütçesi = karar D'deki serbest nakit + `settle_idx ≤ decision_idx` alacakları. Aynı önerideki satış gelirleri ve henüz vadesi gelmemiş alacaklar bütçeye girmez. Riskli BUY ihtiyacı bütçeyi aşarsa riskli BUY atlanır; B0 BUY kalan bütçeyle, eşit oranda ölçeklenir. D+1 fill yeni proposal/finansman üretmez. B0 3–5 aday hedefler; 1–2 aday uyarıyla kabul, 0 aday fail-closed hold. Kanıtlı risky lot 21 işlem günü min-hold'a tabidir; B0 lotları muaftır, DD > %12 istisnası korunur. Backtest B0 endeks vekilidir, paper B0 gerçek lotlardır; getiri eşitliği varsayılmaz ve fark ayrıca raporlanır.

### paper_proposals
| Kolon | Tip | Tanım |
|---|---|---|
| proposal_id | VARCHAR | `YYYYMMDD-NN` |
| date | DATE | Karar günü D |
| created_at | TIMESTAMP | Öneri üretim zamanı |
| expires_at | TIMESTAMP | Geçerlilik: D 12:00 |
| status | VARCHAR | proposed / filled / partially_filled / pending / rejected / skipped / expired |
| orders_json | VARCHAR | Emir listesi JSON |
| evidence_count | INTEGER | Günlük BH q20 kümesinin fon sayısı; emir adedinden bağımsız |
| evidence_codes | VARCHAR | BH q20 kümesi fon kodları JSON listesi (ADR-20 bölümlemesi) |
| target_weights_json | VARCHAR | Tam hedef ağırlıklar; CASH_PROXY dahil |
| execution_blocks_json | VARCHAR | Fon bazında BUY/SELL engel kayıtları (`record_type=block`, reason, snapshot yayım varsayımı); date-only PIT yayın varsayımları ayrıca `publication_assumption` kaydıdır. `source_published_at IS NULL` post-first-PIT kaydı `date_only_pit_assumption` olarak etiketlenir. |

Post-first-PIT icrada karar D için en yeni `snapshot_date ≤ D` profil geçerlidir; profilin `last_success_at ≤ D 09:15` ve yaşı ≤ 7 takvim günü olmalı, alış/satış valörü, tanınan `tefas_status` ve `tax_category` bilinmelidir. Yön maskesi status metninden türetilir; saklı yön bayrakları audit-only'dir. Fee alanlarının eksikliği/değeri emri etkilemez; TEFAS/BES execution fee deterministik 0'dır (ADR-27). Eksik/eski/geç yayımlanmış profil için eski snapshot'a geri düşülmez; ilgili fon emri engellenir ve reason bu JSON'a kalıcı yazılır. Diğer fon emirleri devam edebilir; mevcut lot son bilinen resmi NAV ile değerlenir.

### paper_receivables
| Kolon | Tip | Tanım |
|---|---|---|
| settle_idx | BIGINT | Satış nakdinin tahsil edileceği mutlak NAV işlem-günü indeksi |
| amount | DOUBLE | Vadesi gelmemiş satış alacağı |

### paper_order_results
| Kolon | Tip | Tanım |
|---|---|---|
| proposal_id | VARCHAR | Bağlı öneri |
| order_index | INTEGER | Öneri içindeki emir indeksi |
| fund_code | VARCHAR | Emir fonu |
| side | VARCHAR | BUY / SELL |
| requested_value | DOUBLE | İstenen işlem değeri |
| outcome | VARCHAR | filled / pending / rejected |
| reason | VARCHAR | Bekleme/reddedilme nedeni; başarıda boş |
| units | DOUBLE | Gerçekleşen birim |
| fill_date | DATE | Sonuç değerlendirme tarihi |

### paper_fills
| Kolon | Tip | Tanım |
|---|---|---|
| fill_id | VARCHAR | `proposal_id-timestamp-batch_index` |
| proposal_id | VARCHAR | Bağlı öneri |
| fund_code | VARCHAR | İşlem gören fon |
| side | VARCHAR | BUY / SELL |
| units | DOUBLE | İşlem birimi |
| price | DOUBLE | Dolum fiyatı (NAV) |
| gross | DOUBLE | Brüt işlem tutarı |
| fee | DOUBLE | Giriş/çıkış komisyonu |
| tax | DOUBLE | Satışta ödenen stopaj |
| realized_gain | DOUBLE | Satışta gerçekleşen kazanç |
| fill_date | DATE | Gerçekleşme tarihi |

### paper_equity
| Kolon | Tip | Tanım |
|---|---|---|
| portfolio_name | VARCHAR | `live` (canlı) veya gölge portföy adı (örn. `B0_cash`) |
| date | DATE | Anlık gün |
| equity | DOUBLE | nakit + alacak + risky + CASH_PROXY slotu |
| cash | DOUBLE | Serbest nakit |
| receivables | DOUBLE | Vadesi gelmemiş satış alacakları |
| risky_value | DOUBLE | Riskli fon pozisyonlarının değeri |
| slot_value | DOUBLE | Gerçek fon bazlı B0 üyelerinin (satılmamış eski üyeler dahil) değeri; sanal CASH_PROXY lotu değildir |
| nav_asof | DATE | Değerlemeye kullanılan son NAV tarihi |

PK: `(portfolio_name, date)`. S5-1 öncesi tekli PK `(date)` olan tablolar `portfolio_name` kolonu eklenince yeniden oluşturulur.
Gölge portföy bileşenleri bağımsız defterden hesaplanmıyorsa `cash`, `receivables`, `risky_value`, `slot_value` NULL (bilinmiyor) kalır; sıfır varsayılmaz.

### paper_shadow_runs
| Kolon | Tip | Tanım |
|---|---|---|
| portfolio_name | VARCHAR | Gölge portföy kodu |
| date | DATE | Koşunun NAV as-of günü |
| status | VARCHAR | `ok` / `failed` |
| error | VARCHAR | Başarısız portföy için hata özeti; başarıda NULL |
| n_rows | BIGINT | Başarılı equity serisindeki satır sayısı; hata/boş seride 0 |
| nav_asof | DATE | Değerlemede kullanılan son NAV tarihi |

PK: `(portfolio_name, date)`. Hatalı gölge serisi `paper_equity` içine yazılmaz; diğer portföyler bağımsız devam eder.

### paper_reconcile
| Kolon | Tip | Tanım |
|---|---|---|
| date | DATE | Mutabakat tarihi |
| equity | DOUBLE | Defter equity |
| cash | DOUBLE | Serbest nakit |
| receivables | DOUBLE | Alacaklar |
| risky_value | DOUBLE | Riskli değer |
| slot_value | DOUBLE | CASH_PROXY değeri |
| identity_ok | BOOLEAN | Saklı tarihli equity ve bileşenleri (`cash + receivables + risky + slot`) yeniden hesaplanan defter durumuyla tutarlı |
| liquidation_value | DOUBLE | Tasfiye değeri |
| max_weight_diff | DOUBLE | Hedef/gerçek max ağırlık farkı |
| n_expired | INTEGER | Süresi dolmuş öneri sayısı |

## Config eşikleri (`paper`; S5-6a/S5-4)
| Anahtar | Varsayılan | Tanım |
|---|---|---|
| `min_hold_days` | 21 | Kanıtla alınan lot için işlem-günü asgari tutma; DD > %12 istisnası |

### KPI eşikleri (`paper.kpi`; S5-4)
| Anahtar | Varsayılan | Tanım |
|---|---|---|
| `nightly_success_min` | 0.95 | Son 28 takvim gününde başarılı nightly / beklenen günlük nightly eşiği; eksik run paydada başarısız sayılır |
| `morning_deadline` | "09:30" | Sabah mesajı saat sınırı |
| `reconcile_max_pp` | 0.5 | Mutabakat max ağırlık farkı (pp) |
| `identity_tol` | 1e-6 | Defter özdeşlik toleransı |

28 günlük KPI penceresinde nightly paydası her takvim günüdür; morning ve proposal paydaları yalnızca `calendar.holidays` hariç configured BIST iş günleridir. İş günü olmayan morning koşusu `status=ok`, `business_status=no_new_data` ile kaydedilir; KPI'da başarısız/eksik sayılmaz.

## runs (anahtar: `run_id`)
`kind ∈ {ingest_tefas, ingest_macro, nightly, morning, backtest, suite}`, `status ∈ {ok, partial, failed}`, `summary` JSON (fon kodları ve sayılar; tutar yok), `data_asof`. Morning summary gönderim denetimi için `sent_ok` (Telegram adapter onayı) ve `sent_at` (ISO 8601 yerel timestamp) içerir. `export_parquet` sonrası `summary.snapshot_asof` = parquet'lerin max mtime'ı (ISO 8601; S3b-0b/0c-5). Nightly koşusunda `summary.steps` her adım için sözlüktür: `status ∈ {ok, partial, failed, skipped}`, `business_status` varsa özgün paper işi sonucu, `n_rows` (çıktı satırı sayısı), `elapsed_seconds`; hata varsa güvenli `error` metni. Genel durum: failed varsa failed; yoksa partial varsa partial; aksi ok. Partial akışı durdurmaz; failed yalnız bağımlı zincirde ardıl adımları skipped yapar.

S5-6c-fix2: DuckDB `runs.summary.snapshot_asof` Parquet dışa aktarımı başlamadan önce UTC ISO timestamp olarak yazılır ve Parquet `runs` satırıyla birebir aynıdır. `Store.from_parquet().snapshot_asof` ise generation dosyalarının max mtime'ıdır; farklı ölçümdür.

## Backtest çıktı kolonları (S3b-0c; LEDGER §3–4)
| Kolon | Tanım |
|---|---|
| `liquidation_value` | `V − (tüm pozisyonlar bugün bozulsa ödenecek stopaj)`; defter lotlarından hesaplanır |
| `unrealized_tax` | Elde edilen pozisyonların bugünkü bozumunda ödenecek stopaj (lot bazında, tarihe göre oran) |
| `cash_index` | B0 nakit sepet slotunun MTM serisi (defter içi PPF sepeti; brüt getirinin birikimi) |
| `cash_codes` | Nakit sepet slotunu oluşturan para piyasası fonu kodları (eşit ağırlık; config'ten) |

## Metrikler (vergi ve maliyet sonrası; `VALIDATION_PROTOCOL.md`)
- `ret_net = log(V_t/V_{t−1})`; `V = cash + receivables + Σ units × NAV_last`; vergi satışta bir kez düşülür, `V` tahakkuk içermez; `liquidation_value` ayrı.
- `excess_cagr_pp` = CAGR(strateji) − CAGR(B0 defteri), **yüzde puan**.
- `dd = 1 − V_t / max(V_{≤t})`; tetik `> 0,12` → exposure ≤ `risk.dd_exposure` (0,65; 0c-5); serbest `< 0,06` ve planlı rebalance.
- `drift = |hedef − mevcut|`; eşik 0,03 / 0,06 (LEDGER §6).
- `coverage_gap_pp` = gözlenen kapsama − hedef, tarih bloğu bazında.
- `fresh_ratio` = eligible evrende `stale_days = 0` oranı; referans = beklenen son NAV günü (`decision_date − 1 işlem günü`); `source_stale` tüm kaynak ≥ 2 gün eski.
