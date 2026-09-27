# ADR dizini

Bu public araştırma anlık görüntüsünde yalnızca yayınlanan kabiliyetlerle ilgili ADR'ler yer alır; iç süreçte kullanılan diğer ADR'ler kapsam dışıdır. Sözleşmeler için `docs/TEMPORAL_PROTOCOL.md`, `docs/LEDGER_EXECUTION_SPEC.md`, `docs/VALIDATION_PROTOCOL.md` dosyalarına bakın.

| ADR | Başlık | Durum | Tarih |
|---|---|---|---|
| 0016 | Sapma eşikleri ve vergi erteleme heuristiği (taslak) (`../design/ADR-0016-taslak-sapma-esikleri.md`) | Önerildi | 2026-09-23 |
| 0018 | Nakit tabanı + kanıtla risk; kağıt-ticaret adayı geri çekildi (`ADR-0018-nakit-tabani-kanitla-risk.md`) | Kabul | 2026-09-23 |
| 0019 | Alt sınıra göre seçim kanıt değil; S5 = B0 + gölge portföyler (`ADR-0019-s3b4-sonucu-b0-golge-portfoyler.md`) | Kabul | 2026-09-24 |
| 0020 | B2c-fdr (conformal selection, BH q=0,20) üretim adayı; kağıt-ticaret kuralı; kill-switch v2 red (`ADR-0020-b2c-fdr-uretim-adayi-s5-kurali.md`) | Kabul | 2026-09-24 |
| 0022 | Kanıt ufku tutma kuralı (`ADR-0022-kanit-ufku-tutma.md`) | Kabul | 2026-09-25 |
| 0023 | A3 yürütme alanlarının ilk PIT öncesine taşınması (`ADR-0023-a3-yurutme-alanlari-gecmise-tasima.md`) | Kabul | 2026-09-25 |
| 0026 | Birleşik PYŞ yoğunlaşma kısıtı — B0 + riskli (`ADR-0026-birlesik-pys-yogunlasma.md`) | Kabul | 2026-09-25 |
| 0027 | TEFAS/BES işlem komisyonu semantiği (`ADR-0027-tefas-bes-islem-komisyonu.md`) | Kabul | 2026-09-26 |
| 0028 | Artımlı hatta overlap-öncesi revizyonda fail-closed ve PIT öneki dokunulmazlığı (`ADR-0028-artimli-hat-fail-closed.md`) | Kabul | 2026-09-26 |
| 0029 | Canonical PIT prefix bütünlüğü, terminal doğrulama sırası ve panel takvim fail-closed (`ADR-0029-canonical-prefix-terminal-validation-panel-takvim.md`) | Kabul | 2026-09-26 |
| 0030 | Canonical PIT immutability istisnası: y için tek yönlü NaN→değer label maturation (`ADR-0030-y-nan-maturation-immutability.md`) | Kabul | 2026-09-26 |

Şablon: `ADR-template.md`.
