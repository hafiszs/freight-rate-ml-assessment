# Freight Rate Prediction — Spotter ML Engineer Assessment

Model prediksi tarif angkutan barang (`posted_rate`) berdasarkan data historis
Januari-Oktober 2025, dipakai untuk memprediksi 12,000 load di `validation.csv`
(November-Desember 2025) dan simulasi harian rute tetap Lexington -> Fort Wayne
sepanjang Desember 2025.

## Ringkasan hasil

| Metrik (internal holdout Sep-Okt 2025) | Nilai |
|---|---|
| MAE  | $131.98 |
| RMSE | $639.97 |
| MAPE | 5.84% |

Model: **LightGBM regressor**, target `log1p(posted_rate)`.

## Struktur proyek

```
data/
  train_test.csv                     # data latih berlabel (Jan-Okt 2025)
  validation.csv                     # 12,000 load yang perlu diprediksi (Nov-Des 2025)
  validation_predictions_template.csv# template load_id resmi
  december_chart_inputs.csv          # rute tetap Lexington->Fort Wayne, 31 hari Des 2025
                                      #  (sudah terisi predicted_rate hasil run notebook 02)
notebooks/
  01_eda.ipynb                       # exploratory data analysis
  02_modeling.ipynb                  # cleaning, feature engineering, training, evaluasi, prediksi final
src/
  features.py                        # fungsi cleaning & feature engineering (reusable, fit-on-train)
score.py                             # skrip penilai resmi dari Spotter
validation_predictions.csv           # OUTPUT: 12,000 baris load_id,predicted_rate
scorer_results/candidate_december.png# OUTPUT: chart hasil score.py
requirements.txt
```

## Cara menjalankan

```bash
python -m pip install -r requirements.txt

# 1. Eksplorasi data
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

# 2. Preprocessing, feature engineering, training, validasi, & generate prediksi final
#    (notebook ini yang menghasilkan validation_predictions.csv dan mengisi
#     kolom predicted_rate di data/december_chart_inputs.csv)
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modeling.ipynb

# 3. Validasi format resmi + generate chart Desember
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Output `score.py` yang diharapkan:
```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

## Ringkasan pendekatan

**Data quality issues yang ditemukan & ditangani** (detail lengkap di `01_eda.ipynb`):
- `weight` bernilai negatif pada 292 baris — terbukti murni *sign error* (magnitude sama dgn data positif) -> diperbaiki dengan `abs()`.
- `weight` & `market_index` memiliki missing values (~0.6-0.8%) -> diimputasi (median per `equipment` utk weight; kurva musiman tahunan utk market_index).
- 22 baris `distance` tidak konsisten dengan jarak garis-lurus dari koordinatnya -> dibiarkan (jumlah sangat kecil), model tree-based robust terhadap ini.
- 8 kota di `validation.csv` tidak pernah muncul di `train_test.csv` -> fitur tidak dibuat terlalu bergantung pada identitas kota, memakai koordinat lat/lon sebagai gantinya.
- `december_chart_inputs.csv` tidak memiliki kolom `market_index`/`quote_signal` sama sekali -> nilai ini diestimasi dari kurva musiman tahunan (harmonic regression atas `day_of_year`) yang di-fit dari `train_test.csv`.

**Split train/validasi:** time-based, bukan random — training Jan-Agu 2025, holdout internal Sep-Okt 2025 (dipakai early stopping) — supaya validasi mensimulasikan tugas sebenarnya (ekstrapolasi ke bulan yang belum pernah dilihat model, karena `validation.csv` & Desember berada di Nov-Des 2025). Model final kemudian dilatih ulang di seluruh `train_test.csv` (Jan-Okt) sebelum menghasilkan prediksi akhir.

**Model:** LightGBM dipilih karena robust terhadap outlier/noise yang ditemukan saat EDA, mampu menangkap interaksi non-linear antar fitur (equipment x distance x musim) tanpa scaling manual, mendukung fitur kategorikal secara native, dan menangani missing value secara internal.
