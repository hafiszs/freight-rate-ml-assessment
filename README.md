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

### 1. Setup environment

Disarankan pakai virtual environment supaya dependensi tidak bentrok dengan package Python lain di komputer kamu.

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` sudah termasuk `jupyter` & `ipykernel`, jadi tidak perlu instalasi tambahan untuk membuka notebook.

### 2. Jalankan notebook

Buka Jupyter dari dalam virtual environment yang sudah dibuat:

```bash
jupyter notebook
```

(atau `jupyter lab`, atau buka folder ini langsung lewat VS Code / PyCharm dan pilih interpreter dari `.venv` sebagai kernel-nya)

Lalu jalankan berurutan, **dari atas ke bawah** (menu **Run > Run All Cells**, atau tombol ⏩ *"Restart Kernel and Run All"*):

1. **`notebooks/01_eda.ipynb`** — eksplorasi data, tidak menghasilkan file output, murni analisis.
2. **`notebooks/02_modeling.ipynb`** — cleaning, feature engineering, training, evaluasi. Notebook ini otomatis menghasilkan:
   - `validation_predictions.csv` (di root folder)
   - `data/december_chart_inputs.csv` yang sudah terisi kolom `predicted_rate`
   - Notebook ini juga langsung menjalankan `score.py` di sel terakhir dan menampilkan chart `scorer_results/candidate_december.png` inline.

Tidak perlu menjalankan apa pun lewat terminal untuk notebook-nya — cukup buka file-nya dan klik "Run All", semua sel (termasuk instalasi kecil, loading data, training, sampai generate file output) akan berjalan otomatis sesuai urutan.

### 3. (Opsional) Validasi ulang lewat terminal

Kalau ingin mengecek ulang file output tanpa membuka notebook lagi (misalnya setelah `02_modeling.ipynb` selesai dijalankan sekali):

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Output yang diharapkan:
```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
Final validation metrics are calculated by Spotter after submission.
```

> Catatan: `validation_predictions.csv` dan `data/december_chart_inputs.csv` (kolom `predicted_rate`) di repo ini **sudah** merupakan hasil run notebook sebelumnya, jadi langkah 3 ini bisa langsung dijalankan tanpa perlu re-run notebook dulu kalau hanya mau memverifikasi format & melihat chart-nya.

## Ringkasan pendekatan

**Data quality issues yang ditemukan & ditangani** (detail lengkap di `01_eda.ipynb`):
- `weight` bernilai negatif pada 292 baris — terbukti murni *sign error* (magnitude sama dgn data positif) -> diperbaiki dengan `abs()`.
- `weight` & `market_index` memiliki missing values (~0.6-0.8%) -> diimputasi (median per `equipment` utk weight; kurva musiman tahunan utk market_index).
- 22 baris `distance` tidak konsisten dengan jarak garis-lurus dari koordinatnya -> dibiarkan (jumlah sangat kecil), model tree-based robust terhadap ini.
- 8 kota di `validation.csv` tidak pernah muncul di `train_test.csv` -> fitur tidak dibuat terlalu bergantung pada identitas kota, memakai koordinat lat/lon sebagai gantinya.
- `december_chart_inputs.csv` tidak memiliki kolom `market_index`/`quote_signal` sama sekali -> nilai ini diestimasi dari kurva musiman tahunan (harmonic regression atas `day_of_year`) yang di-fit dari `train_test.csv`.

**Split train/validasi:** time-based, bukan random — training Jan-Agu 2025, holdout internal Sep-Okt 2025 (dipakai early stopping) — supaya validasi mensimulasikan tugas sebenarnya (ekstrapolasi ke bulan yang belum pernah dilihat model, karena `validation.csv` & Desember berada di Nov-Des 2025). Model final kemudian dilatih ulang di seluruh `train_test.csv` (Jan-Okt) sebelum menghasilkan prediksi akhir.

**Model:** LightGBM dipilih karena robust terhadap outlier/noise yang ditemukan saat EDA, mampu menangkap interaksi non-linear antar fitur (equipment x distance x musim) tanpa scaling manual, mendukung fitur kategorikal secara native, dan menangani missing value secara internal.
