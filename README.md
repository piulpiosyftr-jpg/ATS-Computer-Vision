# ATS Computer Vision — Klasifikasi Huruf Tulisan Tangan (RE604)

Proyek ini mengimplementasikan pipeline **Computer Vision** untuk mengenali huruf tulisan tangan (A–Z) menggunakan dataset **EMNIST Letters**. Fitur gambar diekstraksi dengan **HOG (Histogram of Oriented Gradients)**, lalu diklasifikasi dengan **Support Vector Machine (SVM)** yang dioptimasi melalui **Grid Search**.

## Informasi Penulis

| | |
|---|---|
| **Nama** | Viona Syafitri |
| **NIM** | 4222301063 |
| **Mata Kuliah** | Computer Vision (RE604) |

## Struktur Proyek

```
ATS_Computer Vision/
├── ATS_Computer_Vision_RE604.ipynb   # Notebook utama (Google Colab / Jupyter)
├── emnist-letters-train.csv          
└── README.md
```

## Deskripsi

Program melakukan klasifikasi **26 kelas huruf** (label 1–26) dari gambar grayscale berukuran **28×28 piksel**. Alur kerja dibagi menjadi empat bagian:

1. **Preprocessing & Eksplorasi Data** — memuat dataset, menyeimbangkan jumlah sampel per kelas, visualisasi gambar, dan pembagian train/test.
2. **Feature Extraction (HOG)** — mengubah piksel mentah menjadi vektor fitur HOG.
3. **Classification (SVM + Grid Search)** — melatih SVM dan mencari hyperparameter terbaik.
4. **Evaluation** — menghitung metrik performa dan confusion matrix.

## Dataset

- **Sumber:** [EMNIST Letters](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- **File:** `emnist-letters-train.csv`
- **Format:** CSV tanpa header; kolom pertama = label (1–26), kolom 1–784 = nilai piksel (28×28)

### Preprocessing dataset

| Langkah | Detail |
|---|---|
| Dataset asli | 88.800 sampel × 785 kolom |
| Balancing | 100 sampel per kelas × 26 kelas = **2.600 sampel** |
| Split data | 80% training (2.080) / 20% testing (520), stratified |
| Random state | 42 |

## Teknologi & Dependensi

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-image
```

| Library | Fungsi |
|---|---|
| `pandas`, `numpy` | Manipulasi data |
| `matplotlib`, `seaborn` | Visualisasi |
| `scikit-learn` | SVM, GridSearchCV, metrik evaluasi |
| `scikit-image` | Ekstraksi fitur HOG |

## Parameter HOG

Parameter HOG yang digunakan (dimodifikasi dari default):

| Parameter | Nilai | Default |
|---|---|---|
| `orientations` | 9 | 8 |
| `pixels_per_cell` | (4, 4) | (8, 8) |
| `cells_per_block` | (2, 2) | (3, 3) |
| `block_norm` | L2-Hys | — |

Setelah ekstraksi HOG, setiap gambar direpresentasikan sebagai vektor **1.296 dimensi**.

## Model SVM & Grid Search

Hyperparameter yang diuji melalui `GridSearchCV` (5-fold CV, 27 kombinasi):

| Parameter | Nilai yang diuji |
|---|---|
| `kernel` | `rbf`, `poly`, `linear` |
| `C` | 0.1, 1, 10 |
| `gamma` | `scale`, 0.1, 0.01 |

**Parameter terbaik:**

```python
{'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}
```

Skor cross-validation terbaik: **0.8457**

## Hasil Evaluasi

### Train / Test Split (80/20)

| Metrik | Training (80%) | Testing (20%) |
|---|---|---|
| Accuracy | 100.00% | **81.15%** |
| Precision | 1.0000 | 0.8159 |
| Recall | 1.0000 | 0.8115 |
| F1-Score | 1.0000 | 0.8116 |

### Cross-Validation (10-Fold pada seluruh 2.600 sampel)

| Metrik | Nilai |
|---|---|
| Accuracy | **84.69%** |
| Precision | 0.8484 |
| Recall | 0.8469 |
| F1-Score | 0.8472 |

> **Catatan:** Bagian evaluasi awalnya direncanakan menggunakan LOOCV (Leave-One-Out Cross Validation), namun diganti dengan **10-Fold CV** agar proses lebih cepat (LOOCV membutuhkan 2.600 iterasi).

## Cara Menjalankan

### Google Colab

1. Upload `ATS_Computer_Vision_RE604.ipynb` ke Google Colab.
2. Upload file `emnist-letters-train.csv` ke direktori Colab.
3. Jalankan sel secara berurutan (**Runtime → Run all**).

### Jupyter Notebook (Lokal)

1. Pastikan Python 3.x dan semua dependensi terinstal.
2. Letakkan `emnist-letters-train.csv` di folder yang sama dengan notebook.
3. Buka notebook dan jalankan semua cell:

```bash
jupyter notebook ATS_Computer_Vision_RE604.ipynb
```

## Alur Pipeline

```
emnist-letters-train.csv
        │
        ▼
  Balancing (100/kelas)
        │
        ▼
  Train/Test Split (80/20)
        │
        ▼
  Ekstraksi Fitur HOG
        │
        ▼
  SVM + GridSearchCV
        │
        ▼
  Evaluasi (Metrik + Confusion Matrix)
        │
        ▼
  10-Fold Cross-Validation
```

## Output Visualisasi

Notebook menghasilkan beberapa visualisasi, antara lain:

- Sampel gambar huruf tulisan tangan
- Perbandingan gambar asli vs representasi HOG
- Confusion matrix (training, testing, dan cross-validation)
- Ringkasan metrik evaluasi

