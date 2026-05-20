# Workflow CI untuk Retraining Model Sentimen Analisis DANA

## Deskripsi

Proyek ini menyediakan workflow Continuous Integration (CI) untuk melakukan retraining otomatis model sentimen analisis aplikasi DANA. Workflow ini akan menjalankan pipeline machine learning secara terjadwal atau manual, melakukan evaluasi model, dan menyimpan hasil training di MLflow untuk tracking eksperimen.

Workflow CI memastikan bahwa model selalu diperbarui dengan data terbaru dan kualitas yang terjaga melalui automasi testing dan deployment.

## Struktur Folder

```
Workflow-CI/
├── .github/workflows/
│   └── ci.yml
├── MLProject/
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLProject
│   ├── requirements.txt
│   ├── DockerHub.txt
│   └── ulasan-aplikasi-dana_preprocessing/
│       ├── data_preprocessed.csv
│       ├── metadata.json
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── X_val.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── y_val.csv
└── README.md
```

## Prerequisites

Sebelum menjalankan workflow, pastikan Anda memiliki:

- Akun GitHub dengan repository yang sudah siap
- Python 3.12.7 atau lebih tinggi (untuk development lokal)
- Akses ke repository GitHub Actions
- MLflow server (opsional, untuk tracking lokal)
- Akun Docker Hub (opsional, untuk deployment ke container registry)
- Google Drive credentials (opsional, untuk backup artifacts)

## Cara Menjalankan Workflow

Workflow dapat dipicu dengan dua cara:

### 1. Automatic Trigger (Push ke Main)
Workflow akan otomatis berjalan ketika ada push ke branch main dengan mengubah file di folder MLProject.

```
git add .
git commit -m "Update model training"
git push origin main
```

### 2. Manual Trigger (Workflow Dispatch)
Anda dapat menjalankan workflow secara manual melalui GitHub Actions tab.

- Buka repository di GitHub
- Masuk ke tab Actions
- Pilih workflow yang sesuai
- Klik tombol "Run workflow"
- Isi parameter yang diperlukan jika ada
- Klik "Run workflow" untuk memulai

## Environment

Proyek ini menggunakan environment dan dependensi berikut:

- Python 3.12.7
- MLflow 2.19.0 - untuk experiment tracking dan model registry
- Scikit-learn 1.8.0 - untuk machine learning algorithms
- Pandas 2.3.3 - untuk data manipulation
- NumPy 2.2.6 - untuk numerical computing
- SciPy 1.17.1 - untuk scientific computing
- Joblib 1.3.0+ - untuk model serialization
- PyArrow 18.1.0 - untuk data format support
- Psutil 7.2.2 - untuk system monitoring
- Cloudpickle 3.1.2 - untuk object serialization

Semua dependensi sudah terdaftar di file `requirements.txt` dan `conda.yaml`.

## Parameter yang Dapat Disesuaikan

Model Logistic Regression memiliki parameter yang dapat dikonfigurasi:

- C (float, default: 1.0) - Strength of regularization (inverse). Nilai lebih kecil berarti regularisasi lebih kuat.
- max_iter (int, default: 1000) - Maksimum jumlah iterasi untuk solver converge.

Parameter dapat diatur ketika memicu workflow manual atau diubah di file `MLProject` untuk automated trigger.

## Output Workflow

Setiap kali workflow berhasil dijalankan, akan menghasilkan output berikut:

- model.pkl - Model Logistic Regression terlatih
- tfidf_vectorizer.pkl - TF-IDF vectorizer yang digunakan untuk feature extraction
- Classification report - Laporan hasil evaluasi model (accuracy, precision, recall, f1-score)
- MLflow artifacts - Semua artifacts disimpan di MLflow untuk tracking dan versioning
- Logs dan metrics untuk monitoring performa training

Semua output dapat diakses melalui MLflow UI pada endpoint yang dikonfigurasi.

## Konfigurasi Detail

### MLflow Configuration

Workflow menggunakan MLflow untuk tracking eksperimen dengan konfigurasi:

- Tracking URI: http://127.0.0.1:5000 (dapat disesuaikan di modelling.py)
- Experiment Name: Sentimen Analisis DANA
- Autolog: Enabled untuk scikit-learn

### Model Features

Model menggunakan TF-IDF vectorization dengan konfigurasi:

- Max features: 5000
- N-gram range: (1, 2)
- Tokenizer berbasis text preprocessing
