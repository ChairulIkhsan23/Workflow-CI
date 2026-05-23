import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
from pathlib import Path
import argparse

# Argumen parser untuk CI
parser = argparse.ArgumentParser()
parser.add_argument('--C', type=float, default=10.0, help='Regularization strength')
parser.add_argument('--max_iter', type=int, default=500, help='Maximum iterations')
args = parser.parse_args()

# Setup MLflow tracking 
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Sentimen Analisis DANA")

# Aktifkan autolog MLflow untuk scikit-learn
mlflow.sklearn.autolog()

# Direktori hasil preprocessing
data_dir = Path("ulasan-aplikasi-dana_preprocessing")

# Memuat data train, validasi, dan test
print("✓ Memuat data dari direktori preprocessing...")

X_train = pd.read_csv(data_dir / "X_train.csv")
X_val   = pd.read_csv(data_dir / "X_val.csv")
X_test  = pd.read_csv(data_dir / "X_test.csv")

y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
y_val   = pd.read_csv(data_dir / "y_val.csv").values.ravel()
y_test  = pd.read_csv(data_dir / "y_test.csv").values.ravel()

print(f"✓ Ukuran data train      : {len(X_train)}")
print(f"✓ Ukuran data validasi   : {len(X_val)}")
print(f"✓ Ukuran data test       : {len(X_test)}")

print(f"\nDistribusi kelas train :")
for label, count in zip(*np.unique(y_train, return_counts=True)):
    print(f"  {label}: {count} ({count / len(y_train) * 100:.1f}%)")

# Parameter TF-IDF
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)

# Parameter model (menggunakan args dari CLI)
LR_RANDOM_STATE = 42

# Ekstraksi fitur menggunakan TF-IDF
print("\n✓ Membangun representasi TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
)

X_train_tfidf = tfidf.fit_transform(X_train["text"].fillna(''))
X_val_tfidf   = tfidf.transform(X_val["text"].fillna(''))
X_test_tfidf  = tfidf.transform(X_test["text"].fillna(''))

print(f"✓ Ukuran matriks TF-IDF train : {X_train_tfidf.shape}")
print(f"✓ Ukuran matriks TF-IDF val   : {X_val_tfidf.shape}")
print(f"✓ Ukuran matriks TF-IDF test  : {X_test_tfidf.shape}")

# Inisialisasi model Logistic Regression dengan parameter dari CLI
print(f"✓ Training dengan C={args.C}, max_iter={args.max_iter}")
model = LogisticRegression(
    C=args.C,
    max_iter=args.max_iter,
    random_state=LR_RANDOM_STATE,
)

# Training dan logging ke MLflow
print("\n✓ Memulai training model dengan MLflow tracking...")

with mlflow.start_run(run_name="LogisticRegression_TfIdf_DANA"):

    # Melatih model pada data train
    model.fit(X_train_tfidf, y_train)

    # Prediksi pada ketiga split data
    y_pred_train = model.predict(X_train_tfidf)
    y_pred_val   = model.predict(X_val_tfidf)
    y_pred_test  = model.predict(X_test_tfidf)

    # Menghitung metrik evaluasi untuk ditampilkan
    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy   = accuracy_score(y_val, y_pred_val)
    test_accuracy  = accuracy_score(y_test, y_pred_test)

    test_precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
    test_recall    = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
    test_f1        = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)

    val_precision = precision_score(y_val, y_pred_val, average="weighted", zero_division=0)
    val_recall    = recall_score(y_val, y_pred_val, average="weighted", zero_division=0)
    val_f1        = f1_score(y_val, y_pred_val, average="weighted", zero_division=0)

    # Log parameter ke MLflow (autolog akan mencatat, ini tambahan untuk CLI params)
    mlflow.log_param("cli_C", args.C)
    mlflow.log_param("cli_max_iter", args.max_iter)

    # Menampilkan ringkasan hasil evaluasi
    print("\n✓ Hasil Evaluasi Model")
    print(f"✓ Akurasi Train      : {train_accuracy:.4f}")
    print(f"✓ Akurasi Validasi   : {val_accuracy:.4f}")
    print(f"✓ Akurasi Test       : {test_accuracy:.4f}")
    print(f"✓ Precision Test     : {test_precision:.4f}")
    print(f"✓ Recall Test        : {test_recall:.4f}")
    print(f"✓ F1-Score Test      : {test_f1:.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    run_id = mlflow.active_run().info.run_id
    # Simpan run_id ke file 
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    
    print(f"\n✓ MLflow Run ID : {run_id}")
    print(f"✓ Experiment    : Sentimen Analisis DANA")
    print(f"✓ Tracking UI   : file:./mlruns")

print("\n✓ Training selesai.")