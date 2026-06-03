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

# Direktori hasil preprocessing
data_dir = Path("ulasan-aplikasi-dana_preprocessing")

print("✓ Memuat data dari direktori preprocessing...")

X_train = pd.read_csv(data_dir / "X_train.csv")
X_val   = pd.read_csv(data_dir / "X_val.csv")
X_test  = pd.read_csv(data_dir / "X_test.csv")

y_train = pd.read_csv(data_dir / "y_train.csv").values.ravel()
y_val   = pd.read_csv(data_dir / "y_val.csv").values.ravel()
y_test  = pd.read_csv(data_dir / "y_test.csv").values.ravel()

print(f"✓ Ukuran data train: {len(X_train)}")
print(f"✓ Ukuran data validasi: {len(X_val)}")
print(f"✓ Ukuran data test: {len(X_test)}")

print("\n✓ Distribusi kelas train:")
for label, count in zip(*np.unique(y_train, return_counts=True)):
    print(f"  {label}: {count} ({count / len(y_train) * 100:.1f}%)")

# Parameter TF-IDF
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

print("\n✓ Membangun representasi TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=TFIDF_NGRAM_RANGE,
)

X_train_tfidf = tfidf.fit_transform(X_train["text"].fillna(''))
X_val_tfidf   = tfidf.transform(X_val["text"].fillna(''))
X_test_tfidf  = tfidf.transform(X_test["text"].fillna(''))

print(f"✓ Ukuran matriks TF-IDF train: {X_train_tfidf.shape}")
print(f"✓ Ukuran matriks TF-IDF val: {X_val_tfidf.shape}")
print(f"✓ Ukuran matriks TF-IDF test: {X_test_tfidf.shape}")

print(f"\n✓ Training dengan C={args.C}, max_iter={args.max_iter}")
model = LogisticRegression(
    C=args.C,
    max_iter=args.max_iter,
    random_state=42,
)

print("\n✓ Memulai training model...")

with mlflow.start_run(run_name="LogisticRegression_TfIdf_DANA"):

    model.fit(X_train_tfidf, y_train)

    y_pred_train = model.predict(X_train_tfidf)
    y_pred_val   = model.predict(X_val_tfidf)
    y_pred_test  = model.predict(X_test_tfidf)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    val_accuracy   = accuracy_score(y_val, y_pred_val)
    test_accuracy  = accuracy_score(y_test, y_pred_test)

    test_precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
    test_recall    = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
    test_f1        = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)

    mlflow.log_param("C", args.C)
    mlflow.log_param("max_iter", args.max_iter)
    mlflow.log_param("tfidf_max_features", TFIDF_MAX_FEATURES)
    mlflow.log_param("tfidf_ngram_range", str(TFIDF_NGRAM_RANGE))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))
    mlflow.log_param("test_size", len(X_test))

    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)

    # Simpan model dan vectorizer ke file
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump(model, "logistic_regression_model.pkl")

    # Custom PyFunc Model dengan vectorizer
    class ModelWithVectorizer(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.tfidf = joblib.load(context.artifacts["tfidf_vectorizer"])
            self.model = joblib.load(context.artifacts["model"])
        
        def predict(self, context, model_input, params=None):
            if isinstance(model_input, dict):
                texts = model_input.get("text", [])
            elif hasattr(model_input, 'columns'):
                texts = model_input["text"].values
            else:
                texts = model_input
            
            X = self.tfidf.transform(texts)
            return self.model.predict(X)

    # Log custom model dengan artifacts
    mlflow.pyfunc.log_model(
        artifact_path="model_with_vectorizer",
        python_model=ModelWithVectorizer(),
        artifacts={
            "tfidf_vectorizer": "tfidf_vectorizer.pkl",
            "model": "logistic_regression_model.pkl"
        },
        registered_model_name="SentimenDANA_Complete",
        input_example={"text": "aplikasi bagus"}
    )

    # Log vectorizer sebagai artifact
    mlflow.log_artifact("tfidf_vectorizer.pkl")
    
    print("\n✓ Hasil Evaluasi Model")
    print(f"✓ Akurasi Train: {train_accuracy:.4f}")
    print(f"✓ Akurasi Validasi: {val_accuracy:.4f}")
    print(f"✓ Akurasi Test: {test_accuracy:.4f}")
    print(f"✓ Precision Test: {test_precision:.4f}")
    print(f"✓ Recall Test: {test_recall:.4f}")
    print(f"✓ F1-Score Test: {test_f1:.4f}")

    print("\n✓ Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    run_id = mlflow.active_run().info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    
    print(f"\n✓ MLflow Run ID: {run_id}")
    print(f"✓ Experiment: Sentimen Analisis DANA")
    print(f"✓ Tracking UI: file:./mlruns")

print("\n✓ Training selesai.")