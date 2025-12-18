import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# konfigurasi mlflow lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Bank-Clustering")



# mengaktifkan autolog
mlflow.sklearn.autolog()

with mlflow.start_run():

    # memuat dataset preprocessing
    data = pd.read_csv("bank_transactions_preprocessed.csv")

    print("\n*** menampilkan data ***")
    print(data.head())

    print("\n*** info data ***")
    print(data.info())

    print("\n*** ringkasan data ***")
    print(data.describe())

    # memilih kolom numerik
    data_numeric = data.select_dtypes(include=["int64", "float64"])

    # mencari jumlah cluster terbaik menggunakan silhouette score
    silhouette_scores = {}

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data_numeric)
        score = silhouette_score(data_numeric, labels)
        silhouette_scores[k] = score

    print("\nSilhouette Score untuk setiap k:")
    for k, score in silhouette_scores.items():
        print(f"k = {k}: silhouette = {score:.4f}")

    # memilih k terbaik
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"\nJumlah cluster terbaik (best k): {best_k}")

    # melatih model final
    model = KMeans(
        n_clusters=best_k,
        random_state=42
    )

    labels = model.fit_predict(data_numeric)

    # evaluasi akhir
    sil_score = silhouette_score(data_numeric, labels)
    print(f"\nSilhouette Score (final model): {sil_score:.4f}")

    # manual logging
    mlflow.log_param("best_k", best_k)
    mlflow.log_metric("silhouette_score", sil_score)

    print("\nTraining dan evaluasi selesai.")
