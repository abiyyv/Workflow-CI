import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# AKTIFKAN AUTOLOG
mlflow.sklearn.autolog()

# ======================
# LOAD DATA
# ======================
data = pd.read_csv("bank_transactions_preprocessed.csv")

data_numeric = data.select_dtypes(include=["int64", "float64"])

# ======================
# SEARCH BEST K
# ======================
silhouette_scores = {}

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data_numeric)
    score = silhouette_score(data_numeric, labels)
    silhouette_scores[k] = score

best_k = max(silhouette_scores, key=silhouette_scores.get)

# ======================
# TRAIN FINAL MODEL
# ======================
final_model = KMeans(n_clusters=best_k, random_state=42)
labels = final_model.fit_predict(data_numeric)

final_sil = silhouette_score(data_numeric, labels)

# ======================
# MANUAL LOGGING (TAMBAHAN)
# ======================
mlflow.log_param("best_k", best_k)
mlflow.log_metric("final_silhouette_score", final_sil)

print("Training selesai, artifacts & model otomatis tercatat di MLflow")
