from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

def find_optimal_k(X_scaled, k_range=range(2, 9)):
    elbow = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        elbow.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    best_k = list(k_range)[np.argmax(silhouettes)]
    return best_k, silhouettes, elbow

def fit_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels

def profile_segments(df, labels):
    df_out = df.copy()
    df_out["cluster"] = labels
    profiles = df_out.groupby("cluster").mean(numeric_only=True)
    return profiles

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from features import compute_features
    df = generate_synthetic_data()
    X = compute_features(df)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    best_k, sils, elbows = find_optimal_k(Xs)
    print(f"Best k={best_k}, silhouettes={sils}")
    km, lbls = fit_kmeans(Xs, 4)
    print("Cluster distribution:", np.bincount(lbls))