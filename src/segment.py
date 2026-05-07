import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def find_optimal_k(X_scaled, k_range=range(2, 9)):
    elbow = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        elbow.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return elbow, silhouettes

def cluster(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_
    sil = silhouette_score(X_scaled, labels)
    return labels, centroids, sil

def profile_segments(df, labels, feature_cols):
    df = df.copy()
    df['segment_label'] = labels
    profiles = df.groupby('segment_label')[feature_cols].mean()
    return profiles

if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import compute_features
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data()
    X = compute_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    elbow, silhouettes = find_optimal_k(X_scaled)
    print("Silhouette scores by k:", dict(zip(range(2, 9), silhouettes)))

    labels, centroids, sil = cluster(X_scaled, n_clusters=4)
    print(f"\nSilhouette score (k=4): {sil:.4f}")
    print("\nSegment profiles:")
    print(profile_segments(X, labels, X.columns.tolist()))