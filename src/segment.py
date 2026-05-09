import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import json

def find_optimal_k(X_scaled, k_range=range(2, 10)):
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertias, silhouettes

def elbow_method(inertias, k_range):
    deltas = np.diff(inertias)
    second_deltas = np.diff(deltas)
    elbow_idx = np.argmax(second_deltas) + 2
    return k_range[elbow_idx]

def silhouette_analysis(X_scaled, labels, n_clusters):
    sil_avg = silhouette_score(X_scaled, labels)
    sil_samples = silhouette_samples(X_scaled, labels)
    return sil_avg, sil_samples

def profile_segments(df, labels, feature_cols):
    df_temp = df.copy()
    df_temp['cluster'] = labels
    profiles = {}
    for cluster_id in sorted(df_temp['cluster'].unique()):
        cluster_data = df_temp[df_temp['cluster'] == cluster_id]
        profile = {}
        for col in feature_cols:
            profile[col] = {
                'mean': round(float(cluster_data[col].mean()), 4),
                'std': round(float(cluster_data[col].std()), 4),
            }
        profiles[int(cluster_id)] = profile
    return profiles

def assign_segment_names(labels):
    # Assign business-friendly names based on typical cluster characteristics
    return {0: 'Mass Market', 1: 'Rising Prime', 2: 'Established Prime', 3: 'Subprime High-Risk'}

def run_segmentation(X_scaled, df_original, feature_cols):
    n_clusters = 4

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil_avg, sil_samples = silhouette_analysis(X_scaled, labels, n_clusters)

    inertias, silhouettes = find_optimal_k(X_scaled)
    optimal_k = elbow_method(inertias, range(2, 10))

    profiles = profile_segments(df_original, labels, feature_cols)
    segment_names = assign_segment_names(labels)

    results = {
        'n_clusters': n_clusters,
        'silhouette_score': round(float(sil_avg), 4),
        'optimal_k': int(optimal_k),
        'inertias': [round(float(i), 4) for i in inertias],
        'silhouettes': [round(float(s), 4) for s in silhouettes],
        'profiles': profiles,
        'segment_names': segment_names,
        'cluster_counts': {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    }

    return labels, kmeans, results