import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk"
}

def find_optimal_k(X_scaled, k_range=range(2, 9)):
    elbow = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        elbow.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    best_k = list(k_range)[np.argmax(silhouettes)]
    return elbow, silhouettes, best_k

def fit_kmeans(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels

def profile_segments(df, labels, feature_names):
    """Compute mean profile per cluster."""
    profiles = {}
    for seg_id in np.unique(labels):
        mask = labels == seg_id
        profile = {
            "count": int(mask.sum()),
            "pct": round(mask.sum() / len(labels) * 100, 1),
            "segment_name": SEGMENT_NAMES.get(seg_id, f"Segment {seg_id}"),
            "feature_means": {
                feat: round(float(df[feat].values[mask].mean()), 3)
                for feat in ['income','credit_score','employment_years','debt_to_income','loan_history_count','age']
            }
        }
        profiles[int(seg_id)] = profile
    return profiles

def save_results(profiles, silhouette_avg, elbow, silhouettes, output_path):
    results = {
        "silhouette_score": round(silhouette_avg, 4),
        "segment_profiles": profiles,
        "elbow_inertia": [round(float(x), 2) for x in elbow],
        "silhouette_by_k": [round(float(x), 4) for x in silhouettes]
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results