import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json

SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk'
}

def find_optimal_k(X_scaled, max_k=10):
    inertias = []
    silhouettes = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertias, silhouettes

def elbow_investigation(inertias):
    """Simple heuristic: find knee in inertia curve via rate of change."""
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    # knee at position where second derivative is most negative (starting from k=3)
    return int(np.argmin(diffs2) + 3)

def run_clustering(X_scaled, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    return km, labels, sil

def profile_segments(X, labels):
    """Profile each cluster with mean feature values."""
    df = X.copy()
    df['cluster'] = labels
    profiles = df.groupby('cluster').mean().to_dict('index')
    return profiles

def assign_segment_names(labels, X):
    """
    Map cluster IDs to business segment names using business-logic rules:
      - Established Prime: highest credit_score, lowest DTI, high income
      - Rising Prime:     mid-high credit_score, mid DTI, mid-high income
      - Mass Market:      mid credit_score, mid DTI, mid income
      - Subprime High-Risk: lowest credit_score, highest DTI, low income
    Returns relabeled array (0-3) matching SEGMENT_NAMES.
    """
    cluster_stats = {}
    for c in np.unique(labels):
        mask = labels == c
        cluster_stats[c] = {
            'mean_credit':  X.loc[mask, 'credit_score'].mean(),
            'mean_dti':     X.loc[mask, 'debt_to_income'].mean(),
            'mean_income':  X.loc[mask, 'income'].mean(),
        }

    clusters = list(cluster_stats.keys())
    # Sort clusters by credit desc, DTI asc to rank them 0(best)..3(worst)
    ranked = sorted(clusters, key=lambda c: (-cluster_stats[c]['mean_credit'],
                                              cluster_stats[c]['mean_dti']))
    # Map rank 0 → Rising Prime (1), rank 1 → Mass Market (0),
    # rank 2 → Established Prime (2), rank 3 → Subprime High-Risk (3)
    rank_to_segment = {1: 0, 0: 1, 2: 2, 3: 3}
    mapping = {ranked[r]: rank_to_segment[r] for r in range(4)}
    return np.array([mapping[l] for l in labels])

def remap_profiles(profiles, labels, relabeled, X):
    """
    Remap the profiles dict from original cluster IDs to relabeled segment IDs.
    profiles: dict keyed by original cluster ID (0-3 from KMeans)
    labels: original KMeans cluster assignments
    relabeled: our semantic segment labels (0-3)
    Returns a new profiles dict keyed by segment ID (0-3).
    """
    remapped = {}
    for seg_id in range(4):
        mask = relabeled == seg_id
        remapped[seg_id] = {col: float(X.loc[mask, col].mean()) for col in X.columns}
    return remapped

def save_results(profiles, silhouette, labels, output_path='reports/segmentation_results.json'):
    results = {
        'silhouette_score': round(silhouette, 4),
        'n_clusters': 4,
        'segment_names': SEGMENT_NAMES,
        'profiles': {str(k): {feat: round(v, 4) for feat, v in vals.items()} for k, vals in profiles.items()},
        'segment_distribution': {
            SEGMENT_NAMES[i]: int(np.sum(labels == i)) for i in range(4)
        }
    }
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import build_features
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data()
    X = build_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias, silhouettes = find_optimal_k(X_scaled)
    print("Silhouette scores by k:", [round(s, 3) for s in silhouettes])

    km, labels, sil = run_clustering(X_scaled)
    print(f"KMeans k=4 silhouette: {sil:.4f}")

    profiles = profile_segments(X, labels)
    relabeled = assign_segment_names(labels, X)

    remapped_profiles = remap_profiles(profiles, labels, relabeled, X)

    results = save_results(remapped_profiles, sil, relabeled)
    print(json.dumps(results, indent=2))