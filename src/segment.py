"""KMeans clustering, silhouette analysis, elbow method, segment profiling."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import json


SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk',
}


def find_optimal_k(X_scaled: np.ndarray, k_range: range, random_state: int = 42):
    """Compute inertia and silhouette scores for k in k_range."""
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertias, silhouettes


def plot_elbow_silhouette(k_range, inertias, silhouettes, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inertias, 'bo-')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')

    axes[1].plot(list(k_range), silhouettes, 'go-')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].axvline(4, color='r', linestyle='--', label='k=4')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{out_dir}/elbow_silhouette.png', dpi=150)
    plt.close()


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4, random_state: int = 42):
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(labels: np.ndarray, X_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Profile each segment with mean feature values."""
    df_out = X_df[feature_cols].copy()
    df_out['cluster'] = labels

    profile = df_out.groupby('cluster')[feature_cols].mean().round(3)
    profile['count'] = df_out.groupby('cluster').size()

    # Map cluster IDs to segment names by analyzing centroids
    # Sort by first feature (income proxy) to assign meaningful names
    first_feat = feature_cols[0]
    income_means = profile[first_feat].sort_values()
    name_map = {}
    for i, cluster_id in enumerate(income_means.index):
        name_map[cluster_id] = SEGMENT_NAMES[i]

    profile['segment_name'] = profile.index.map(name_map)
    profile = profile.reset_index().rename(columns={'cluster': 'cluster_id'})
    return profile


def save_results(profile: pd.DataFrame, silhouette_avg: float,
                 out_path: str, centroids: np.ndarray):
    results = {
        'silhouette_score': round(silhouette_avg, 4),
        'segments': [],
    }
    for _, row in profile.iterrows():
        results['segments'].append({
            'cluster_id':    int(row['cluster_id']),
            'segment_name':  row['segment_name'],
            'count':         int(row['count']),
            'pct':           round(int(row['count']) / int(profile['count'].sum()) * 100, 1),
            'centroid':      centroids[int(row['cluster_id'])].tolist(),
        })

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)


def run_segmentation(X_df: pd.DataFrame, feature_cols: list, out_dir: str = 'reports'):
    # Scale features
    scaler = StandardScaler()
    X = X_df[feature_cols].values
    X_scaled = scaler.fit_transform(X)

    # Find optimal k
    k_range = range(2, 11)
    inertias, silhouettes = find_optimal_k(X_scaled, k_range)
    plot_elbow_silhouette(k_range, inertias, silhouettes, out_dir)

    best_k = 4
    silhouette_avg = silhouettes[best_k - 2]

    # Fit final model
    km, labels = fit_kmeans(X_scaled, n_clusters=best_k)

    # Profile
    profile = profile_segments(labels, X_df, feature_cols)

    print(f"\n=== KMeans Segmentation (k={best_k}) ===")
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"\nSegment Profile:\n{profile.to_string(index=False)}")

    # Save results
    save_results(profile, silhouette_avg, f'{out_dir}/segmentation_results.json', km.cluster_centers_)

    return km, labels, scaler, profile