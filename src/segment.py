"""KMeans clustering with Elbow + Silhouette analysis, segment profiling."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import json


SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk',
}

SEGMENT_NAMES_BY_INCOME_RANK = [
    'Subprime High-Risk',  # lowest income
    'Mass Market',
    'Rising Prime',
    'Established Prime',   # highest income
]


def find_optimal_k(X_scaled: np.ndarray, k_range: range, verbose: bool = True):
    """Run Elbow and Silhouette analysis across a range of K."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        if verbose:
            print(f"  k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.4f}")

    best_k = k_range[np.argmax(silhouettes)]
    if verbose:
        print(f"\nBest k by silhouette: {best_k}")
    return best_k, inertias, silhouettes


def fit_kmeans(X_scaled: np.ndarray, n_clusters: int = 4):
    """Fit final KMeans model."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return km, labels


def profile_segments(df: pd.DataFrame, labels: np.ndarray, feature_cols: list) -> pd.DataFrame:
    """Print and return segment profile statistics. Clusters remapped by income rank."""
    df = df.copy()
    df['segment'] = labels

    numeric_cols = ['income', 'credit_score', 'employment_years',
                    'debt_to_income', 'loan_history_count', 'age', 'verified_income']

    # Profile by cluster
    raw_profiles = df.groupby('segment')[numeric_cols].mean()

    # Remap clusters to income rank (0=lowest income -> Subprime High-Risk)
    income_means = raw_profiles['income']
    rank_order = income_means.sort_values().index.tolist()  # [lowest, ..., highest]

    name_map = {rank_order[i]: SEGMENT_NAMES_BY_INCOME_RANK[i] for i in range(len(rank_order))}

    # Build remapped profiles with business names
    remapped_profiles = raw_profiles.rename(index=name_map)
    remapped_profiles.index = [f"{name}" for name in remapped_profiles.index]

    print("\n=== Segment Profiles ===")
    print(remapped_profiles.to_string())

    return remapped_profiles, name_map  # name_map: cluster_id -> business_name


def save_results(labels: np.ndarray, profiles: pd.DataFrame,
                 silhouette: float, best_k: int,
                 output_path: str = 'reports/segmentation_results.json'):
    """Save segmentation results to JSON."""
    counts = np.bincount(labels.astype(int))
    segment_counts = {str(i): int(c) for i, c in enumerate(counts)}

    results = {
        'best_k': best_k,
        'silhouette_score': round(silhouette, 4),
        'segment_counts': segment_counts,
        'profiles': {name: {col: round(float(v), 2) for col, v in row.items()}
                     for name, row in profiles.iterrows()},
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import build_features

    df = generate_customer_data()
    feat_df = build_features(df)

    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership_enc', 'verified_income',
        'rfm_monetary', 'behavioral_dti', 'stability_tenure_score',
    ]

    X = feat_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, _, silhouettes = find_optimal_k(X_scaled, range(2, 8))
    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil = silhouette_score(X_scaled, labels)

    profiles, _ = profile_segments(feat_df, labels, feature_cols)
    results = save_results(labels, profiles, sil, best_k)