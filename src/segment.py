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

def find_optimal_k(X_scaled, k_range=range(2, 10)):
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
    return km.fit_predict(X_scaled)

def profile_segments(df, labels, segment_names=None):
    """Profile each cluster with key statistics."""
    if segment_names is None:
        segment_names = SEGMENT_NAMES
    profiles = {}
    for seg_id in sorted(np.unique(labels)):
        mask = labels == seg_id
        seg_df = df[mask]
        profiles[int(seg_id)] = {
            'name': segment_names.get(seg_id, f'Segment {seg_id}'),
            'count': int(mask.sum()),
            'pct': float(mask.sum() / len(labels) * 100),
            'mean_income': float(seg_df['income'].mean()),
            'mean_credit_score': float(seg_df['credit_score'].mean()),
            'mean_debt_to_income': float(seg_df['debt_to_income'].mean()),
            'mean_employment_years': float(seg_df['employment_years'].mean()),
            'mean_loan_history_count': float(seg_df['loan_history_count'].mean()),
            'mean_age': float(seg_df['age'].mean()),
            'home_ownership_rate': float(seg_df['home_ownership'].mean()),
            'verified_income_rate': float(seg_df['verified_income'].mean()),
        }
    return profiles

def remap_to_segment_names(df, labels):
    """Try to map cluster IDs to business segment names based on characteristics.
    Business logic ordering (worst to best risk):
      3=Subprime High-Risk, 0=Mass Market, 1=Rising Prime, 2=Established Prime
    """
    seg_df = df.copy()
    seg_df['label'] = labels
    means = seg_df.groupby('label')[['credit_score', 'income']].mean()

    # Sort clusters by credit_score (asc), then income (asc)
    sorted_labels = means.sort_values(['credit_score', 'income']).index.tolist()

    # Map to business segment IDs
    # sorted order is worst risk to best risk
    business_order = [3, 0, 1, 2]  # Subprime, Mass Market, Rising Prime, Established Prime
    mapping = {old: new for new, old in enumerate(business_order) if old in sorted_labels}

    # Handle case where not all 4 clusters present
    remaining = [l for l in sorted_labels if l not in mapping]
    for i, l in enumerate(remaining):
        mapping[l] = list(mapping.values())[i] if i < len(mapping) else i

    remapped = np.array([mapping[l] for l in labels])
    return remapped

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    from features import build_features
    df = generate_synthetic_data()
    X = build_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    labels = cluster(X_scaled)
    labels = remap_to_segment_names(df, labels)
    profiles = profile_segments(df, labels)
    print(json.dumps(profiles, indent=2))