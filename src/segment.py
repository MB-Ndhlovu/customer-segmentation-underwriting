"""
KMeans clustering with silhouette analysis and elbow method.
Profiles each resulting segment.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def cluster_customers(X, n_clusters=4, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(X)
    inertia = kmeans.inertia_
    return labels, kmeans, inertia


def find_optimal_k(X, k_range=range(2, 9)):
    """
    Compute inertias and silhouette scores for a range of k.
    Returns lists and saves elbow plot.
    """
    inertias, silhouettes = [], []
    for k in k_range:
        labels, _, inertia = cluster_customers(X, n_clusters=k)
        inertias.append(inertia)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        print(f"  k={k}  inertia={inertia:.0f}  silhouette={sil:.4f}")

    # Plot elbow + silhouette
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(list(k_range), inertias, 'b-o', label='Inertia')
    ax2.plot(list(k_range), silhouettes, 'g--x', label='Silhouette')
    ax1.set_xlabel('k'); ax1.set_ylabel('Inertia', color='b'); ax2.set_ylabel('Silhouette', color='g')
    plt.title('Elbow + Silhouette Analysis')
    plt.tight_layout()
    plt.savefig('/home/workspace/Projects/customer-segmentation-underwriting/reports/elbow_silhouette.png', dpi=150)
    plt.close()
    print("Saved elbow plot to reports/elbow_silhouette.png")

    return dict(zip(k_range, silhouettes))


def profile_segments(df, labels):
    """Profile each cluster with mean values."""
    df_temp = df.copy()
    df_temp['cluster'] = labels
    profiles = df_temp.groupby('cluster').mean(numeric_only=True)
    counts = df_temp['cluster'].value_counts().sort_index()
    print("\n=== Segment Profiles ===")
    for c in sorted(df_temp['cluster'].unique()):
        seg_names = {0: 'Mass Market', 1: 'Rising Prime', 2: 'Established Prime', 3: 'Subprime High-Risk'}
        print(f"\nCluster {c} ({seg_names.get(c, 'Unknown')}) — {counts[c]} customers")
        row = profiles.loc[c]
        for col, val in row.items():
            if col != 'cluster':
                print(f"  {col}: {val:.2f}")
    return profiles