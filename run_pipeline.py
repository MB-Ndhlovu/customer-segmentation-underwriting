"""Execute full customer segmentation pipeline."""

import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import generate_customer_data
from src.features import build_features
from src.segment import find_optimal_k, fit_kmeans, profile_segments, save_results
from src.classify import train_classifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def run():
    print("=" * 60)
    print("Customer Segmentation for Underwriting — Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (n=5000)...")
    df = generate_customer_data(n=5000, seed=42)
    print(f"  Rows: {len(df)}, Columns: {list(df.columns)}")

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    feat_df = build_features(df)
    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership_enc', 'verified_income',
        'rfm_monetary', 'behavioral_dti', 'stability_tenure_score',
    ]
    X = feat_df[feature_cols].values
    print(f"  Feature matrix shape: {X.shape}")

    # 3. Scale
    print("\n[3/5] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("  StandardScaler applied.")

    # 4. Clustering
    print("\n[4/5] Running KMeans clustering...")
    print("  Finding optimal k (2-7) via Silhouette analysis...")
    best_k, inertias, silhouettes = find_optimal_k(X_scaled, range(2, 8), verbose=True)
    print(f"\n  Optimal k by silhouette: {best_k} (score={max(silhouettes):.4f})")

    # Fit k=4 as requested
    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil = silhouette_score(X_scaled, labels)
    print(f"\n  Using k=4 (silhouette={sil:.4f})")
    counts = np.bincount(labels.astype(int))
    print(f"  Cluster distribution: {dict(enumerate(counts))}")

    profiles, _ = profile_segments(feat_df, labels, feature_cols)

    results = save_results(labels, profiles, sil, best_k,
                           output_path='reports/segmentation_results.json')

    # 5. Classification
    print("\n[5/5] Training RandomForest classifier on cluster labels...")
    clf, acc = train_classifier(X_scaled, labels)

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"  Best k (silhouette analysis): {best_k}")
    print(f"  Silhouette (k=4):             {sil:.4f}")
    print(f"  RF Accuracy:                  {acc:.4f}")
    print(f"  Results saved:                reports/segmentation_results.json")
    print(f"  Segment counts:               {results['segment_counts']}")

    return results, clf, acc, sil, best_k


if __name__ == '__main__':
    results, clf, acc, sil, best_k = run()