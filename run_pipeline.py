#!/usr/bin/env python3
"""End-to-end Customer Segmentation Pipeline for Underwriting."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_customer_data
from src.features import engineer_features, get_feature_names
from src.segment import find_optimal_k, fit_kmeans, profile_segments, compute_silhouette_details
from src.classify import train_segment_classifier, save_artifacts


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Generating synthetic customer data (5000 rows)...")
    df = generate_customer_data(5000)
    print(f"    Shape: {df.shape}")
    print(f"    True segment distribution:\n{df['true_segment'].value_counts().sort_index().to_string()}")

    # 2. Feature engineering
    print("\n[2] Engineering features (RFM + behavioral + stability)...")
    features = engineer_features(df)
    feature_names = get_feature_names()
    print(f"    Features ({len(feature_names)}): {feature_names}")

    # Scale for clustering
    scaler_seg = StandardScaler()
    X_scaled = scaler_seg.fit_transform(features[feature_names])

    # 3. Find optimal k (2–8 range)
    print("\n[3] Running Elbow + Silhouette analysis (k=2..8)...")
    k_results = find_optimal_k(X_scaled, range(2, 9))
    print(f"    Optimal k = {k_results['best_k']}  (best silhouette)")
    print(f"    Silhouette scores: {dict(zip(k_results['k_values'], [round(s,4) for s in k_results['silhouettes']]))}")

    # 4. Fit KMeans with k=4
    n_clusters = 4
    print(f"\n[4] Fitting KMeans (k={n_clusters})...")
    km, labels = fit_kmeans(X_scaled, n_clusters=n_clusters)
    sil_details = compute_silhouette_details(X_scaled, labels)
    print(f"    Silhouette average: {sil_details['silhouette_avg']}")
    print(f"    Inertia: {km.inertia_:.2f}")
    print(f"    Cluster distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"      Cluster {u}: {c} ({c/len(labels)*100:.1f}%)")

    # 5. Profile segments
    print("\n[5] Profiling segments...")
    profiles = profile_segments(features, labels, feature_names)
    for cid, prof in profiles.items():
        print(f"    [{cid}] {prof['name']} — n={prof['count']} ({prof['pct']}%)")
        print(f"        Top centroid features:")
        sorted_feats = sorted(prof["centroid_features"].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        for f, v in sorted_feats:
            print(f"          {f}: {v}")

    # 6. Train supervised classifier
    print("\n[6] Training RandomForest classifier on cluster labels...")
    base_features = ["income", "credit_score", "employment_years", "debt_to_income",
                     "loan_history_count", "age", "home_ownership", "verified_income"]
    clf, clf_scaler, clf_metrics = train_segment_classifier(
        df, pd.Series(labels), base_features
    )
    print(f"    Accuracy: {clf_metrics['accuracy']}")
    print(f"    F1 (weighted): {clf_metrics['f1_weighted']}")
    print("    Feature importances:")
    sorted_fi = sorted(clf_metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True)
    for f, v in sorted_fi:
        print(f"      {f}: {v}")

    # 7. Save artifacts
    output_path = "reports/segmentation_results.json"
    save_artifacts(clf, clf_scaler, clf_metrics, profiles, output_path)
    print(f"\n[7] Artifacts saved to {output_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return {
        "profiles": profiles,
        "clf_metrics": clf_metrics,
        "silhouette_avg": sil_details["silhouette_avg"],
        "optimal_k": k_results["best_k"],
        "n_clusters": n_clusters,
    }


if __name__ == "__main__":
    results = main()