"""Execute full customer segmentation pipeline."""

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.data_loader import load_customer_data
from src.features import build_features, get_feature_columns
from src.segment import find_optimal_k, fit_kmeans, profile_segments
from src.classify import train_classifier


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = load_customer_data(n_rows=5000)
    print(f"  Generated {len(df)} rows")

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    df = build_features(df)
    feat_cols = get_feature_columns()
    print(f"  Feature columns: {feat_cols}")

    X = df[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Clustering
    print("\n[3/5] KMeans clustering (k=4)...")
    k_opt, diag = find_optimal_k(X_scaled, range(2, 11))
    print(f"  Optimal k detected: {k_opt}")

    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil = silhouette_score(X_scaled, labels)
    print(f"  Silhouette Score: {sil:.4f}")

    # Profile clusters
    profile = profile_segments(df, labels, feat_cols)
    print("\n  Cluster Profiles (mean values):")
    print(profile.to_string())

    # Remap clusters to business-friendly segment names
    # Determine mapping by looking at centroids
    centroids = km.cluster_centers_
    # centroid order in scaled space correlates with segment tiers
    # Find cluster order: most negative = lowest income/credit, most positive = highest
    cluster_means = df.groupby(labels)[feat_cols].mean()
    # Map cluster indices to business segment order
    # 0=Mass Market, 1=Rising Prime, 2=Established Prime, 3=Subprime High-Risk
    # Order by: credit_score + income descending = best to worst
    tier_order = cluster_means.sort_values(
        by=["credit_score", "income"], ascending=False
    ).index.tolist()
    # Map: best cluster → Established Prime, 2nd best → Rising Prime, ...
    reverse_tier = {v: i for i, v in enumerate(tier_order)}
    segment_labels = np.array([reverse_tier[c] for c in labels])
    df["segment_label"] = segment_labels

    # 4. Classification
    print("\n[4/5] Training RandomForest classifier...")
    clf_results = train_classifier(X, segment_labels, feat_cols)
    print(f"  Test Accuracy: {clf_results['accuracy']:.2%}")
    print(f"  Feature Importances:")
    for feat, imp in sorted(
        clf_results["feature_importances"].items(), key=lambda x: -x[1]
    ):
        print(f"    {feat}: {imp}")

    # 5. Save results
    print("\n[5/5] Saving results...")
    results = {
        "n_customers": int(len(df)),
        "n_features": len(feat_cols),
        "silhouette_score": round(sil, 4),
        "optimal_k": k_opt,
        "segment_profiles": profile.round(4).to_dict(),
        "classifier_accuracy": clf_results["accuracy"],
        "feature_importances": clf_results["feature_importances"],
        "confusion_matrix": clf_results["confusion_matrix"],
        "segments": SEGMENT_NAMES,
    }

    import os
    os.makedirs("/home/workspace/Projects/customer-segmentation-underwriting/reports", exist_ok=True)
    out_path = "/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved → {out_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()