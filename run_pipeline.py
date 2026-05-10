"""Full pipeline orchestration."""

import json
import os
import pandas as pd

from src.data_loader import generate_customer_data
from src.features import build_features, scale_features
from src.segment import (
    find_optimal_k,
    fit_kmeans,
    profile_segments,
    assign_segment_names,
    silhouette_detail,
)
from src.classify import train_classifier, get_feature_importance


def run():
    print("=" * 60)
    print("  Customer Segmentation Pipeline — Underwriting")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (n=5000)...")
    df = generate_customer_data(n=5000)
    print(f"  Shape: {df.shape}")
    print(f"  True segment distribution:\n{df['segment_true'].value_counts().sort_index().to_string()}")

    # 2. Build features
    print("\n[2/5] Engineering features...")
    X_raw = build_features(df)
    print(f"  Feature count: {X_raw.shape[1]}")
    X, scaler = scale_features(X_raw)

    # 3. Clustering
    print("\n[3/5] Running KMeans clustering...")
    inertias, silhouettes = find_optimal_k(X, k_range=range(2, 9))
    print(f"  K=2 silhouette={silhouettes[0]:.3f}  K=3 silhouette={silhouettes[1]:.3f}  K=4 silhouette={silhouettes[2]:.3f}  K=5 silhouette={silhouettes[3]:.3f}")

    km, labels = fit_kmeans(X, n_clusters=4)
    sil_score, _ = silhouette_detail(X, labels)
    print(f"  Selected k=4 — Silhouette Score: {sil_score:.4f}")
    print(f"  Inertia: {km.inertia_:.2f}")

    # Profile clusters
    profiles = profile_segments(df, labels)
    seg_names = assign_segment_names(labels, profiles)

    # Map labels to named segments
    label_to_name = {i: seg_names[i] for i in range(4)}

    print("\n  Cluster profiles (mean):")
    display_cols = ["income", "credit_score", "employment_years", "debt_to_income", "loan_history_count"]
    print(profiles[display_cols].to_string())

    print("\n  Segment assignment:")
    for name in ["Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"]:
        inv_map = {v: k for k, v in label_to_name.items()}
        cl = inv_map[name]
        count = int((labels == cl).sum())
        print(f"  Cluster {cl} → {name}: {count} customers ({count / len(labels) * 100:.1f}%)")

    # 4. Classification
    print("\n[4/5] Training RandomForest classifier on cluster labels...")
    clf, acc, X_test, y_test, preds = train_classifier(X, labels)
    print(f"  Test Accuracy: {acc:.4f}")
    fi = get_feature_importance(clf, X.columns.tolist())
    print("\n  Feature Importance (Top 5):")
    for _, row in fi.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")

    # 5. Save report
    print("\n[5/5] Saving results...")
    os.makedirs("reports", exist_ok=True)
    results = {
        "silhouette_score": round(sil_score, 4),
        "inertia": round(float(km.inertia_), 2),
        "test_accuracy": round(acc, 4),
        "n_clusters": 4,
        "n_features": int(X_raw.shape[1]),
        "n_samples": int(len(df)),
        "cluster_profiles": profiles.to_dict(),
        "feature_importance": fi.to_dict(orient="records"),
        "segment_labels": label_to_name,
        "segment_counts": {label_to_name[i]: int((labels == i).sum()) for i in range(4)},
    }
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved: reports/segmentation_results.json")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    results = run()