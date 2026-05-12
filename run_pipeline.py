"""Execute full customer segmentation pipeline: load -> engineer -> cluster -> classify -> report."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))

from src import data_loader, features, segment, classify

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def run():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE — UNDERWRITING")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (n=5000)...")
    df = data_loader.generate_customer_data(5000)
    print(f"    Rows: {len(df)}")
    print(f"    True segment distribution:\n{df['segment_label'].value_counts().sort_index().to_string()}")

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    df = features.build_features(df)
    feature_cols = features.get_feature_cols()
    print(f"    Features: {feature_cols}")

    X = df[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Clustering
    print("\n[3/5] Finding optimal K (Elbow + Silhouette)...")
    opt = segment.find_optimal_k(X_scaled, k_range=range(2, 8))
    best_k = opt["best_k"]
    print(f"    Optimal K: {best_k}  (silhouette={opt['best_silhouette']:.4f})")

    print("\n[3/5] Fitting KMeans (k=4)...")
    km, labels = segment.fit_kmeans(X_scaled, n_clusters=4)

    # Profile clusters
    profiles = segment.profile_segments(df, labels, feature_cols)
    df["predicted_cluster"] = labels
    df["segment_name"] = segment.assign_segment_names(df, labels)

    print("\n    Cluster profiles (mean values):")
    for _, row in profiles.iterrows():
        print(
            f"    Cluster {int(row['cluster'])}: "
            f"income={row['income']:.0f}, credit={row['credit_score']:.0f}, "
            f"DTI={row['debt_to_income']:.2f}, n={int(row['count'])}"
        )

    # 4. Classification
    print("\n[4/5] Training RandomForest classifier on cluster labels...")
    result = classify.train_classifier(X, labels)
    print(f"    Test accuracy: {result['accuracy']:.4f}")
    print(f"    Train/Test: {result['train_size']}/{result['test_size_n']}")

    top_features = sorted(
        result["feature_importance"].items(), key=lambda x: x[1], reverse=True
    )[:5]
    print("    Top 5 features by importance:")
    for feat, imp in top_features:
        print(f"      {feat}: {imp:.4f}")

    # 5. Save results
    print("\n[5/5] Saving reports...")
    report = {
        "optimal_k": best_k,
        "best_silhouette": opt["best_silhouette"],
        "silhouettes_by_k": opt["silhouettes"],
        "inertias_by_k": opt["inertias"],
        "cluster_profiles": profiles.to_dict(orient="records"),
        "segment_names": df["segment_name"].value_counts().to_dict(),
        "classification_accuracy": result["accuracy"],
        "classification_report": result["classification_report"],
        "feature_importance": result["feature_importance"],
        "n_samples": len(df),
        "features_used": feature_cols,
    }

    out_path = REPORTS_DIR / "segmentation_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"    Saved: {out_path}")

    # Summary print
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Samples:          5000")
    print(f"  Clusters (K):     4")
    print(f"  Best silhouette:  {opt['best_silhouette']:.4f}")
    print(f"  RF accuracy:      {result['accuracy']:.4f}")
    print(f"  Segments:         {list(df['segment_name'].value_counts().index)}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run()