"""Full pipeline orchestration."""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_customer_data
from src.features import build_features, add_original_features
from src.segment import find_optimal_k, cluster, profile_segments, save_results
from src.classify import train_classifier

SEGMENT_NAMES = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (5000 rows)...")
    df = generate_customer_data(5000)
    print(f"  Shape: {df.shape}")
    print(df.describe().round(2).to_string())

    # 2. Engineer features
    print("\n[2/5] Building engineered features...")
    X_eng = build_features(df)
    X = add_original_features(X_eng, df)
    print(f"  Total features: {X.shape[1]}")
    print(f"  Features: {list(X.columns)}")

    # 3. Scale and find optimal k
    print("\n[3/5] Scaling features and finding optimal cluster count...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    optimal_k, metrics = find_optimal_k(X_scaled, range(2, 10))
    print(f"  Optimal k (by silhouette): {optimal_k}")
    print(f"  Silhouette scores: {dict(zip(metrics['k_values'], [round(s,4) for s in metrics['silhouettes']]))}")

    # 4. Cluster
    print(f"\n[4/5] Running KMeans (k={optimal_k})...")
    labels = cluster(X_scaled, n_clusters=optimal_k)
    print(f"  Cluster distribution:")
    for seg_id in range(optimal_k):
        count = int(np.sum(labels == seg_id))
        name = SEGMENT_NAMES.get(seg_id, f"Cluster {seg_id}")
        print(f"    Cluster {seg_id} ({name}): {count} ({count/len(labels)*100:.1f}%)")

    # Profile
    profiles = profile_segments(X, labels, df)
    print("\n  Cluster profiles (mean values):")
    print(profiles.round(2).to_string())

    # 5. Train classifier
    print("\n[5/5] Training RandomForest classifier on cluster labels...")
    clf, clf_scaler = train_classifier(X, labels)

    # Save results
    save_results(profiles, metrics, "reports/segmentation_results.json")

    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nBusiness Segments:")
    for seg_id, name in SEGMENT_NAMES.items():
        count = int(np.sum(labels == seg_id))
        avg_income = df.loc[labels == seg_id, "income"].mean()
        avg_credit = df.loc[labels == seg_id, "credit_score"].mean()
        avg_dti = df.loc[labels == seg_id, "debt_to_income"].mean()
        print(f"  [{seg_id}] {name:22s} | n={count:5d} | avg_income=${avg_income:,.0f} | avg_credit={avg_credit:.0f} | avg_dti={avg_dti:.3f}")

    print(f"\nOutput: reports/segmentation_results.json")

    # Return summary dict for Telegram
    return {
        "segments": SEGMENT_NAMES,
        "cluster_counts": {str(k): int(np.sum(labels == k)) for k in range(optimal_k)},
        "silhouette": float(max(metrics["silhouettes"])),
        "cv_accuracy": None,  # will fill from classify output
        "profiles": profiles.to_dict(),
    }


if __name__ == "__main__":
    results = main()