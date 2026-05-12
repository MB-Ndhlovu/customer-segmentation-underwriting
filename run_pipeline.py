"""Execute full customer segmentation pipeline."""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_customer_data
from src.features import build_features, FEATURE_COLS
from src.segment import find_optimal_k, fit_kmeans, profile_segments, remap_clusters
from src.classify import train_classifier, save_classifier


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE — UNDERWRITING")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (n=5000)...")
    df = generate_customer_data(5000)
    print(f"      Shape: {df.shape}")

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    df_fe = build_features(df)
    print(f"      Features: {len(FEATURE_COLS)} input + {len(df_fe.columns) - len(df.columns)} engineered")

    # 3. Clustering
    print("\n[3/5] Running Elbow + Silhouette analysis (k=2..8)...")
    X = df_fe[FEATURE_COLS]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    analysis = find_optimal_k(X_scaled, range(2, 9))
    best_k_idx = int(np.argmax(analysis["silhouettes"]))
    best_k = analysis["k_values"][best_k_idx]
    best_sil = analysis["silhouettes"][best_k_idx]
    print(f"      Optimal k={best_k}  |  Silhouette={best_sil:.4f}")

    print(f"\n      Fitting KMeans (k=4)...")
    labels, centroids, sil = fit_kmeans(X_scaled, n_clusters=4)
    df_fe["segment_label"] = labels

    # 4. Profile segments
    print("\n[4/5] Profiling segments...")
    profiles = profile_segments(df_fe, labels)
    mapping = remap_clusters(profiles)

    SEGMENT_NAMES_INV = {v: k for k, v in {
        0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"
    }.items()}

    for seg_id, name in mapping.items():
        row = profiles[profiles["segment_label"] == seg_id].iloc[0]
        print(f"\n      Segment {seg_id}: {name}")
        print(f"        n={int(row['count'])}  |  income={row['income']:.0f}  |  credit_score={row['credit_score']:.0f}")
        print(f"        DTI={row['debt_to_income']:.3f}  |  emp_years={row['employment_years']:.1f}  |  age={row['age']:.0f}")

    # 5. Train classifier
    print("\n[5/5] Training RandomForestClassifier on cluster labels...")
    result = train_classifier(df_fe, labels)
    print(f"      Test Accuracy: {result['accuracy']:.4f}")
    print("\n      Feature Importance:")
    for feat, imp in result["feature_importance"][:5]:
        print(f"        {feat}: {imp:.4f}")

    # Save artifacts
    save_classifier(result["model"], "models/segment_classifier.joblib")
    scaler_path = "models/scaler.joblib"
    import joblib
    joblib.dump(scaler, scaler_path)

    # Build results JSON
    results = {
        "n_customers": int(len(df)),
        "n_features": len(FEATURE_COLS),
        "optimal_k": best_k,
        "silhouette_score": round(sil, 4),
        "elbow_k_silhouettes": {str(k): round(s, 4) for k, s in zip(analysis["k_values"], analysis["silhouettes"])},
        "classifier_accuracy": round(result["accuracy"], 4),
        "segments": {}
    }

    for seg_id, name in mapping.items():
        row = profiles[profiles["segment_label"] == seg_id].iloc[0]
        results["segments"][name] = {
            "cluster_id": int(seg_id),
            "count": int(row["count"]),
            "mean_income": round(float(row["income"]), 2),
            "mean_credit_score": round(float(row["credit_score"]), 2),
            "mean_dti": round(float(row["debt_to_income"]), 4),
            "mean_employment_years": round(float(row["employment_years"]), 2),
            "mean_age": round(float(row["age"]), 1),
            "pct_homeowners": round(float(row["home_ownership"]), 4),
            "pct_verified_income": round(float(row["verified_income"]), 4),
        }

    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"  reports/segmentation_results.json")
    print(f"  models/segment_classifier.joblib")
    print(f"  models/scaler.joblib")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    results = main()