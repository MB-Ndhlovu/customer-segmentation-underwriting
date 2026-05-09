"""
Customer Segmentation Pipeline — Full Execution
=================================================
Loads data → engineers features → clusters → classifies → profiles → saves artifacts.
"""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_customer_data, get_feature_columns, get_segment_names
from src.features import build_features, get_feature_names
from src.segment import find_optimal_k, fit_kmeans, profile_segments, save_results
from src.classify import train_classifier, cross_validate


def run():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading synthetic customer data (5000 rows)...")
    df = generate_customer_data(n=5000, seed=42)
    print(f"    Data shape: {df.shape}")
    print(f"    Features: {get_feature_columns()}")
    seg_names = get_segment_names()

    # 2. Engineer features
    print("\n[2/6] Engineering features...")
    features_df = build_features(df)
    feature_names = get_feature_names()
    print(f"    Engineered {len(feature_names)} features: {feature_names}")

    # 3. Prepare for clustering
    print("\n[3/6] Scaling features and running cluster analysis...")
    raw_features = df[get_feature_columns()].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_features)

    # Elbow + silhouette analysis (advisory only — we target 4 segments)
    search_results = find_optimal_k(X_scaled, k_range=range(2, 10))
    silhouette_map = dict(zip(search_results["k"], search_results["silhouette"]))
    print(f"    Silhouette scores by k: { {k: round(v,4) for k,v in silhouette_map.items()} }")

    # Target 4 segments as required for underwriting
    TARGET_K = 4
    print(f"    Targeting k={TARGET_K} segments (per business requirement)")

    # 4. Fit KMeans with k=4
    print(f"\n[4/6] Fitting KMeans with k={TARGET_K}...")
    km, labels = fit_kmeans(X_scaled, n_clusters=TARGET_K, random_state=42)
    sil_score = float(silhouette_map[TARGET_K])
    print(f"    Silhouette score @ k={TARGET_K}: {sil_score:.4f}")
    print(f"    Inertia: {km.inertia_:.2f}")

    # 5. Profile segments
    print("\n[5/6] Profiling segments...")
    profiles = profile_segments(df, labels, seg_names)

    print("\n    Segment Profiles:")
    print("    " + "-" * 56)
    for p in profiles:
        print(f"    [{p['segment_id']}] {p['segment_name']}")
        print(f"        Size: {p['size']} ({p['pct']}%) | "
              f"Income: ${p['mean_income']:,.0f} | "
              f"Credit: {p['mean_credit_score']:.0f} | "
              f"DTI: {p['mean_debt_to_income']:.2f} | "
              f"Employment: {p['mean_employment_years']:.1f} yrs")
        print(f"        Home ownership: {p['home_ownership_rate']:.1%} | "
              f"Verified income: {p['verified_income_rate']:.1%}")

    # 6. Train classifier
    print("\n[6/6] Training RandomForest classifier...")
    clf, clf_metrics, imp = train_classifier(X_scaled, df["segment_label"].values)
    print(f"    Accuracy: {clf_metrics['accuracy']:.4f}")
    print(f"    Train: {clf_metrics['n_train']} | Test: {clf_metrics['n_test']}")

    cv_results = cross_validate(X_scaled, df["segment_label"].values)
    print(f"    5-fold CV: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")

    # Save results
    print("\n[SAVING] Writing artifacts...")
    save_results(profiles, TARGET_K, [silhouette_map[k] for k in search_results["k"]], "reports/segmentation_results.json")

    # Summary JSON
    summary = {
        "n_samples": 5000,
        "target_k": TARGET_K,
        "silhouette_at_target_k": round(sil_score, 4),
        "advisory_silhouette_map": {str(k): round(float(v), 4) for k, v in silhouette_map.items()},
        "segments": {p["segment_name"]: p for p in profiles},
        "classifier_accuracy": round(clf_metrics["accuracy"], 4),
        "classifier_cv": {k: round(float(v), 4) if isinstance(v, float) else v for k, v in cv_results.items()}
    }

    with open("reports/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  reports/segmentation_results.json")
    print(f"  reports/pipeline_summary.json")

    # Return summary for Telegram message
    return summary


if __name__ == "__main__":
    summary = run()
    print("\n--- FULL SUMMARY ---")
    print(json.dumps(summary, indent=2))