"""End-to-end pipeline: data generation → feature engineering → clustering → classification."""
import json
import pandas as pd
from src.data_loader import generate_customer_data
from src.features import build_features, scale_features, get_feature_columns
from src.segment import run_segmentation, SEGMENT_NAMES
from src.classify import train_classifier, get_feature_importance, save_artifacts


def run_pipeline(n_samples=5000, n_clusters=4):
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE — Underwriting")
    print("=" * 60)

    # Step 1: Load/generate data
    print("\n[1/5] Generating synthetic customer data...")
    df = generate_customer_data(n_samples=n_samples)
    print(f"  → {len(df)} customers generated")
    print(f"  → Columns: {list(df.columns)}")

    # Step 2: Feature engineering
    print("\n[2/5] Engineering features...")
    df_feat = build_features(df)
    feature_cols = get_feature_columns()
    print(f"  → {len(feature_cols)} features created")

    # Step 3: Scale & cluster
    print("\n[3/5] Scaling features and running KMeans clustering...")
    X_scaled, scaler = scale_features(df_feat)
    df_labeled, seg_results, km = run_segmentation(X_scaled, df_feat, n_clusters=n_clusters)
    print(f"  → Silhouette score: {seg_results['silhouette_score']}")
    print(f"  → Best k (silhouette): {seg_results['optimal_k_analysis']['best_k']}")

    # Step 4: Train classifier
    print("\n[4/5] Training RandomForest classifier on cluster labels...")
    base_features = [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age", "home_ownership", "verified_income",
    ]
    clf, clf_results = train_classifier(df_labeled, base_features)
    print(f"  → Accuracy: {clf_results['accuracy']:.2%}")
    importance = get_feature_importance(clf, base_features)

    # Step 5: Save artifacts
    print("\n[5/5] Saving artifacts...")
    combined_results = {
        "dataset": {
            "n_samples": n_samples,
            "n_features": len(base_features),
        },
        "segmentation": {
            "silhouette_score": seg_results["silhouette_score"],
            "optimal_k": seg_results["optimal_k_analysis"]["best_k"],
            "segment_profiles": seg_results["segment_profiles"],
            "segment_names": SEGMENT_NAMES,
        },
        "classification": {
            "accuracy": clf_results["accuracy"],
            "n_train": clf_results["n_train"],
            "n_test": clf_results["n_test"],
            "feature_importance_top5": dict(list(importance.items())[:5]),
        },
    }

    save_artifacts(clf, clf_results, importance, path_prefix="reports/")

    with open("reports/segmentation_results.json", "w") as f:
        json.dump(combined_results, f, indent=2)
    print("  → reports/segmentation_results.json")
    print("  → reports/segment_classifier.joblib")

    # Print summary
    print("\n" + "=" * 60)
    print("SEGMENT SUMMARY")
    print("=" * 60)
    for seg_name, profile in seg_results["segment_profiles"].items():
        print(f"\n  [{seg_name}]")
        print(f"    Count: {profile['count']} ({profile['pct']}%)")
        print(f"    Avg Income: ${profile['income_mean']:,.0f}")
        print(f"    Avg Credit Score: {profile['credit_score_mean']}")
        print(f"    Avg Employment: {profile['employment_years_mean']} yrs")
        print(f"    Avg DTI: {profile['debt_to_income_mean']:.1%}")
        print(f"    Avg Loans: {profile['loan_history_count_mean']}")
        print(f"    Homeownership: {profile['home_ownership_rate']:.1%}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Model: RandomForestClassifier (200 trees, depth=10)")
    print(f"  Train/Test: {clf_results['n_train']} / {clf_results['n_test']}")
    print(f"  Accuracy: {clf_results['accuracy']:.2%}")
    print(f"\n  Top 5 Feature Importances:")
    for feat, imp in list(importance.items())[:5]:
        print(f"    {feat}: {imp:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return combined_results


if __name__ == "__main__":
    results = run_pipeline()