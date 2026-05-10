"""End-to-end pipeline: generate data → engineer features → cluster → classify → report."""

import json
import pandas as pd
from src import data_loader, features, segment, classify

REPORT_PATH = "reports/segmentation_results.json"
FEATURE_COLS = [
    "income",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "loan_history_count",
    "age",
    "home_ownership",
    "verified_income",
]


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — PIPELINE")
    print("=" * 60)

    # 1. Load / generate data
    print("\n[1/5] Generating synthetic customer dataset (5000 rows)...")
    df = data_loader.generate(n_total=5000, seed=42)
    print(f"  Rows: {len(df)}")
    print(f"  True segment distribution:\n{df['segment_label'].value_counts().sort_index().to_string()}")

    # 2. Engineer features
    print("\n[2/5] Engineering features (RFM + behavioral + stability)...")
    df_feat = features.derive_features(df)
    X = features.get_feature_matrix(df_feat)
    X_scaled, scaler = features.scale_features(X)
    print(f"  Engineered features: {list(X.columns)}")

    # 3. Clustering
    print("\n[3/5] Running KMeans clustering (k=4)...")
    df_labeled, labels, km_scaler, km_model, seg_results = segment.run_clustering(
        df_feat, FEATURE_COLS, k=4, random_state=42
    )
    print(f"  Silhouette Score: {seg_results['silhouette_avg']}")
    print(f"  Cluster sizes: {seg_results['cluster_sizes']}")
    print(f"  Cluster → Segment mapping:")
    for cluster_id, name in df_labeled[["cluster", "segment_name"]].drop_duplicates().values:
        print(f"    Cluster {cluster_id} → {name}")

    # 4. Supervised classifier
    print("\n[4/5] Training RandomForestClassifier on cluster labels...")
    clf, clf_results = classify.train_classifier(df_labeled, FEATURE_COLS)
    print(f"  Train Accuracy: {clf_results['train_accuracy']:.2%}")
    print(f"  Test Accuracy:  {clf_results['test_accuracy']:.2%}")
    print(f"  Feature Importances:")
    for feat, imp in sorted(clf_results["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"    {feat}: {imp:.4f}")

    # 5. Build segment profiles
    print("\n[5/5] Building segment profiles...")
    profiles = {}
    segment_mapping = df_labeled[["cluster", "segment_name"]].drop_duplicates().set_index("cluster")["segment_name"].to_dict()
    for cluster_id in sorted(df_labeled["cluster"].unique()):
        sub = df_labeled[df_labeled["cluster"] == cluster_id]
        segment_name = segment_mapping.get(cluster_id, f"Cluster {cluster_id}")
        profiles[segment_name] = {
            "n": int(len(sub)),
            "pct": round(len(sub) / len(df_labeled) * 100, 1),
            "income_mean": round(float(sub["income"].mean()), 0),
            "credit_score_mean": round(float(sub["credit_score"].mean()), 1),
            "employment_years_mean": round(float(sub["employment_years"].mean()), 2),
            "debt_to_income_mean": round(float(sub["debt_to_income"].mean()), 4),
            "loan_history_count_mean": round(float(sub["loan_history_count"].mean()), 2),
            "age_mean": round(float(sub["age"].mean()), 1),
            "home_ownership_rate": round(float(sub["home_ownership"].mean()), 4),
            "verified_income_rate": round(float(sub["verified_income"].mean()), 4),
        }
        print(f"  {segment_name}: n={profiles[segment_name]['n']}, "
              f"income=${profiles[segment_name]['income_mean']:.0f}, "
              f"credit={profiles[segment_name]['credit_score_mean']:.0f}, "
              f"DTI={profiles[segment_name]['debt_to_income_mean']:.2%}")

    # Assemble report
    report = {
        "pipeline": "customer-segmentation-underwriting",
        "n_customers": 5000,
        "n_features": len(FEATURE_COLS),
        "kmeans": {
            "k": seg_results["k"],
            "silhouette_avg": seg_results["silhouette_avg"],
            "inertias": seg_results["inertias"],
            "silhouette_by_k": seg_results["silhouette_scores"],
            "cluster_sizes": seg_results["cluster_sizes"],
        },
        "classifier": {
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 10,
            "train_accuracy": clf_results["train_accuracy"],
            "test_accuracy": clf_results["test_accuracy"],
            "feature_importances": clf_results["feature_importances"],
        },
        "segment_profiles": profiles,
        "business_impact": {
            "Established Prime": "Low risk, approve with standard terms",
            "Rising Prime": "Good risk, approve with mild underwriting",
            "Mass Market": "Moderate risk, standard underwriting required",
            "Subprime High-Risk": "High risk, decline or apply alternative lending criteria",
        },
    }

    # Save report
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {REPORT_PATH}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return report


if __name__ == "__main__":
    main()