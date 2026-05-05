"""Execute full customer segmentation pipeline."""
import json
import joblib
from src.data_loader import generate_customers
from src.features import build_features
from src.segment import run_segmentation
from src.classify import train_classifier


def main():
    print("=" * 60)
    print("Customer Segmentation for Underwriting — Pipeline")
    print("=" * 60)

    print("\n[1/4] Generating synthetic customer data (5000 rows)...")
    df = generate_customers(5000)
    print(f"  Shape: {df.shape}")
    print(f"  Segments: {df['segment_label'].value_counts().sort_index().to_dict()}")

    print("\n[2/4] Building engineered features...")
    df = build_features(df)
    engineered_cols = [
        "income_per_employment_year", "loan_per_year", "credit_to_dti_ratio",
        "dti_risk_tier", "employment_stability_tier", "credit_tier",
        "income_growth_proxy", "combined_risk_score",
    ]
    print(f"  Added {len(engineered_cols)} features: {engineered_cols}")

    print("\n[3/4] Running KMeans segmentation (k=4)...")
    seg_out = run_segmentation(df)
    res = seg_out["results"]
    print(f"  Silhouette Score: {res['silhouette_score']}")
    print(f"  Best k by silhouette: {res['best_k_by_silhouette']}")
    print("  Cluster Profiles (mean feature values):")
    for p in res["cluster_profiles"]:
        print(f"    {p['segment_name']}: n={p.get('count',0)}, "
              f"income={p.get('income',0)}, credit={p.get('credit_score',0)}, "
              f"DTI={p.get('debt_to_income',0)}, emp_yrs={p.get('employment_years',0)}")

    print("\n[4/4] Training RandomForest classifier on cluster labels...")
    clf_out = train_classifier(df, seg_out["labels"])
    clf_res = clf_out["results"]
    print(f"  Test Accuracy: {clf_res['accuracy']:.2%}")
    print("  Feature Importances (top 4):")
    sorted_fi = sorted(clf_res["feature_importances"].items(), key=lambda x: -x[1])[:4]
    for feat, imp in sorted_fi:
        print(f"    {feat}: {imp:.4f}")

    report = {
        "pipeline": "customer-segmentation-underwriting",
        "n_customers": 5000,
        "n_features": 8,
        "segmentation": {
            "method": "KMeans",
            "k": res["n_clusters"],
            "silhouette_score": res["silhouette_score"],
            "best_k_by_silhouette": res["best_k_by_silhouette"],
            "cluster_counts": res["cluster_counts"],
            "segment_mapping": res["segment_mapping"],
            "cluster_profiles": res["cluster_profiles"],
        },
        "classification": {
            "method": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 12,
            "test_accuracy": clf_res["accuracy"],
            "feature_importances": clf_res["feature_importances"],
            "classification_report": clf_res["classification_report"],
        },
    }

    report_path = "/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Report saved: {report_path}")

    model_path = "/home/workspace/Projects/customer-segmentation-underwriting/model_clustering.joblib"
    scaler_path = "/home/workspace/Projects/customer-segmentation-underwriting/scaler.joblib"
    clf_path = "/home/workspace/Projects/customer-segmentation-underwriting/model_classifier.joblib"
    joblib.dump(seg_out["km"], model_path)
    joblib.dump(seg_out["scaler"], scaler_path)
    joblib.dump(clf_out["clf"], clf_path)
    print(f"[+] Artifacts saved: {model_path}, {scaler_path}, {clf_path}")

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(json.dumps({
        "silhouette_score": res["silhouette_score"],
        "classification_accuracy": clf_res["accuracy"],
        "segments": res["segment_mapping"],
    }, indent=2))

    return report


if __name__ == "__main__":
    main()