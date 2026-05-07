"""Execute full segmentation + classification pipeline."""
import json
import pandas as pd
from src.data_loader import generate_customer_data
from src.features import build_features, get_feature_columns
from src.segment import assign_segment_labels
from src.classify import train_classifier

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Generating synthetic customer data...")
    df = generate_customer_data(5000)
    print(f"    {len(df)} records generated")

    # 2. Feature engineering
    print("\n[2] Building features...")
    df_fe = build_features(df)
    feature_cols = get_feature_columns()
    print(f"    Features: {feature_cols}")

    # 3. Clustering
    print("\n[3] Running KMeans segmentation (k=4)...")
    df_seg, analysis = assign_segment_labels(df_fe, feature_cols, n_clusters=4)

    print(f"    Silhouette score @ k=4: {analysis['silhouette_at_k4']}")
    print(f"    Best k from analysis:   {analysis['best_k_from_analysis']}")
    print("\n    Segment profiles:")
    for seg_id, prof in analysis["segment_profiles"].items():
        print(f"    [{seg_id}] {prof['name']:<22} n={prof['count']:>5} ({prof['pct']}%)\n"
              f"        income_mean=${prof['income_mean']:,.0f}  "
              f"credit={prof['credit_score_mean']:.0f}  "
              f"DTI={prof['debt_to_income_mean']:.3f}  "
              f"tenure={prof['employment_years_mean']:.1f}yr  "
              f"verified={prof['verified_income_pct']}%")

    # 4. Classification
    print("\n[4] Training RandomForest classifier on cluster labels...")
    clf_results = train_classifier(df_seg, feature_cols)
    print(f"    Test accuracy: {clf_results['test_accuracy']}")
    print(f"    CV mean accuracy: {clf_results['cv_mean_accuracy']} +/- {clf_results['cv_std']}")
    print("    Feature importances:")
    for feat, imp in sorted(clf_results["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"      {feat:<25} {imp:.4f}")

    # 5. Save results
    results = {
        "n_records": len(df_seg),
        "features": feature_cols,
        "clustering": {
            "method": "KMeans",
            "k": 4,
            "silhouette_at_k4": analysis["silhouette_at_k4"],
            "best_k_from_analysis": analysis["best_k_from_analysis"],
            "silhouette_scores": analysis["silhouette_scores"],
            "segment_profiles": analysis["segment_profiles"],
        },
        "classification": {
            "model": "RandomForestClassifier",
            "test_accuracy": clf_results["test_accuracy"],
            "cv_mean_accuracy": clf_results["cv_mean_accuracy"],
            "cv_std": clf_results["cv_std"],
            "feature_importances": clf_results["feature_importances"],
        },
    }

    out_path = "reports/segmentation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[5] Results saved to {out_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    return results

if __name__ == "__main__":
    main()