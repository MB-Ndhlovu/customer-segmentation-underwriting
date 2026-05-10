import os
import json
import numpy as np
from src.data_loader import load_data
from src.features import build_features
from src.segment import run_segmentation, save_results
from src.classify import train_classifier, predict_segment

FEATURE_COLS = [
    "income",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "loan_history_count",
    "age",
    "home_ownership_status",
    "verified_income",
]

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data...")
    df = load_data(n=5000)
    print(f"      {len(df)} rows loaded")

    # 2. Feature engineering
    print("\n[2/5] Building features...")
    df = build_features(df)
    engineered = [
        "recency_score", "frequency_score", "monetary_score",
        "high_dti_flag", "credit_to_income_ratio", "income_stability",
        "loan_density", "employment_tenure_bucket", "income_age_ratio",
        "home_verified_combo"
    ]
    print(f"      {len(engineered)} engineered features added")

    # 3. Segmentation
    print("\n[3/5] Running KMeans segmentation (k=4)...")
    labels, scaler, seg_results = run_segmentation(df, FEATURE_COLS)
    df["segment_label"] = labels
    df["segment_name"] = [SEGMENT_NAMES[l] for l in labels]

    print(f"      Silhouette score: {seg_results['silhouette_score']}")
    print(f"      Optimal k found:  {seg_results['optimal_k_found']}")
    print("\n      Segment profiles:")
    for cid, profile in seg_results["profiles"].items():
        name = seg_results["label_map"][str(cid)]
        print(f"      Cluster {cid} ({name}):")
        print(f"        count={profile['count']}, pct={profile['pct']}%")
        print(f"        income=${profile['mean_income']:,.0f}, credit={profile['mean_credit_score']}")
        print(f"        employment={profile['mean_employment_years']} yrs, DTI={profile['mean_debt_to_income']:.3f}")

    # 4. Classification
    print("\n[4/5] Training RandomForest classifier...")
    clf, acc, importance, report = train_classifier(df, labels, FEATURE_COLS)
    print(f"      Test accuracy: {acc:.4f}")
    print("\n      Feature importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"        {feat}: {imp}")

    print("\n      Classification report:")
    for seg_name in SEGMENT_NAMES.values():
        r = report[seg_name]
        print(f"        {seg_name}: precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")

    # 5. Save artifacts
    print("\n[5/5] Saving artifacts...")

    # Add segment info to results for JSON
    seg_results["test_accuracy"] = round(acc, 4)
    seg_results["feature_importance"] = importance
    seg_results["classification_report"] = {
        k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)}
        for k, v in report.items() if k in SEGMENT_NAMES.values()
    }

    save_path = "reports/segmentation_results.json"
    os.makedirs("reports", exist_ok=True)
    save_results(seg_results, save_path)

    # Segment distribution
    seg_dist = df["segment_name"].value_counts().to_dict()
    print(f"\nSegment distribution:")
    for seg, cnt in seg_dist.items():
        print(f"  {seg}: {cnt} ({cnt/len(df)*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)

    return seg_results


if __name__ == "__main__":
    results = main()
    output_summary = json.dumps(results, indent=2)