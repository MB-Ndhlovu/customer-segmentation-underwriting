"""Execute the full customer segmentation pipeline."""

import json
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_customer_data
from src.features import build_features, get_feature_names
from src.segment import run_segmentation, profile_segments, SEGMENT_NAMES
from src.classify import train_classifier


def run():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Generating 5000 synthetic customer records...")
    df = generate_customer_data(5000)
    print(f"    Records: {len(df)}")

    # 2. Feature engineering
    print("\n[2] Building features...")
    X = build_features(df)
    feature_names = get_feature_names()
    print(f"    Features: {len(feature_names)}")

    # 3. Scale
    print("\n[3] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_names])
    print("    Done.")

    # 4. Segmentation
    print("\n[4] Running KMeans segmentation...")
    seg_stats, labels, km = run_segmentation(X_scaled, feature_names)
    print(f"    Silhouette Score: {seg_stats['silhouette_score']}")
    print(f"    Clusters used: {seg_stats['k_used']}")

    # 5. Profile segments
    print("\n[5] Profiling segments...")
    profile = profile_segments(X, labels)

    segment_counts = X.copy()
    segment_counts["segment_label"] = labels
    counts = segment_counts.groupby("segment_label").size().to_dict()

    profile_data = {}
    for idx, row in profile.iterrows():
        profile_data[int(idx)] = {
            "name": SEGMENT_NAMES[int(idx)],
            "count": int(row["count"]),
            "pct": round(float(row["count"]) / 5000 * 100, 1),
            "income_mean": float(row["income"]),
            "credit_score_mean": float(row["credit_score"]),
            "employment_years_mean": float(row["employment_years"]),
            "debt_to_income_mean": float(row["debt_to_income"]),
            "loan_history_count_mean": float(row["loan_history_count"]),
            "age_mean": float(row["age"]),
            "home_ownership_rate": round(float(row["home_ownership"]), 3),
            "verified_income_rate": round(float(row["verified_income"]), 3),
        }

    print("\n  Segment Distribution:")
    for seg_id, info in profile_data.items():
        print(f"    [{seg_id}] {info['name']:25s} — {info['count']:4d} ({info['pct']}%)  "
              f"income={int(info['income_mean']):6d}  credit={int(info['credit_score_mean']):4d}  "
              f"DTI={info['debt_to_income_mean']:.2f}")

    # 6. Classification
    print("\n[6] Training RandomForest classifier...")
    # Use raw features only (what's available at application time)
    raw_features = feature_names[:8]
    clf_results, clf = train_classifier(X, labels, raw_features)
    print(f"    Test Accuracy: {clf_results['accuracy']}")
    print(f"    Train/Test split: {clf_results['train_size']}/{clf_results['test_size']}")

    print("\n  Top 5 Predictive Features:")
    top_feats = sorted(clf_results["feature_importance"].items(), key=lambda x: -x[1])[:5]
    for feat, imp in top_feats:
        print(f"    {feat:25s}  {imp:.4f}")

    # 7. Assemble report
    print("\n[7] Saving results...")
    report = {
        "pipeline_version": "1.0",
        "n_records": 5000,
        "n_features": len(feature_names),
        "segmentation": {
            "method": "KMeans",
            "n_clusters": 4,
            "silhouette_score": seg_stats["silhouette_score"],
            "segment_names": SEGMENT_NAMES,
        },
        "segment_profiles": profile_data,
        "classification": {
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 10,
            "test_accuracy": clf_results["accuracy"],
            "train_size": clf_results["train_size"],
            "test_size": clf_results["test_size"],
            "features_used": raw_features,
            "feature_importance": clf_results["feature_importance"],
        },
    }

    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("    Saved: reports/segmentation_results.json")

    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Records processed : 5000")
    print(f"  Features used     : {len(feature_names)} total, {len(raw_features)} for classification")
    print(f"  Silhouette Score  : {seg_stats['silhouette_score']} (cluster separation)")
    print(f"  Classifier Acc   : {clf_results['accuracy']} (predict segment from app data)")
    print(f"\n  Segments:")
    for seg_id, info in profile_data.items():
        print(f"    [{seg_id}] {info['name']:22s}  n={info['count']:4d}  "
              f"income={int(info['income_mean']):6d}  credit={int(info['credit_score_mean']):4d}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    run()