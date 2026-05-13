import json
import sys
from pathlib import Path

from src.data_loader import generate_customer_data, SEGMENT_NAMES
from src.features import build_features
from src.segment import assign_segments, profile_segments
from src.classify import train_classifier, get_feature_importance


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def run():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE FOR UNDERWRITING")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (5000 rows)...")
    df = generate_customer_data(5000)
    print(f"    Rows: {len(df)}")
    print(f"    True segment distribution:\n{df['segment_label'].value_counts().sort_index().to_string()}")

    # 2. Feature engineering
    print("\n[2/5] Building features...")
    df = build_features(df)
    print(f"    Features: {df.columns.tolist()}")

    # 3. Clustering
    print("\n[3/5] Running KMeans segmentation...")
    df, sil, scaler, km = assign_segments(df, n_clusters=4)
    profiles = profile_segments(df)
    print(f"    Silhouette Score: {sil:.4f}")
    print("\n    Segment Profiles:")
    print(profiles.to_string(index=False))

    # 4. Supervised classification
    print("\n[4/5] Training RandomForest classifier...")
    clf, clf_acc = train_classifier(df)
    print(f"    Classifier Test Accuracy: {clf_acc:.4f}")

    imp = get_feature_importance(clf, [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_history_count", "age",
        "home_ownership", "verified_income",
    ])
    print("\n    Feature Importance:")
    print(imp.to_string(index=False))

    # 5. Save results
    print("\n[5/5] Saving results...")
    result = {
        "silhouette_score": round(sil, 4),
        "classifier_accuracy": round(clf_acc, 4),
        "n_clusters": 4,
        "segment_profiles": profiles.to_dict(orient="records"),
        "feature_importance": imp.to_dict(orient="records"),
        "segment_names": SEGMENT_NAMES,
    }

    out_path = REPORT_DIR / "segmentation_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Saved to {out_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nSilhouette Score : {sil:.4f}")
    print(f"Classifier Accuracy: {clf_acc:.4f}")
    print("\nSegment Distribution:")
    for seg_id, name in SEGMENT_NAMES.items():
        count = int((df["segment_label"] == seg_id).sum())
        pct = count / len(df) * 100
        print(f"  {seg_id} ({name}): {count} ({pct:.1f}%)")

    return result


if __name__ == "__main__":
    result = run()
