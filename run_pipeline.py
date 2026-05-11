import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import build_dataset
from src.features import build_features
from src.segment import fit_kmeans, profile_segments, assign_segment_names
from src.classify import train_classifier, feature_importance

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = build_dataset()
    print(f"      Loaded {len(df)} records")

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    df = build_features(df)
    engineered = [c for c in df.columns if c not in [
        "income","credit_score","employment_years","debt_to_income",
        "loan_history_count","age","home_ownership","verified_income","segment_label"
    ]]
    print(f"      Engineered {len(engineered)} features: {engineered}")

    # 3. Clustering
    print("\n[3/5] Running KMeans clustering (k=4)...")
    df, km, scaler, X_scaled, sil, inertia = fit_kmeans(df, n_clusters=4)
    profiles = profile_segments(df)
    segment_names = assign_segment_names(profiles)
    print(f"      Silhouette score: {sil:.4f}")
    print(f"      Inertia: {inertia:.0f}")
    print("\n      Segment Profiles:")
    for seg, stats in profiles.items():
        name = segment_names.get(seg, f"Segment {seg}")
        print(f"      [{seg}] {name}")
        print(f"          n={stats['count']} ({stats['pct']}%) | "
              f"Income=${stats['income_mean']:,.0f} | "
              f"Credit={stats['credit_score_mean']} | "
              f"DTI={stats['debt_to_income_mean']:.2f} | "
              f"EmpYrs={stats['employment_years_mean']} | "
              f"HomeOwner={stats['home_ownership_pct']}% | "
              f"VerifiedInc={stats['verified_income_pct']}%")

    # 4. Classification
    print("\n[4/5] Training RandomForest classifier...")
    clf, acc, report = train_classifier(df)
    fi = feature_importance(clf)
    print(f"      Test accuracy: {acc:.4f}")
    print("\n      Classification Report:")
    for label, metrics in report.items():
        if label not in ("accuracy", "macro avg", "weighted avg"):
            name = segment_names.get(int(label), f"Segment {label}")
            m = metrics
            print(f"      [{label}] {name}: precision={m['precision']:.2f}  "
                  f"recall={m['recall']:.2f}  f1={m['f1-score']:.2f}  "
                  f"support={int(m['support'])}")
    print("\n      Feature Importance:")
    for feat, score in fi.items():
        bar = "█" * int(score * 50)
        print(f"      {feat:<30} {score:.4f} {bar}")

    # 5. Save results
    print("\n[5/5] Saving results...")
    out = {
        "n_records": len(df),
        "silhouette_score": round(sil, 4),
        "inertia": round(inertia, 2),
        "test_accuracy": round(acc, 4),
        "segment_names": {str(k): v for k, v in segment_names.items()},
        "segment_profiles": {str(k): v for k, v in profiles.items()},
        "feature_importance": {k: round(v, 4) for k, v in fi.items()},
        "classification_report": report,
    }
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "reports"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "reports", "segmentation_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"      Saved to {out_path}")

    # Summary line for Telegram
    summary = (
        f"Pipeline complete — {len(df)} records | "
        f"4 segments | Silhouette={sil:.3f} | "
        f"RF accuracy={acc:.1%}\n"
        f"Segments: Established Prime (high income/excellent credit), "
        f"Rising Prime (growing income), Mass Market (moderate risk), "
        f"Subprime High-Risk (high DTI/low credit)"
    )
    print("\n" + "=" * 60)
    print("SUMMARY:", summary)
    print("=" * 60)

    return out, summary

if __name__ == "__main__":
    results, summary = main()