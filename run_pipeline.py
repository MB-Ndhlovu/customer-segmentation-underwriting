import sys
sys.path.insert(0, "/home/workspace/Projects/customer-segmentation-underwriting")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

from src.data_loader import load_data
from src.features import build_features
from src.segment import (
    fit_kmeans, find_optimal_k, profile_segments, save_results,
    remap_labels_to_segments, SEGMENT_LABELS
)
from src.classify import train_classifier

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    # 1. Load / generate data
    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"    Rows: {len(df)}")

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    X_raw = build_features(df)
    base_features = [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age", "home_ownership", "verified_income",
    ]
    print(f"    Features ({len(X_raw.columns)}): {list(X_raw.columns)}")

    # 3. Scale + cluster
    print("\n[3/5] Running KMeans segmentation...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    k_range = range(2, 9)
    elbow, silhouettes, best_k = find_optimal_k(X_scaled, k_range)
    print(f"    Optimal k (silhouette): {best_k}")
    print(f"    Silhouette by k: {dict(zip(k_range, [round(s,4) for s in silhouettes]))}")

    km, raw_labels, sil_score = fit_kmeans(X_scaled, n_clusters=4)
    print(f"    Raw Silhouette @ k=4: {sil_score:.4f}")
    print(f"    Inertia: {km.inertia_:.2f}")

    # Remap clusters to match target segment order
    labels, mapping = remap_labels_to_segments(df, raw_labels, base_features)
    print(f"    Cluster remap: {mapping}")

    # 4. Profile segments
    print("\n[4/5] Profiling segments...")
    df["segment_label"] = labels
    profiles = profile_segments(df, labels)

    print("\n  Segment Summary:")
    for name, p in profiles.items():
        print(f"    [{name}]")
        print(f"      n={p['count']} ({p['pct']}%) | "
              f"income=${p['income_mean']:,.0f} | "
              f"credit={p['credit_score_mean']} | "
              f"DTI={p['debt_to_income_mean']:.2f} | "
              f"home_own={p['home_ownership_rate']:.1%} | "
              f"verified={p['verified_income_rate']:.1%}")

    # 5. Train supervised classifier
    print("\n[5/5] Training RandomForest classifier...")
    y = df["segment_label"]
    clf, acc, feat_imp, report = train_classifier(X_raw, y)

    print(f"    Accuracy: {acc:.4f}")
    print("\n    Top 5 Feature Importances:")
    for i, (feat, imp) in enumerate(list(feat_imp.items())[:5]):
        print(f"      {i+1}. {feat}: {imp}")

    # Save results
    os_path = "/home/workspace/Projects/customer-segmentation-underwriting"
    import os
    os.makedirs(os.path.join(os_path, "reports"), exist_ok=True)

    save_results(
        profiles,
        sil_score,
        elbow,
        {str(k): round(s, 4) for k, s in zip(k_range, silhouettes)},
        best_k,
        os.path.join(os_path, "reports/segmentation_results.json"),
    )

    clf_results = {
        "accuracy": round(acc, 4),
        "feature_importance": feat_imp,
        "classification_report": {
            str(k): {kk: round(float(vv), 4) if isinstance(vv, float) else vv
                     for kk, vv in v.items()}
            if isinstance(v, dict) else round(float(v), 4)
            for k, v in report.items()
        },
    }
    with open(os.path.join(os_path, "reports/classifier_results.json"), "w") as f:
        json.dump(clf_results, f, indent=2)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Silhouette Score  : {sil_score:.4f}")
    print(f"  Classifier Accuracy: {acc:.4f}")
    print(f"  Artifacts: reports/segmentation_results.json")
    print(f"            reports/classifier_results.json")

    return {
        "silhouette": sil_score,
        "accuracy": acc,
        "best_k": best_k,
        "profiles": profiles,
        "feat_imp": feat_imp,
    }


if __name__ == "__main__":
    result = main()

    summary = {
        "silhouette_score": round(result["silhouette"], 4),
        "classifier_accuracy": round(result["accuracy"], 4),
        "best_k": result["best_k"],
        "segments": result["profiles"],
    }
    with open("/tmp/pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[TG_SUMMARY_START]")
    print(json.dumps(summary, indent=2))
    print("[TG_SUMMARY_END]")