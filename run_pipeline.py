import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.data_loader import load_data
from src.features import build_features, get_feature_names, FEATURE_COLS
from src.segment import fit_kmeans, find_optimal_k, profile_segments, assign_segment_names, SEGMENT_NAMES
from src.classify import train_classifier

def run():
    print("=" * 60)
    print("Customer Segmentation Pipeline — Underwriting")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading data...")
    df = load_data(n=5000)
    print(f"    Loaded {len(df)} rows")
    print(f"    True segment distribution:")
    for seg, cnt in df["_true_segment"].value_counts().sort_index().items():
        print(f"      {seg} ({SEGMENT_NAMES[seg]}): {cnt}")

    # 2. Build features
    print("\n[2] Engineering features...")
    X = build_features(df)
    feature_names = get_feature_names()
    print(f"    {len(feature_names)} features total")
    print(f"    Features: {feature_names}")

    # 3. Scale
    print("\n[3] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("    StandardScaler applied")

    # 4. Find optimal K (Elbow + Silhouette)
    print("\n[4] Finding optimal K...")
    k_range = range(2, 9)
    inertias, silhouettes = find_optimal_k(X_scaled, k_range)
    for k, (iner, sil) in zip(k_range, zip(inertias, silhouettes)):
        print(f"    k={k}: inertia={iner:.0f}, silhouette={sil:.4f}")

    # 5. Fit KMeans with k=4
    print("\n[5] Fitting KMeans (k=4)...")
    km, cluster_labels = fit_kmeans(X_scaled, n_clusters=4)
    sil_score = silhouette_score(X_scaled, cluster_labels)
    print(f"    Silhouette Score: {sil_score:.4f}")

    # 6. Profile segments
    print("\n[6] Profiling segments...")
    profiles = profile_segments(X, cluster_labels, df)
    for seg_id, p in profiles.items():
        print(f"    Segment {seg_id} — {SEGMENT_NAMES[seg_id]}:")
        print(f"      n={p['n']}, avg income={p['mean_income']:.0f}, "
              f"avg credit={p['mean_credit_score']:.0f}, DTI={p['mean_dti']:.3f}")
        print(f"      homeownership={p['pct_homeowners']:.1%}, "
              f"verified_income={p['pct_verified_income']:.1%}")

    # 7. Supervised classification
    print("\n[7] Training RandomForest classifier...")
    clf, clf_acc, clf_report = train_classifier(df[FEATURE_COLS], cluster_labels)
    print(f"    Accuracy: {clf_acc:.4f}")
    target_names = ["Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"]
    for cls_name in target_names:
        r = clf_report[cls_name]
        print(f"    {cls_name}: precision={r['precision']:.3f}, "
              f"recall={r['recall']:.3f}, f1={r['f1-score']:.3f}")

    # 8. Top features
    importances = clf.feature_importances_
    feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    print("\n[8] Feature importances:")
    for fname, imp in feat_imp:
        print(f"    {fname}: {imp:.4f}")

    # 9. Build classification report dict safely
    classification_report_dict = {}
    for cls_name in target_names:
        r = clf_report[cls_name]
        classification_report_dict[cls_name] = {
            m: round(float(v), 4) for m, v in r.items() if m != "support"
        }

    # 10. Save results
    results = {
        "n_samples": len(df),
        "n_features": len(feature_names),
        "k_chosen": 4,
        "silhouette_score": round(float(sil_score), 4),
        "classifier_accuracy": round(float(clf_acc), 4),
        "segment_profiles": {
            str(k): v for k, v in profiles.items()
        },
        "segment_names": SEGMENT_NAMES,
        "feature_importances": {fname: round(float(imp), 4) for fname, imp in feat_imp},
        "k_evaluation": {
            str(k): {"inertia": round(float(i), 2), "silhouette": round(float(s), 4)}
            for k, (i, s) in zip(k_range, zip(inertias, silhouettes))
        },
        "classification_report": classification_report_dict,
    }

    import os
    os.makedirs("reports", exist_ok=True)
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[9] Results saved to reports/segmentation_results.json")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)

    return results

if __name__ == "__main__":
    results = run()
