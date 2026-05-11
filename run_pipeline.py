import json
import numpy as np
from src.data_loader import load_data, FEATURE_COLS
from src.features import build_features, prepare_for_clustering
from src.segment import find_optimal_k, fit_kmeans, profile_segments, assign_segment_names
from src.classify import train_classifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run():
    print("=== Loading data ===")
    df = load_data()
    print(f"  {len(df)} records, columns: {list(df.columns)}")

    print("\n=== Building features ===")
    df = build_features(df)
    engineered = [
        "income_per_year_of_employment", "loan_density", "income_per_age",
        "credit_per_income", "active_borrower", "high_loan_density",
        "income_stability_score", "homeowner", "long_tenure",
    ]
    all_features = FEATURE_COLS + engineered
    print(f"  {len(all_features)} total features")

    print("\n=== Preparing for clustering ===")
    X_scaled, X_raw, scaler = prepare_for_clustering(df, all_features)
    print(f"  Scaled shape: {X_scaled.shape}")

    print("\n=== Finding optimal K (Elbow + Silhouette) ===")
    inertias, silhouettes = find_optimal_k(X_scaled)
    best_k_idx = int(np.argmax(silhouettes[2:])) + 2  # skip k=2 for min range
    best_k = list(range(2, 9))[best_k_idx]
    best_sil = silhouettes[best_k_idx]
    print(f"  K range 2-8: silhouettes = {[round(s,3) for s in silhouettes]}")
    print(f"  Best K = {best_k}  (silhouette = {round(best_sil,3)})")
    print(f"  Using K = 4 as specified for business interpretability")

    print("\n=== KMeans clustering (K=4) ===")
    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    sil_score = float(np.round(silhouette_score(X_scaled, labels), 4))
    inertia = float(np.round(km.inertia_, 1))
    print(f"  Silhouette score: {sil_score}")
    print(f"  Inertia: {inertia}")

    print("\n=== Segment profiling ===")
    profiles = profile_segments(df, labels)
    for seg_id, p in profiles.items():
        print(f"  Segment {seg_id}: n={p['count']} ({p['pct']}%)  "
              f"income={p['income_mean']:,.0f}  credit={p['credit_score_mean']:.0f}  "
              f"emp_yrs={p['employment_years_mean']:.1f}  DTI={p['debt_to_income_mean']:.3f}")

    print("\n=== Training supervised classifier ===")
    clf, acc, report = train_classifier(X_scaled, labels)
    print(f"  RandomForest accuracy: {round(acc,4)}")

    print("\n=== Saving results ===")
    results = {
        "n_records": len(df),
        "n_features": len(all_features),
        "silhouette_score": sil_score,
        "inertia": inertia,
        "best_k": 4,
        "segment_profiles": profiles,
        "classifier_accuracy": round(float(acc), 4),
        "feature_importances": {
            f"f{i}": round(float(v), 4)
            for i, v in enumerate(clf.feature_importances_)
        },
    }

    import os
    os.makedirs("/home/workspace/Projects/customer-segmentation-underwriting/reports", exist_ok=True)
    out_path = "/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}")

    print("\n=== Pipeline complete ===")
    return results


if __name__ == "__main__":
    results = run()