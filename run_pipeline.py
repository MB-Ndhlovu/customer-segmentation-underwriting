import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_synthetic_data
from src.features import compute_features
from src.segment import find_optimal_k, fit_kmeans, profile_segments
from src.classify import train_classifier

SEGMENT_NAMES = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}

def run_pipeline():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE")
    print("=" * 60)

    print("\n[1/5] Generating synthetic data...")
    df = generate_synthetic_data(n=5000)
    print(f"  → {len(df)} rows, columns: {list(df.columns)}")

    print("\n[2/5] Computing features...")
    X = compute_features(df)
    print(f"  → {X.shape[1]} features: {list(X.columns)}")

    print("\n[3/5] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  → Scaled mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")

    print("\n[4/5] KMeans clustering...")
    best_k, silhouettes, elbows = find_optimal_k(X_scaled)
    print(f"  → Optimal k={best_k} (silhouette scores: {[round(s,3) for s in silhouettes]})")
    km, labels = fit_kmeans(X_scaled, n_clusters=4)
    print(f"  → Inertia: {km.inertia_:.2f}")
    print(f"  → Cluster sizes: {dict(enumerate(np.bincount(labels)))}")

    print("\n[5/5] Training RandomForest classifier...")
    clf, acc, report = train_classifier(X, labels)
    print(f"  → Test accuracy: {acc:.4f}")

    profiles = profile_segments(df, labels)
    feature_importance = dict(zip(X.columns, clf.feature_importances_.round(4)))

    seg_map = {i: SEGMENT_NAMES[i] for i in range(4)}
    cluster_cardinality = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}

    results = {
        "best_k": int(best_k),
        "silhouette_scores": {str(k): round(v, 4) for k, v in zip(range(2, 2+len(silhouettes)), silhouettes)},
        "inertia": round(float(km.inertia_), 4),
        "cluster_cardinality": cluster_cardinality,
        "segment_profiles": profiles.to_dict(),
        "classifier_accuracy": round(acc, 4),
        "classification_report": report,
        "feature_importance": feature_importance,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n[✓] Results saved to reports/segmentation_results.json")

    print("\n" + "=" * 60)
    print("SEGMENT SUMMARY")
    print("=" * 60)
    for cluster_id in sorted(cluster_cardinality):
        pct = cluster_cardinality[cluster_id] / len(labels) * 100
        seg_name = SEGMENT_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        income = profiles.loc[cluster_id, "income"]
        credit = profiles.loc[cluster_id, "credit_score"]
        dti = profiles.loc[cluster_id, "debt_to_income"]
        print(f"  Cluster {cluster_id} ({seg_name}): {cluster_cardinality[cluster_id]} ({pct:.1f}%) | "
              f"Avg income=${income:,.0f} | credit={credit:.0f} | DTI={dti:.3f}")

    print("\n" + "=" * 60)
    print("TOP FEATURE IMPORTANCES (RandomForest)")
    print("=" * 60)
    sorted_fi = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feat, imp in sorted_fi[:6]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f} {bar}")

    print("\n[✓] Pipeline complete.")
    return results

if __name__ == "__main__":
    results = run_pipeline()