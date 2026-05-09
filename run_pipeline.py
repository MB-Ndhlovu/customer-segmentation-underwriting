import json
from sklearn.metrics import silhouette_score
from src.data_loader import generate_customer_data
from src.features import build_features
from src.segment import find_optimal_k, cluster, profile_segments, SEGMENT_NAMES
from src.classify import train_classifier

def run_pipeline():
    print("=== Loading data ===")
    df = generate_customer_data(5000)
    print(f"Generated {len(df)} rows")

    print("\n=== Building features ===")
    X = build_features(df)
    print(f"Features: {list(X.columns)}")

    print("\n=== Elbow + Silhouette analysis ===")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias, silhouettes = find_optimal_k(X_scaled)
    best_k = 4
    best_sil = max(silhouettes)
    print(f"K range 2-8 | Best silhouette: {best_sil:.4f} at k={silhouettes.index(best_sil)+2}")
    print(f"Using k={best_k}")

    print("\n=== Clustering ===")
    labels, km_model = cluster(X_scaled, n_clusters=best_k)
    sil_score = silhouette_score(X_scaled, labels)
    print(f"Silhouette score: {sil_score:.4f}")

    print("\n=== Segment profiling ===")
    profiles = profile_segments(X, labels)
    for seg_id, prof in profiles.items():
        print(f"  [{seg_id}] {prof['name']} — n={prof['count']}, "
              f"income={prof['mean_income']:.0f}, credit_score={prof['mean_credit_score']:.0f}")

    print("\n=== Training RandomForest classifier ===")
    clf, acc, report = train_classifier(X, labels)
    print(f"Accuracy: {acc:.4f}")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"  class {label}: precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}")

    results = {
        "n_samples": len(df),
        "silhouette_score": round(sil_score, 4),
        "n_clusters": best_k,
        "rf_accuracy": round(acc, 4),
        "segment_profiles": {str(k): v for k, v in profiles.items()},
    }

    with open("reports/segmentation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Saved reports/segmentation_results.json ===")

    summary = (
        f"Segments: {best_k} | Silhouette: {sil_score:.4f} | "
        f"RF Accuracy: {acc:.4f} | Segments: "
        + ", ".join([f"{v['name']}({v['count']})" for v in profiles.values()])
    )
    print(f"\nPipeline summary: {summary}")
    return results

if __name__ == "__main__":
    run_pipeline()