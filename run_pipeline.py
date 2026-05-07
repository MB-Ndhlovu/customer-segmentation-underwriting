import json
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_loader import generate_customer_data
from src.features import compute_features
from src.segment import find_optimal_k, cluster, profile_segments
from src.classify import train_classifier

SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk'
}

def assign_segment_names(df, labels, X):
    """
    Assign our 4 canonical underwriting segment names to KMeans cluster IDs.
    Use canonical ordering: sorted by income asc → [Mass Market, Rising Prime, Established Prime, Subprime High-Risk].
    But Subprime is actually low-income high-risk, so we sort by a composite risk score instead.
    """
    # Compute composite: income (higher=better) + credit_score (higher=better) - dti (lower=better)
    centroid_df = profile_segments(X, labels, X.columns.tolist())

    # Composite score: higher = more prime / lower risk
    score = (
        centroid_df['income'] +
        centroid_df['credit_score'] * 1000 -
        centroid_df['debt_to_income'] * 100000
    )
    sorted_segs = score.sort_values().index.tolist()  # lowest score = worst

    # Map to canonical names: lowest score = Subprime High-Risk
    canonical = ['Subprime High-Risk', 'Mass Market', 'Rising Prime', 'Established Prime']
    name_map = {sorted_segs[i]: canonical[i] for i in range(4)}
    return name_map

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE FOR UNDERWRITING")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Generating synthetic customer data (n=5000)...")
    df = generate_customer_data(n=5000)
    print(f"    Shape: {df.shape}")

    # 2. Feature engineering
    print("\n[2] Computing features...")
    X = compute_features(df)
    print(f"    Feature columns: {X.shape[1]}")

    # 3. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Elbow + Silhouette analysis
    print("\n[3] Running Elbow + Silhouette analysis (k=2..8)...")
    elbow, silhouettes = find_optimal_k(X_scaled, k_range=range(2, 9))
    best_k_idx = int(np.argmax(silhouettes) + 2)
    best_sil = max(silhouettes)
    print(f"    Best k by silhouette: {best_k_idx}  (score: {best_sil:.4f})")
    print("    Silhouette scores:")
    for k, s in zip(range(2, 9), silhouettes):
        marker = " <-- best" if k == best_k_idx else ""
        print(f"      k={k}: {s:.4f}{marker}")

    # 5. KMeans clustering (k=4 as specified)
    print("\n[4] Clustering with KMeans (k=4)...")
    labels, centroids, sil = cluster(X_scaled, n_clusters=4)
    name_map = assign_segment_names(df, labels, X)
    cluster_sizes = np.bincount(labels)
    print(f"    Silhouette score: {sil:.4f}")
    print("    Cluster sizes:", {int(k): int(v) for k, v in enumerate(cluster_sizes)})
    print("    Cluster -> Name mapping:", {int(k): v for k, v in name_map.items()})

    # 6. Train supervised classifier
    print("\n[5] Training RandomForest classifier on cluster labels...")
    clf, acc, report, X_test, y_test, y_pred = train_classifier(X_scaled, labels)
    print(f"    Test accuracy: {acc:.4f}")

    # 7. Segment profiles
    print("\n[6] Segment profiles (mean feature values):")
    profiles = profile_segments(X, labels, X.columns.tolist())
    for seg_id in sorted(profiles.index):
        seg_name = name_map[int(seg_id)]
        size = int(cluster_sizes[seg_id])
        print(f"\n    Segment {seg_id}: {seg_name}  (n={size})")
        row = profiles.loc[seg_id]
        print(f"      income:          {row['income']:>14.2f}")
        print(f"      credit_score:    {row['credit_score']:>14.2f}")
        print(f"      employment_yrs:  {row['employment_years']:>14.2f}")
        print(f"      dti:             {row['debt_to_income']:>14.4f}")
        print(f"      loan_count:      {row['loan_history_count']:>14.2f}")
        print(f"      age:             {row['age']:>14.2f}")
        print(f"      home_owner:      {row['home_ownership']:>14.2f}")
        print(f"      verified_income: {row['verified_income']:>14.2f}")

    # 8. Save results
    results = {
        'silhouette_k4': float(sil),
        'silhouette_by_k': {str(k): float(s) for k, s in zip(range(2, 9), silhouettes)},
        'best_k': int(best_k_idx),
        'best_silhouette': float(best_sil),
        'cluster_sizes': {str(k): int(v) for k, v in enumerate(cluster_sizes)},
        'segment_name_map': {str(k): v for k, v in name_map.items()},
        'centroids': centroids.tolist(),
        'classifier_accuracy': float(acc),
        'classification_report': report,
        'profiles': {str(k): {col: float(v) for col, v in row.items()}
                     for k, row in profiles.iterrows()}
    }

    out_path = '/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[7] Results saved to {out_path}")

    # Human-readable summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Silhouette (k=4): {sil:.4f}")
    print(f"Best k={best_k_idx}  silhouette={best_sil:.4f}")
    print(f"Classifier accuracy: {acc:.4f}")
    print(f"Cluster sizes: {[int(cluster_sizes[i]) for i in range(4)]}")
    print("Segments:")
    for seg_id in sorted(name_map.keys()):
        print(f"  Cluster {seg_id} -> {name_map[seg_id]}")

    summary = (
        f"Pipeline complete.\n"
        f"Silhouette (k=4): {sil:.4f}\n"
        f"Best k={best_k_idx} silhouette={best_sil:.4f}\n"
        f"Classifier accuracy: {acc:.4f}\n"
        f"Cluster sizes: {[int(cluster_sizes[i]) for i in range(4)]}\n"
        f"Segments: {[name_map[i] for i in sorted(name_map)]}"
    )
    return results, summary

if __name__ == '__main__':
    main()