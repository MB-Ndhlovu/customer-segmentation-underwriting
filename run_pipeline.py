import os
import json
import joblib
import numpy as np

from src.data_loader import generate_customer_data
from src.features import build_features
from src.segment import run_clustering, profile_segments, assign_segment_names, find_optimal_k, save_results, remap_profiles, SEGMENT_NAMES
from src.classify import train_classifier, feature_importance
from sklearn.preprocessing import StandardScaler

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE — UNDERWRITING")
    print("=" * 60)

    # 1. Load / generate data
    print("\n[1] Generating synthetic customer data (n=5000)...")
    df = generate_customer_data(n=5000, seed=42)
    print(f"    Rows: {len(df)}")
    print(f"    Features: income, credit_score, employment_years, "
          "debt_to_income, loan_history_count, age, home_ownership, verified_income")

    # 2. Feature engineering
    print("\n[2] Building features (RFM + behavioral + stability)...")
    X = build_features(df)
    print(f"    Total features: {X.shape[1]}")
    print(f"    Feature list: {list(X.columns)}")

    # 3. Scale and cluster
    print("\n[3] Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n[4] Running Elbow + Silhouette analysis (k=2..10)...")
    inertias, silhouettes = find_optimal_k(X_scaled, max_k=10)
    best_k_idx = int(np.argmax(silhouettes))
    best_sil = silhouettes[best_k_idx]
    print(f"    Best silhouette score: {best_sil:.4f} at k={best_k_idx + 2}")

    print("\n[5] Running KMeans with k=4...")
    km, labels, sil = run_clustering(X_scaled, n_clusters=4)
    print(f"    Silhouette score: {sil:.4f}")
    print(f"    Inertia: {km.inertia_:.2f}")

    # 4. Profile segments
    print("\n[6] Profiling segments...")
    profiles = profile_segments(X, labels)
    relabeled = assign_segment_names(labels, X)
    remapped_profiles = remap_profiles(profiles, labels, relabeled, X)

    print("\n    Segment distribution:")
    for i in range(4):
        count = int(np.sum(relabeled == i))
        pct = count / len(relabeled) * 100
        print(f"    [{i}] {SEGMENT_NAMES[i]:25s} — {count:5d} customers ({pct:.1f}%)")

    # 5. Train supervised classifier
    print("\n[7] Training RandomForest classifier on cluster labels...")
    clf, acc, report, _ = train_classifier(X, relabeled, test_size=0.2, random_state=42)
    print(f"    Test accuracy: {acc:.4f}")

    print("\n    Feature importances (top → bottom):")
    for feat, imp in feature_importance(clf, X.columns):
        bar = '█' * int(imp * 50)
        print(f"    {feat:30s} {imp:.4f} {bar}")

    # 6. Save artifacts
    print("\n[8] Saving artifacts...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    joblib.dump(clf, 'models/classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(km, 'models/kmeans.pkl')
    print("    Saved: models/classifier.pkl, models/scaler.pkl, models/kmeans.pkl")

    results = save_results(remapped_profiles, sil, relabeled, 'reports/segmentation_results.json')
    print("    Saved: reports/segmentation_results.json")

    # 7. Summary
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (RandomForest)")
    print("=" * 60)
    print(report)

    print("\nSEGMENT PROFILES (mean values)")
    print("-" * 60)
    feature_cols = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                    'loan_history_count', 'age', 'home_ownership', 'verified_income']
    header = f"{'Feature':<30}" + "".join(f"{SEGMENT_NAMES[i]:>18}" for i in range(4))
    print(header)
    print("-" * 60)
    for feat in feature_cols:
        row = f"{feat:<30}"
        for i in range(4):
            row += f"{remapped_profiles[i].get(feat, 0):>18.2f}"
        print(row)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return results

if __name__ == '__main__':
    results = main()