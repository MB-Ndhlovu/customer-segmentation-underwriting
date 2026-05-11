import sys
import json
import joblib
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_data
from src.features import build_features, get_feature_names
from src.segment import find_optimal_k, fit_kmeans, profile_segments, assign_segment_names
from src.classify import train_classifier, get_feature_importance


def run():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE FOR UNDERWRITING")
    print("=" * 60)

    # 1. Load / Generate data
    print("\n[1/6] Loading synthetic customer data (5000 rows)...")
    df = load_data()
    print(f"    Dataset shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")

    # 2. Build features
    print("\n[2/6] Engineering features...")
    X_raw = df[['income', 'credit_score', 'employment_years',
                'debt_to_income', 'loan_history_count', 'age',
                'home_ownership', 'verified_income']]
    X_feat = build_features(df)
    feature_names = get_feature_names()
    print(f"    Features ({len(feature_names)}): {feature_names}")

    # 3. Scale
    print("\n[3/6] Scaling features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)
    print(f"    Scaled shape: {X_scaled.shape}")

    # 4. KMeans clustering
    print("\n[4/6] Running KMeans clustering (k=4)...")
    inertias, silhouettes = find_optimal_k(X_scaled)
    print(f"    K range tested: 2–{2 + len(inertias) - 1}")
    print(f"    Silhouette scores: {[round(s, 3) for s in silhouettes]}")

    best_k = 4  # fixed per business requirement
    km, labels = fit_kmeans(X_scaled, n_clusters=best_k)
    sil = silhouettes[best_k - 2]
    print(f"    Silhouette @ k=4: {sil:.4f}")
    print(f"    Inertia @ k=4: {km.inertia_:.2f}")

    # 5. Profile segments
    print("\n[5/6] Profiling segments...")
    profiles = profile_segments(df, labels, feature_names)
    seg_names = assign_segment_names(profiles)

    print("\n    Segment Profiles:")
    print("    " + "-" * 55)
    for seg_id, prof in profiles.items():
        name = seg_names[seg_id]
        print(f"    Segment {seg_id} [{name}]")
        print(f"      Count: {prof['count']} ({prof['pct']}%)")
        print(f"      Avg Income: ${prof['mean_income']:,.0f}")
        print(f"      Avg Credit Score: {prof['mean_credit_score']}")
        print(f"      Avg Employment Years: {prof['mean_employment_years']}")
        print(f"      Debt-to-Income: {prof['mean_debt_to_income']}")
        print(f"      Loan History Count: {prof['mean_loan_history_count']}")
        print(f"      Home Ownership Rate: {prof['home_ownership_rate']:.1%}")
        print(f"      Verified Income Rate: {prof['verified_income_rate']:.1%}")
        print()

    # 6. Train supervised classifier
    print("[6/6] Training RandomForest classifier on cluster labels...")
    clf, acc, report, splits = train_classifier(X_feat, labels)
    print(f"    Test accuracy: {acc:.4f}")
    print(f"    Classification Report:")
    for label, metrics in report.items():
        if label in ['0', '1', '2', '3']:
            name = seg_names[int(label)]
            print(f"      Segment {label} [{name}]: precision={metrics['precision']:.3f}, "
                  f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}")

    # Feature importance
    importance = get_feature_importance(clf, feature_names)
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\n    Top 5 Feature Importances:")
    for feat, imp in sorted_imp[:5]:
        print(f"      {feat}: {imp:.4f}")

    # Save artifacts
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)

    joblib.dump(km, 'models/kmeans_model.pkl')
    joblib.dump(clf, 'models/classifier_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("    Saved: models/kmeans_model.pkl")
    print("    Saved: models/classifier_model.pkl")
    print("    Saved: models/scaler.pkl")

    results = {
        'n_clusters': best_k,
        'silhouette_score': round(sil, 4),
        'inertia': round(float(km.inertia_), 2),
        'classifier_accuracy': round(acc, 4),
        'segment_profiles': {str(k): {kk: vv for kk, vv in v.items()} for k, v in profiles.items()},
        'segment_names': {str(k): v for k, v in seg_names.items()},
        'feature_importance': {k: round(v, 4) for k, v in importance.items()},
        'k_range_silhouettes': {str(k + 2): round(v, 4) for k, v in enumerate(silhouettes)}
    }

    with open('reports/segmentation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("    Saved: reports/segmentation_results.json")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return results


if __name__ == '__main__':
    results = run()