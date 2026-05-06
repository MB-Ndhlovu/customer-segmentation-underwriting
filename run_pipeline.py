import json
import sys
from src.data_loader import generate_customer_data
from src.features import build_features, scale_features
from src.segment import find_optimal_k, run_segmentation
from src.classify import train_classifier

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION FOR UNDERWRITING — PIPELINE")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Generating synthetic customer data (5000 rows)...")
    df = generate_customer_data(n_samples=5000)
    print(f"  Shape: {df.shape}")
    print(f"  Segments:\n{df['segment_name'].value_counts().to_string()}")

    # 2. Feature engineering
    print("\n[2/5] Building features...")
    X, feature_cols = build_features(df)
    print(f"  Features: {feature_cols}")

    # 3. Scale
    print("\n[3/5] Scaling features...")
    X_scaled, scaler = scale_features(X)
    print(f"  Scaled shape: {X_scaled.shape}")

    # 4. Clustering
    print("\n[4/5] KMeans clustering (k=4)...")
    result = run_segmentation(X_scaled, X, feature_cols, n_clusters=4)
    print(f"  Silhouette Score: {result['silhouette_score']:.4f}")
    print("\n  Segment Profiles:")
    for seg, stats in result['profiles'].items():
        print(f"  [{seg}]")
        print(f"    income=${stats['mean_income']:,.0f}  credit={stats['mean_credit_score']:.0f}"
              f"  DTI={stats['mean_debt_to_income']:.2f}  emp_yrs={stats['mean_employment_years']:.1f}"
              f"  loans={stats['mean_loan_history_count']:.1f}  age={stats['mean_age']:.0f}"
              f"  owned={stats['home_ownership_rate']:.0%}  verified={stats['verified_income_rate']:.0%}")

    # 5. Classification
    print("\n[5/5] Training RandomForest classifier...")
    clf_result = train_classifier(X.values, result['labels'])
    print(f"  Accuracy: {clf_result['accuracy']:.4f}")
    print(f"  Train: {clf_result['train_size']}  Test: {clf_result['test_size']}")
    print("\n  Top 5 Feature Importances:")
    sorted_fi = sorted(clf_result['feature_importance'].items(), key=lambda x: -x[1])
    for fname, fimp in sorted_fi[:5]:
        print(f"    {fname}: {fimp:.4f}")

    # Save results
    output = {
        'silhouette_score': result['silhouette_score'],
        'segment_profiles': result['profiles'],
        'classifier_accuracy': clf_result['accuracy'],
        'feature_importance': clf_result['feature_importance'],
        'n_samples': 5000,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
    }
    out_path = '/home/workspace/Projects/customer-segmentation-underwriting/reports/segmentation_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {out_path}")

    summary = (
        f"Segmented 5000 customers into 4 groups.\n"
        f"Silhouette: {result['silhouette_score']:.4f} | "
        f"RF Accuracy: {clf_result['accuracy']:.4f}\n"
        f"Segments: Mass Market, Rising Prime, Established Prime, Subprime High-Risk"
    )
    print("\n" + "=" * 60)
    print(summary)
    print("=" * 60)
    return output

if __name__ == '__main__':
    main()