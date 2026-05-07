"""Execute the full customer segmentation pipeline."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, silhouette_score

from src.data_loader import generate_customer_data
from src.features import build_features, scale_features
from src.segment import find_optimal_k, fit_kmeans, profile_segments, SEGMENT_NAMES
from src.classify import train_classifier


def run_pipeline(n: int = 5000, n_clusters: int = 4, test_size: float = 0.2) -> dict:
    """
    Run the full segmentation + classification pipeline.

    Returns results dict for reporting.
    """
    print("=" * 60)
    print("CUSTOMER SEGMENTATION PIPELINE — Underwriting")
    print("=" * 60)

    # ── 1. Load data ────────────────────────────────────────────────────
    print("\n[1/5] Generating synthetic customer data...")
    df = generate_customer_data(n)
    print(f"  → {len(df)} records, {len(df.columns)} raw features")

    # ── 2. Feature engineering ─────────────────────────────────────────
    print("\n[2/5] Building features...")
    X = build_features(df)
    print(f"  → {len(X.columns)} total features (8 raw + 12 engineered)")

    X_scaled, scaler = scale_features(X)
    print("  → Features standardised")

    # ── 3. Clustering ───────────────────────────────────────────────────
    print("\n[3/5] Running KMeans clustering...")

    print("  Elbow + Silhouette analysis (k=2..9):")
    opt = find_optimal_k(X_scaled)
    for k, v in opt.items():
        print(f"    k={k}: inertia={v['inertia']:.0f}  silhouette={v['silhouette']:.4f}")

    labels, km = fit_kmeans(X_scaled, n_clusters=n_clusters)
    sil = silhouette_score(X_scaled, labels)
    print(f"\n  → k={n_clusters} silhouette={sil:.4f}")

    # ── 4. Segment profiling ────────────────────────────────────────────
    print("\n[4/5] Profiling segments...")
    profile = profile_segments(df, labels)

    segment_counts = pd.Series(labels).value_counts().sort_index()
    print(f"\n  Segment distribution:")
    for seg_id in sorted(profile.index):
        name = profile.loc[seg_id, 'segment_name']
        count = segment_counts.get(seg_id, 0)
        pct = count / n * 100
        print(f"    Cluster {seg_id} → {name:<25} ({count:>4} rows, {pct:.1f}%)")

    # Feature means per segment
    print("\n  Key feature means per segment:")
    key_feats = ['income', 'credit_score', 'debt_to_income', 'employment_years', 'age']
    print(profile[key_feats + ['segment_name']].to_string())

    # ── 5. Classification ──────────────────────────────────────────────
    print("\n[5/5] Training RandomForest classifier...")
    clf_result = train_classifier(df, labels, test_size=test_size)

    print(f"\n  → Test accuracy: {clf_result['accuracy']:.4f}")
    print(f"\n  Classification report:")
    report_dict = clf_result['report']
    for label_id in sorted(report_dict.keys()):
        if label_id not in ('accuracy', 'macro avg', 'weighted avg'):
            name = SEGMENT_NAMES.get(int(label_id), label_id)
            r = report_dict[label_id]
            print(f"    {label_id} {name:<25} "
                  f"precision={r['precision']:.4f}  "
                  f"recall={r['recall']:.4f}  "
                  f"f1={r['f1-score']:.4f}")

    print(f"\n  Feature importance:")
    for feat, imp in sorted(clf_result['feature_importance'].items(), key=lambda x: -x[1]):
        bar = '█' * int(imp * 50)
        print(f"    {feat:<25} {imp:.4f} {bar}")

    # Confusion matrix
    cm = clf_result['confusion_matrix']
    print(f"\n  Confusion matrix:")
    header = "".join([f"{SEGMENT_NAMES[i]:<22}" for i in range(n_clusters)])
    print(f"    {'':22}{header}")
    for i, row in enumerate(cm):
        print(f"    {SEGMENT_NAMES[i]:<22} {'  '.join(str(x).rjust(8) for x in row)}")

    # ── Save artifacts ──────────────────────────────────────────────────
    reports_dir = Path(__file__).parent / 'reports'
    reports_dir.mkdir(exist_ok=True)

    results = {
        'n_samples': n,
        'n_clusters': n_clusters,
        'silhouette_score': round(sil, 4),
        'segment_counts': {SEGMENT_NAMES[i]: int(segment_counts.get(i, 0))
                           for i in range(n_clusters)},
        'segment_profiles': profile.drop(columns=['segment_name', 'segment_label'],
                                         errors='ignore').to_dict(orient='index'),
        'classifier_accuracy': round(clf_result['accuracy'], 4),
        'feature_importance': {k: round(v, 4) for k, v in
                               clf_result['feature_importance'].items()},
        'confusion_matrix': cm,
        'classification_report': {
            k: {kk: round(float(vv), 4) for kk, vv in v.items()}
            if isinstance(v, dict) else round(float(v), 4)
            for k, v in report_dict.items()
        },
    }

    results_path = reports_dir / 'segmentation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"  Silhouette score: {sil:.4f}")
    print(f"  Classifier accuracy: {clf_result['accuracy']:.4f}")
    print(f"  Results saved to: {results_path}")
    print(f"  Model saved to:   {Path(__file__).parent / 'segment_classifier.joblib'}")
    print("=" * 60)

    # Save classifier
    import joblib
    joblib.dump(clf_result['model'], Path(__file__).parent / 'segment_classifier.joblib')

    return results


if __name__ == '__main__':
    results = run_pipeline(n=5000, n_clusters=4)