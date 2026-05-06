"""
Full pipeline: load data → engineer features → cluster → classify → save results.
"""

from sklearn.metrics import silhouette_score
import os, json, joblib
from src.data_loader import load_customer_data
from src.features import compute_features
from src.segment import cluster_customers, find_optimal_k, profile_segments
from src.classify import train_classifier

REPORT_DIR = '/home/workspace/Projects/customer-segmentation-underwriting/reports'
os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 60)
print("CUSTOMER SEGMENTATION PIPELINE — Underwriting")
print("=" * 60)

# 1. Load data
print("\n[1] Loading / generating customer data...")
df = load_customer_data(n=5000)
print(f"  Dataset shape: {df.shape}")
print(f"  Segment distribution:\n{df['segment_label'].value_counts().sort_index().to_string()}")

# 2. Feature engineering
print("\n[2] Engineering features...")
X, scaler, feature_names = compute_features(df)
print(f"  Feature matrix shape: {X.shape}")

# 3. Clustering
print("\n[3] Running KMeans clustering (k=4)...")
labels, kmeans, inertia = cluster_customers(X, n_clusters=4)
sil_score = silhouette_score(X, labels)
print(f"  Inertia: {inertia:.0f}")
print(f"  Silhouette Score: {sil_score:.4f}")

print("\n[4] Silhouette analysis (k=2..8)...")
sil_dict = find_optimal_k(X)

# 5. Profile segments
print("\n[5] Profiling segments...")
profiles = profile_segments(df, labels)

# 6. Train classifier
print("\n[6] Training RandomForest classifier...")
clf, acc, y_test, y_pred = train_classifier(X, labels)

# 7. Save artifacts
print("\n[7] Saving artifacts...")
joblib.dump(kmeans, f'{REPORT_DIR}/kmeans_model.pkl')
joblib.dump(scaler, f'{REPORT_DIR}/feature_scaler.pkl')
joblib.dump(clf, f'{REPORT_DIR}/segment_classifier.pkl')

# Save JSON report
segment_names = {0: 'Mass Market', 1: 'Rising Prime', 2: 'Established Prime', 3: 'Subprime High-Risk'}
results = {
    'silhouette_score': round(sil_score, 4),
    'inertia': round(inertia, 2),
    'n_clusters': 4,
    'classification_accuracy': round(acc, 4),
    'segment_profiles': {
        int(c): {k: round(v, 4) for k, v in row.items() if k != 'cluster'}
        for c, row in profiles.iterrows()
    },
    'segment_names': segment_names,
    'feature_names': feature_names,
    'optimal_k_analysis': {str(k): round(v, 4) for k, v in sil_dict.items()}
}

with open(f'{REPORT_DIR}/segmentation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"  Saved: kmeans_model.pkl, feature_scaler.pkl, segment_classifier.pkl")
print(f"  Saved: segmentation_results.json")
print(f"  Saved: elbow_silhouette.png")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print(f"  Silhouette Score: {sil_score:.4f}")
print(f"  Classification Accuracy: {acc:.4f}")
print("=" * 60)