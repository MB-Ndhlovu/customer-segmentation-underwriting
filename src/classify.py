import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk'
}

def train_classifier(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=list(SEGMENT_NAMES.values()))

    return clf, acc, report, (X_test, y_test)

def feature_importance(clf, feature_names):
    importances = clf.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    return ranked

if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import build_features
    from segment import assign_segment_names, run_clustering, profile_segments
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data()
    X = build_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km, labels, sil = run_clustering(X_scaled)
    profiles = profile_segments(X, labels)
    y = assign_segment_names(labels, X)

    clf, acc, report, _ = train_classifier(X, y)
    print(f"Test accuracy: {acc:.4f}")
    print(report)
    print("\nFeature importances:")
    for feat, imp in feature_importance(clf, X.columns):
        print(f"  {feat:30s} {imp:.4f}")

    joblib.dump(clf, 'models/classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')