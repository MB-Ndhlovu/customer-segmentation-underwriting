"""Train RandomForestClassifier to predict segment from application features."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json


SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk',
}


def train_classifier(X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.2, random_state: int = 42):
    """Train and evaluate a RandomForest classifier on cluster labels."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n=== Classifier Performance ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    target_names = [SEGMENT_NAMES[i] for i in sorted(np.unique(y))]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Feature importance
    print("\nTop Feature Importances:")
    feature_names = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership_enc', 'verified_income',
        'rfm_monetary', 'behavioral_dti', 'stability_tenure_score',
    ]
    importances = sorted(zip(feature_names, clf.feature_importances_),
                         key=lambda x: x[1], reverse=True)
    for fname, imp in importances:
        print(f"  {fname}: {imp:.4f}")

    return clf, acc


def predict_segment(clf, X: np.ndarray, feature_names: list) -> np.ndarray:
    """Predict segment labels for new application data."""
    return clf.predict(X)


if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import build_features
    from segment import fit_kmeans
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data()
    feat_df = build_features(df)

    feature_cols = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership_enc', 'verified_income',
        'rfm_monetary', 'behavioral_dti', 'stability_tenure_score',
    ]

    X = feat_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    _, labels = fit_kmeans(X_scaled, n_clusters=4)
    clf, acc = train_classifier(X_scaled, labels)

    print(f"\nClassifier trained with accuracy: {acc:.4f}")