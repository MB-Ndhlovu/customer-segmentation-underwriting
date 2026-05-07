"""Train supervised model on cluster labels — predict segment from application features."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix
)
import joblib
import json

# Feature names used at classification time (original 8, no engineered features needed)
CLASSIFY_FEATURE_NAMES = [
    'income', 'credit_score', 'employment_years', 'debt_to_income',
    'loan_history_count', 'age', 'home_ownership', 'verified_income',
]


def train_classifier(df: pd.DataFrame, labels: np.ndarray, test_size: float = 0.2,
                     random_state: int = 42) -> dict:
    """
    Train a RandomForestClassifier to predict segment labels from
    the 8 core application features.

    Returns dict with model, accuracy, classification_report, confusion_matrix.
    """
    X = df[CLASSIFY_FEATURE_NAMES].values
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'model': clf,
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm.tolist(),
        'feature_importance': dict(zip(CLASSIFY_FEATURE_NAMES, clf.feature_importances_.tolist())),
    }


def predict_segment(model, features_dict: dict) -> dict:
    """
    Predict segment for a single application.
    features_dict: {feature_name: value}
    Returns dict with predicted label, segment name, and probabilities.
    """
    from segment import SEGMENT_NAMES

    arr = np.array([[features_dict.get(f, 0) for f in CLASSIFY_FEATURE_NAMES]])
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr)[0]

    return {
        'predicted_label': int(pred),
        'segment_name': SEGMENT_NAMES[pred],
        'probabilities': {SEGMENT_NAMES[i]: round(p, 4) for i, p in enumerate(prob)},
    }


if __name__ == '__main__':
    from data_loader import generate_customer_data
    from features import build_features, scale_features
    from segment import run_clustering, profile_segments

    df = generate_customer_data(5000)
    X = build_features(df)
    X_scaled, scaler = scale_features(X)

    result = run_clustering(X_scaled, n_clusters=4)
    labels = result['labels']

    clf_result = train_classifier(df, labels)
    print(f"Test Accuracy: {clf_result['accuracy']:.4f}")
    print(classification_report(
        clf_result['report']['1'],  # only need summary
        digits=4
    ))
    print("Feature importance:", clf_result['feature_importance'])