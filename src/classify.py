"""Train RandomForestClassifier to predict segment from application features."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json


def train_classifier(X: pd.DataFrame, labels: np.ndarray, feature_names: list):
    """Train/evaluate RandomForest on cluster labels."""
    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_names], labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Feature importance
    importance = dict(zip(feature_names, clf.feature_importances_.round(4)))

    return {
        "accuracy": round(float(acc), 4),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_importance": importance,
    }, clf


def predict_segment(clf, X: pd.DataFrame, feature_names: list) -> np.ndarray:
    """Predict segment labels for new application data."""
    return clf.predict(X[feature_names])


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, get_feature_names
    from segment import fit_kmeans
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data(5000)
    X = build_features(df)
    feature_names = get_feature_names()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_names])

    _, labels, _ = fit_kmeans(X_scaled, n_clusters=4)
    results, clf = train_classifier(X, labels, feature_names[:8])  # raw features only

    print("Classifier results:", results)
    print("\nTop features:", sorted(results["feature_importance"].items(), key=lambda x: -x[1])[:5])