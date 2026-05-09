"""
Supervised classification of customer segments.
Trains RandomForestClassifier on cluster labels from KMeans.
Enables real-time segment prediction from application features.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import json


def train_classifier(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Train RandomForest classifier on cluster labels.
    Returns model, metrics, and feature importances.
    """
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
        class_weight="balanced"
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    importances = {
        f"feature_{i}": float(v)
        for i, v in enumerate(clf.feature_importances_)
    }

    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

    return clf, metrics, importances


def predict_segment(clf, X):
    """Predict segment for new application data."""
    return clf.predict(X)


def cross_validate(X, y, cv: int = 5):
    """Run cross-validation on the classifier."""
    from sklearn.model_selection import cross_val_score
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        n_jobs=-1, class_weight="balanced"
    )
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "cv_folds": cv
    }


if __name__ == "__main__":
    from .data_loader import generate_customer_data, get_feature_columns
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data()
    X = df[get_feature_columns()].values
    y = df["segment_label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf, metrics, imp = train_classifier(X_scaled, y)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(classification_report(y, clf.predict(X_scaled)))