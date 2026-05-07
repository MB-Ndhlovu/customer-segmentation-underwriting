"""Supervised classification of cluster segments via RandomForest."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def train_classifier(X: pd.DataFrame, y: np.ndarray) -> RandomForestClassifier:
    """Train RandomForest on cluster labels; return fitted model."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled, y)

    cv_scores = cross_val_score(
        clf, X_scaled, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    )

    print(f"RandomForest CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"Per-fold scores: {[round(s, 4) for s in cv_scores]}")

    return clf, scaler


def predict_segment(clf: RandomForestClassifier, scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    """Predict cluster labels for new application data."""
    X_scaled = scaler.transform(X)
    return clf.predict(X_scaled)