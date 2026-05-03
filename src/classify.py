"""Train supervised classifier on cluster labels (RandomForest)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import json
import os


def train_segment_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """Train RandomForest on cluster labels; return metrics and model info."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )

    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).to_dict(orient="records")

    results = {
        "test_accuracy": round(accuracy, 4),
        "cv_mean_accuracy": round(cv_scores.mean(), 4),
        "cv_std_accuracy": round(cv_scores.std(), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importance": feature_importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return {
        "model": clf,
        "scaler": scaler,
        "metrics": results,
    }


def predict_segment(model, scaler, X: pd.DataFrame) -> np.ndarray:
    """Predict segment label for new application data."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def save_results(metrics: dict, output_path: str) -> None:
    """Save metrics dict to JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serializable = {
        k: v for k, v in metrics.items()
        if k not in ("model", "scaler")
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)


def run_classification(X: pd.DataFrame, y: pd.Series) -> dict:
    """Full classification pipeline."""
    result = train_segment_classifier(X, y)
    return result