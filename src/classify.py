import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
)


def train_classifier(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Train RandomForest on cluster labels; return model, train/test split, and metrics."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)

    metrics = {
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "precision_weighted": round(precision_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_te, y_pred, average="weighted", zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "classification_report": classification_report(
            y_te, y_pred, zero_division=0
        ),
    }

    return clf, (X_tr, X_te, y_tr, y_te), metrics


def get_feature_importance(clf: RandomForestClassifier, feature_names: list) -> dict:
    """Return sorted feature importance."""
    importances = clf.feature_importances_
    feat_imp = dict(zip(feature_names, np.round(importances, 4)))
    return dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
