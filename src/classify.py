"""Train RandomForestClassifier on cluster labels for real-time segment prediction."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def train_classifier(X: pd.DataFrame, y: np.ndarray, test_size=0.2, random_state=42):
    """Train and evaluate RandomForest on cluster labels."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")

    return clf, {
        "test_accuracy": round(acc, 4),
        "cv_mean_accuracy": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
    }


def get_feature_importance(clf, feature_names):
    """Return sorted feature importance dataframe."""
    imp = pd.DataFrame({
        "feature": feature_names,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp


def predict_segment(clf, application_features: pd.DataFrame):
    """Predict segment label for a new application."""
    pred = clf.predict(application_features)
    proba = clf.predict_proba(application_features)
    return pred, proba