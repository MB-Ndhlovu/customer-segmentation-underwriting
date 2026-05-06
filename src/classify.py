import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import json


def train_segment_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Train a RandomForest to predict cluster labels from application features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_names], y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_sc, y_train)

    y_pred = clf.predict(X_test_sc)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "classification_report": classification_report(y_test, y_pred, target_names=[
            "Mass Market", "Rising Prime", "Established Prime", "Subprime High-Risk"
        ], output_dict=True),
        "feature_importances": {
            f: round(float(v), 4)
            for f, v in zip(feature_names, clf.feature_importances_)
        },
    }

    return clf, scaler, metrics


def save_artifacts(clf, scaler, metrics, profiles, output_path: str):
    artifacts = {
        "segment_profiles": profiles,
        "classifier_metrics": metrics,
    }
    with open(output_path, "w") as f:
        json.dump(artifacts, f, indent=2, default=float)