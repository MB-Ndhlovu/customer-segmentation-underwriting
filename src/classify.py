"""Train RandomForest classifier to predict segment from application features."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import json


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def get_feature_cols() -> list:
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
    ]


def train_classifier(df: pd.DataFrame, labels: np.ndarray, seed: int = 42):
    """Train RandomForest on cluster labels and return model + metrics."""
    feature_cols = get_feature_cols()
    X = df[feature_cols].values
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Feature importances
    importances = dict(zip(feature_cols, clf.feature_importances_.round(4).tolist()))

    # Per-class metrics
    report = classification_report(
        y_test, y_pred, target_names=list(SEGMENT_NAMES.values()), output_dict=True
    )

    return {
        "model": clf,
        "accuracy": round(acc, 4),
        "feature_importances": importances,
        "classification_report": {
            k: v for k, v in report.items() if k in SEGMENT_NAMES.values()
        },
        "test_size": len(y_test),
        "train_size": len(y_train),
    }


def predict_segment(model: RandomForestClassifier, application: dict) -> dict:
    """Predict segment for a single loan application."""
    feature_cols = get_feature_cols()
    X = np.array([[application[col] for col in feature_cols]])
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].round(4).tolist()

    return {
        "predicted_segment_id": pred,
        "predicted_segment_name": SEGMENT_NAMES[pred],
        "probabilities": {
            SEGMENT_NAMES[i]: round(p, 4) for i, p in enumerate(proba)
        },
    }


if __name__ == "__main__":
    from src.data_loader import generate_customer_data
    from src.features import build_features
    from src.segment import run_segmentation

    df = generate_customer_data()
    seg_result = run_segmentation(df, n_clusters=4)
    labels = seg_result["labels"]

    result = train_classifier(df, labels)
    print(f"Classifier Accuracy: {result['accuracy']}")
    print("Feature Importances:", result["feature_importances"])