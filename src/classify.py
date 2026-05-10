import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}


def train_classifier(df, labels, feature_cols, test_size=0.2, random_state=42):
    """Train RandomForest on cluster labels."""
    X = df[feature_cols].values
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Feature importance
    importance = dict(zip(feature_cols, clf.feature_importances_.round(4)))

    report = classification_report(
        y_test, y_pred, target_names=[SEGMENT_NAMES[i] for i in range(4)], output_dict=True
    )

    return clf, acc, importance, report


def predict_segment(clf, df, feature_cols):
    """Predict segment for new applications."""
    X = df[feature_cols].values
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    return preds, probs