"""Train supervised classifier on cluster labels."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_classifier(X: pd.DataFrame, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return clf, acc, X_test, y_test, preds


def get_feature_importance(clf, feature_names) -> pd.DataFrame:
    imp = clf.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp.round(4)})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def predict_segment(clf, X: pd.DataFrame):
    return clf.predict(X)