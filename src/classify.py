from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def train_classifier(X, labels):
    """Train RandomForest to predict cluster labels from features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.25, random_state=42, stratify=labels
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return {
        'model': clf,
        'accuracy': round(acc, 4),
        'n_features': X.shape[1],
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


def feature_importance(clf, feature_names):
    imp = clf.feature_importances_
    ranking = sorted(zip(feature_names, imp), key=lambda x: -x[1])
    return {name: round(float(score), 4) for name, score in ranking}