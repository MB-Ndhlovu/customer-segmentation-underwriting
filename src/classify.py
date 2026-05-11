import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def train_classifier(X: pd.DataFrame, y: np.ndarray, seed: int = 42):
    """Train RandomForestClassifier on cluster labels, evaluate with hold-out."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Feature importances
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {
        "model": clf,
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importances": imp.to_dict(),
    }


def predict_segment(clf, X: pd.DataFrame) -> np.ndarray:
    """Predict cluster segment for new application data."""
    return clf.predict(X)