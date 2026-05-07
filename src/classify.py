"""Train RandomForestClassifier on cluster labels for segment prediction."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import joblib


SEGMENT_NAMES = {
    0: 'Mass Market',
    1: 'Rising Prime',
    2: 'Established Prime',
    3: 'Subprime High-Risk',
}


def train_classifier(df: pd.DataFrame, labels: np.ndarray, feature_cols: list, seed: int = 42):
    """Train RandomForest on cluster labels; return model and metrics."""

    X = df[feature_cols].values
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        class_weight='balanced',
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    print(f"\n=== Supervised Classification ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=list(SEGMENT_NAMES.values())))

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_,
    }).sort_values('importance', ascending=False)

    print(f"\nTop Feature Importances:\n{importance.head(10).to_string(index=False)}")

    return clf, acc, f1, importance


def save_classifier(clf, out_path: str):
    joblib.dump(clf, out_path)
    print(f"Model saved to {out_path}")