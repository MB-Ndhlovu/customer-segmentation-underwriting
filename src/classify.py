"""Train RandomForestClassifier on cluster labels for segment inference."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def train_classifier(df: pd.DataFrame, feature_cols: list) -> dict:
    """Train RF on cluster labels; return model, metrics, and feature importances."""
    X = df[feature_cols].values
    y = df["segment_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5)

    # Feature importances
    importances = dict(zip(feature_cols, clf.feature_importances_.round(4)))

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "test_accuracy": round(acc, 4),
        "cv_mean_accuracy": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "feature_importances": importances,
        "classification_report": report,
        "clf": clf,
        "scaler": scaler,
    }