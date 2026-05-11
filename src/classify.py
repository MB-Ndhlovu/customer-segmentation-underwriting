"""Train RandomForest classifier on cluster labels."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score


def train_classifier(X: pd.DataFrame, labels: np.ndarray) -> dict:
    """Train-test split + RandomForest. Return metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    results = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "feature_importances": {
            feat: float(imp) for feat, imp in zip(X.columns, clf.feature_importances_)
        },
    }
    return results


def run_classification(X: pd.DataFrame, labels: np.ndarray) -> dict:
    metrics = train_classifier(X, labels)
    return metrics


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features
    from segment import fit_kmeans
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data(5000)
    X = build_features(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    labels, _ = fit_kmeans(X_scaled, n_clusters=4)

    res = run_classification(X, labels)
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"F1 Macro: {res['f1_macro']:.4f}")
    print("\nFeature importances:")
    for feat, imp in sorted(res["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")