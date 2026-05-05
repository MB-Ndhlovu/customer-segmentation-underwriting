"""Train supervised classifier on cluster labels for production inference."""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

SEGMENT_NAMES = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}


FEATURE_COLS = [
    "income", "credit_score", "employment_years", "debt_to_income",
    "loan_history_count", "age", "home_ownership_enc", "verified_income_enc",
]


def train_classifier(df: pd.DataFrame, labels: np.ndarray, test_size: float = 0.2) -> dict:
    """Train RandomForest on cluster labels and return model + metrics."""
    X = df[FEATURE_COLS].values
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, target_names=list(SEGMENT_NAMES.values()), output_dict=True)

    importances = {
        FEATURE_COLS[i]: round(float(clf.feature_importances_[i]), 4)
        for i in range(len(FEATURE_COLS))
    }

    results = {
        "accuracy": round(acc, 4),
        "classification_report": report,
        "feature_importances": importances,
    }

    return {
        "results": results,
        "clf": clf,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def classify_new_application(clf, scaler, app_features: dict) -> dict:
    """Classify a new loan application. Returns segment prediction and probabilities."""
    import numpy as np
    X = np.array([[
        app_features["income"],
        app_features["credit_score"],
        app_features["employment_years"],
        app_features["debt_to_income"],
        app_features["loan_history_count"],
        app_features["age"],
        1 if app_features["home_ownership"] == "own" else 0,
        1 if app_features["verified_income"] == "verified" else 0,
    ]])
    X_scaled = scaler.transform(X)
    pred = clf.predict(X_scaled)[0]
    proba = clf.predict_proba(X_scaled)[0]
    return {
        "predicted_segment": SEGMENT_NAMES[pred],
        "segment_id": int(pred),
        "probabilities": {SEGMENT_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)},
    }


if __name__ == "__main__":
    from data_loader import generate_customers
    from features import build_features
    from segment import run_segmentation

    df = generate_customers(5000)
    df = build_features(df)
    seg_out = run_segmentation(df)
    clf_out = train_classifier(df, seg_out["labels"])

    print("Accuracy:", clf_out["results"]["accuracy"])
    print("\nFeature Importances:")
    for feat, imp in sorted(clf_out["results"]["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp}")

    print("\nClassification Report:")
    print(classification_report(
        clf_out["y_test"], clf_out["y_pred"],
        target_names=list(SEGMENT_NAMES.values())
    ))