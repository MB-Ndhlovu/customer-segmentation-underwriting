import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


SEGMENT_NAMES = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}


def train_classifier(df: pd.DataFrame) -> tuple:
    feature_cols = [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_history_count", "age",
        "home_ownership", "verified_income",
    ]

    X = df[feature_cols]
    y = df["segment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, acc


def get_feature_importance(clf, feature_cols: list) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return imp.reset_index(drop=True)


if __name__ == "__main__":
    from src.data_loader import generate_customer_data
    from src.features import build_features
    from src.segment import assign_segments

    df = generate_customer_data(5000)
    df = build_features(df)
    df, sil, scaler, km = assign_segments(df)
    clf, acc = train_classifier(df)
    print(f"Classifier Accuracy: {acc:.4f}")
    imp = get_feature_importance(clf, [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_history_count", "age",
        "home_ownership", "verified_income",
    ])
    print(imp)
