import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def train_classifier(X: pd.DataFrame, y: pd.Series, feature_cols: list):
    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_cols], y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return clf, acc, report, importance


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, get_feature_columns
    from segment import segment_customers
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data()
    X = build_features(df)
    feature_cols = get_feature_columns()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_cols])
    labels, _ = segment_customers(X_scaled, n_clusters=4)

    clf, acc, report, importance = train_classifier(X, pd.Series(labels), feature_cols)
    print(f"Classification accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(pd.Series(labels[:len(X_test)]), clf.predict(X_test[:len(X_test)])) if 'X_test' in dir() else "Run full pipeline for report")
    print("\nFeature Importance (top 10):")
    print(importance.head(10))