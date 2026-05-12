import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def train_classifier(
    X: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train RandomForestClassifier on cluster labels.
    Returns trained model + metrics.
    """
    X_features = X[feature_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Feature importances
    importances = {
        feat: float(imp) for feat, imp in zip(feature_cols, clf.feature_importances_)
    }

    return {
        "model": clf,
        "accuracy": float(acc),
        "feature_importances": importances,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def predict_segment(clf: RandomForestClassifier, X: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Predict segments for new data."""
    return clf.predict(X[feature_cols])


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, get_feature_columns
    from segment import fit_kmeans
    from sklearn.preprocessing import StandardScaler

    df = generate_customer_data()
    X = build_features(df)
    feature_cols = get_feature_columns()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_cols])

    labels, _ = fit_kmeans(X_scaled, n_clusters=4)

    result = train_classifier(X, labels, feature_cols)
    print(f"Classifier Accuracy: {result['accuracy']:.4f}")
    print("\nFeature Importances:")
    for feat, imp in sorted(result["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")
