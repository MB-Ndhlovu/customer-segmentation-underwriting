import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


def train_segment_classifier(X: np.ndarray, y: np.ndarray,
                              test_size: float = 0.2,
                              random_state: int = 42) -> dict:
    """
    Train a RandomForest to predict segment labels from application features.
    This enables real-time segment inference on new applications.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    feature_names = ["income", "credit_score", "employment_years", "debt_to_income",
                     "loan_history_count", "age", "home_ownership", "verified_income"]

    return {
        "model": clf,
        "accuracy": float(acc),
        "classification_report": report,
        "feature_importances": {
            name: float(imp) for name, imp in zip(feature_names, clf.feature_importances_)
        },
        "X_test_shape": X_test.shape,
        "y_test_distribution": {
            str(k): int(v) for k, v in pd.Series(y_test).value_counts().items()
        }
    }


def predict_segment(clf, X: np.ndarray) -> np.ndarray:
    return clf.predict(X)


def save_model(clf, path: str):
    joblib.dump(clf, path)


def load_model(path: str):
    return joblib.load(path)


if __name__ == "__main__":
    from data_loader import generate_synthetic_data, get_feature_matrix, scale_features
    from features import get_engineered_features

    df = generate_synthetic_data()
    X_raw = get_feature_matrix(df)
    X_scaled, _ = scale_features(X_raw)

    # Simulate labels (using true_segment as proxy — in practice from clustering)
    result = train_segment_classifier(X_raw, df["true_segment"].values)

    print(f"RandomForest Accuracy: {result['accuracy']:.4f}")
    print("\nFeature Importances:")
    for feat, imp in sorted(result["feature_importances"].items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")