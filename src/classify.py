"""Train RandomForestClassifier on cluster labels to predict segment from application features."""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_classifier(X, y, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Split, train RandomForest, evaluate; return metrics + model + report."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Feature importance
    importance = dict(zip(X.columns.tolist(), clf.feature_importances_.tolist()))

    return {
        "accuracy": float(acc),
        "classification_report": report,
        "feature_importance": importance,
        "test_size": test_size,
        "train_size": int(len(X_train)),
        "test_size_n": int(len(X_test)),
    }


def predict_segment(clf, X: list[dict]) -> list[int]:
    """Given a trained classifier and input feature dicts, return predicted segments."""
    import pandas as pd

    df = pd.DataFrame(X)
    return clf.predict(df).tolist()