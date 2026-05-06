"""Train RandomForest classifier on cluster labels to predict segment from application features."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


def train_classifier(X, y, test_size=0.2, random_state=42):
    """Train RandomForest on cluster labels; return model, accuracy, report."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[
        'Mass Market', 'Rising Prime', 'Established Prime', 'Subprime High-Risk'
    ])

    return clf, acc, report


def save_model(clf, path):
    joblib.dump(clf, path)


def load_model(path):
    return joblib.load(path)