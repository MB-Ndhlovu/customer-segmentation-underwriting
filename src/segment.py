import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


SEGMENT_NAMES = {0: "Mass Market", 1: "Rising Prime", 2: "Established Prime", 3: "Subprime High-Risk"}


def find_optimal_k(X_scaled: np.ndarray, k_range: range) -> tuple:
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    return inertias, silhouettes


def assign_segments(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    feature_cols = [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_history_count", "age",
        "home_ownership", "verified_income",
    ]

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["segment_label"] = km.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, df["segment_label"])
    return df, sil, scaler, km


def profile_segments(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_history_count", "age",
        "home_ownership", "verified_income",
    ]

    profiles = df.groupby("segment_label")[feature_cols].mean().round(2)
    profiles["count"] = df.groupby("segment_label").size()
    profiles["segment_name"] = profiles.index.map(SEGMENT_NAMES)
    return profiles.reset_index()


if __name__ == "__main__":
    from src.data_loader import generate_customer_data
    from src.features import build_features

    df = generate_customer_data(5000)
    df = build_features(df)
    df, sil, scaler, km = assign_segments(df)
    profiles = profile_segments(df)
    print(profiles)
    print(f"\nSilhouette Score: {sil:.4f}")
