import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 10)):
    inertia = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    return inertia, silhouettes


def segment_customers(X_scaled: np.ndarray, n_clusters: int = 4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km


def profile_segments(df: pd.DataFrame, labels: np.ndarray, feature_cols: list) -> pd.DataFrame:
    df_labeled = df.copy()
    df_labeled["cluster"] = labels

    segment_names = {
        0: "Mass Market",
        1: "Rising Prime",
        2: "Established Prime",
        3: "Subprime High-Risk",
    }

    profiles = []
    for seg_id in sorted(df_labeled["cluster"].unique()):
        mask = df_labeled["cluster"] == seg_id
        row = {"segment_id": int(seg_id), "segment_name": segment_names.get(seg_id, f"Segment {seg_id}"), "count": int(mask.sum())}
        for col in feature_cols:
            row[f"{col}_mean"] = round(df_labeled.loc[mask, col].mean(), 4)
            row[f"{col}_std"] = round(df_labeled.loc[mask, col].std(), 4)
        profiles.append(row)

    return pd.DataFrame(profiles)


if __name__ == "__main__":
    from data_loader import generate_customer_data
    from features import build_features, get_feature_columns
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = generate_customer_data()
    X = build_features(df)
    feature_cols = get_feature_columns()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[feature_cols])

    inertia, silhouettes = find_optimal_k(X_scaled)
    print("K | Inertia | Silhouette")
    for i, k in enumerate(range(2, 10)):
        print(f"{k} | {inertia[i]:.1f} | {silhouettes[i]:.4f}")

    labels, km = segment_customers(X_scaled, n_clusters=4)
    sil = silhouette_score(X_scaled, labels)
    print(f"\nFinal silhouette (k=4): {sil:.4f}")
    print(f"Inertia (k=4): {km.inertia_:.1f}")

    profiles = profile_segments(X, labels, feature_cols)
    print("\nSegment Profiles:")
    print(profiles[["segment_id", "segment_name", "count", "income_mean", "credit_score_mean", "debt_to_income_mean"]])