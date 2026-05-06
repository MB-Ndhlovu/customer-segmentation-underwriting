"""Feature engineering for customer segmentation."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build RFM, behavioral, and stability features."""
    df = df.copy()

    # --- RFM-style features ---
    # Income-to-age ratio (proxy for income velocity)
    df["income_per_age"] = df["income"] / (df["age"] + 1)

    # Credit score normalized (0–1 scale)
    df["credit_score_norm"] = (df["credit_score"] - 300) / (850 - 300)

    # --- Behavioral features ---
    # Loan frequency rate (loans per year of employment)
    df["loan_frequency"] = df["loan_history_count"] / (df["employment_years"] + 1)

    # Debt burden severity
    df["debt_burden_flag"] = (df["debt_to_income"] > 0.36).astype(int)

    # Verified income premium (indicator of stability)
    df["income_verified_flag"] = df["verified_income"]

    # --- Stability features ---
    # Employment duration buckets (tenure security)
    df["tenure_short"] = (df["employment_years"] < 2).astype(int)
    df["tenure_long"] = (df["employment_years"] > 10).astype(int)

    # Home ownership as stability proxy
    df["homeowner"] = df["home_ownership"]

    # Age-based experience proxy
    df["experience_proxy"] = df["age"] - 22  # assumes first job at 22
    df["experience_proxy"] = df["experience_proxy"].clip(lower=0)

    # Credit score stability (inverse of loan frequency × DTI)
    df["credit_stability_score"] = df["credit_score_norm"] / (df["loan_frequency"] + 0.1)

    return df


def get_feature_columns() -> list:
    """Return the list of features used for clustering."""
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
        "income_per_age",
        "credit_score_norm",
        "loan_frequency",
        "debt_burden_flag",
        "tenure_short",
        "tenure_long",
        "homeowner",
        "experience_proxy",
        "credit_stability_score",
    ]


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None) -> tuple:
    """Scale features for clustering."""
    feature_cols = get_feature_columns()
    X = df[feature_cols].values

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


if __name__ == "__main__":
    from data_loader import generate_customer_data
    df = generate_customer_data()
    df_feat = build_features(df)
    print(df_feat.head())
    print(f"\nFeature columns: {get_feature_columns()}")