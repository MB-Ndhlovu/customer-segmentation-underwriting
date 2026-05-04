"""Feature engineering for underwriting customer data."""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create RFM, behavioral, and stability features from raw customer data.

    Returns a DataFrame with original columns plus engineered features,
    ready for clustering.
    """
    df = df.copy()

    # RFM-style recency proxy — employment stability as "recency" of stable income
    df["employment_recency"] = df["employment_years"].clip(upper=15)

    # Income robustness: income relative to debt obligation
    df["income_robustness"] = df["income"] / (df["debt_to_income"] * df["income"] + 1)

    # Behavioral: loan density — loans per year of employment
    df["loan_density"] = df["loan_history_count"] / (df["employment_years"] + 0.5)

    # Stability: credit-to-age ratio (normalized credit ambition)
    df["credit_age_ratio"] = df["credit_score"] / (df["age"] - 17)

    # Verified income signal
    df["verified_income_flag"] = df["verified_income"]

    # Home ownership as stability indicator (already 0/1)
    df["home_stability"] = df["home_ownership"]

    # Income per employment year — growth proxy
    df["income_per_tenure"] = df["income"] / (df["employment_years"] + 1)

    # Debt burden tier (binned)
    df["dti_tier"] = pd.cut(
        df["debt_to_income"],
        bins=[-np.inf, 0.2, 0.35, 0.5, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Credit score tier
    df["credit_tier"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 850],
        labels=[0, 1, 2, 3],
    ).astype(int)

    return df


def get_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the feature matrix used for KMeans clustering."""
    feature_cols = [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
        "income_per_tenure",
        "dti_tier",
        "credit_tier",
    ]
    return df[feature_cols]