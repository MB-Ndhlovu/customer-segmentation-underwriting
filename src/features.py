"""Feature engineering for underwriting segmentation."""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the customer DataFrame.

    RFM-inspired features (Recency approximated via loan history depth):
        - loan_velocity: loan_history_count / years employed
        - income_per_employment_year: income / max(1, employment_years)

    Behavioral features:
        - credit_to_income_ratio: credit_score / income (scaled)
        - employment_stability: tanh(employment_years / 10)  # 0-1 scale

    Stability features:
        - dti_band: ordinal bands for debt-to-income risk buckets
        - verified_bonus: +1 when income is verified (stronger underwriting signal)
    """
    df = df.copy()

    # RFM-inspired
    df["loan_velocity"] = df["loan_history_count"] / df["employment_years"].clip(lower=0.1)
    df["income_per_year"] = df["income"] / df["employment_years"].clip(lower=0.5)

    # Behavioral
    df["credit_income_ratio"] = df["credit_score"] / (df["income"] / 10_000)
    df["employment_stability"] = np.tanh(df["employment_years"] / 10)

    # Stability bands
    def dti_band(dti):
        if dti < 0.20:
            return 0
        elif dti < 0.35:
            return 1
        elif dti < 0.50:
            return 2
        else:
            return 3

    df["dti_band"] = df["debt_to_income"].apply(dti_band)
    df["verified_bonus"] = df["verified_income"]

    return df


def get_feature_cols() -> list[str]:
    """Return the list of features used for clustering and classification."""
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
        "loan_velocity",
        "income_per_year",
        "credit_income_ratio",
        "employment_stability",
        "dti_band",
        "verified_bonus",
    ]