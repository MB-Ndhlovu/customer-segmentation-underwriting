"""Feature engineering: RFM, behavioral, and stability features."""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the customer dataframe."""
    df = df.copy()

    # Encode home_ownership
    home_map = {"own": 2, "rent": 1, "none": 0}
    df["home_ownership_enc"] = df["home_ownership"].map(home_map)

    # Income stability proxy: verified income + home ownership
    df["income_stability_score"] = (
        df["verified_income"] * 2 + df["home_ownership_enc"]
    )

    # Debt burden indicator
    df["high_dti_flag"] = (df["debt_to_income"] > 0.36).astype(int)

    # Credit quality bucket
    df["credit_bucket"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 850],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Employment tenure score
    df["employment_tenure_score"] = np.clip(df["employment_years"] / 10, 0, 1)

    # Loan burden ratio
    df["loan_burden_ratio"] = df["loan_history_count"] / (df["age"] - 17)
    df["loan_burden_ratio"] = df["loan_burden_ratio"].clip(0, 1)

    # Income per age year (normalised earning power)
    df["income_age_ratio"] = df["income"] / df["age"]

    # Log income for scaling (right-skewed)
    df["log_income"] = np.log1p(df["income"])

    return df


def get_feature_columns() -> list:
    """Return the list of features used for clustering / classification."""
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership_enc",
        "verified_income",
        "income_stability_score",
        "high_dti_flag",
        "credit_bucket",
        "employment_tenure_score",
        "loan_burden_ratio",
        "income_age_ratio",
        "log_income",
    ]