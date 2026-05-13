"""Feature engineering for customer segmentation."""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the customer dataframe."""
    df = df.copy()

    # Home ownership → ordinal
    home_map = {"rent": 0, "mortgage": 1, "own": 2}
    df["home_ownership_enc"] = df["home_ownership"].map(home_map)

    # RFM-adjacent: income stability proxy via verified_income + employment tenure
    df["income_stability_score"] = (
        df["verified_income"].astype(float) * 0.4
        + (df["employment_years"] / 20.0).clip(0, 1) * 0.6
    )

    # Behavioral: loan density (loan_history per year of employment)
    df["loan_density"] = df["loan_history_count"] / (df["employment_years"] + 1)

    # Stability: debt burden flag
    df["high_dti_flag"] = (df["debt_to_income"] > 0.38).astype(int)

    # Credit strength bucket
    df["credit_bucket"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 850],
        labels=[0, 1, 2, 3],
    ).astype(int)

    # Employment stability ratio
    df["employment_stability"] = (df["employment_years"] / (df["age"] - 18)).clip(0, 1)

    # Income per age year (proxy for career trajectory)
    df["income_per_age"] = df["income"] / df["age"]

    return df


def get_feature_columns() -> list[str]:
    """Columns used for clustering / classification."""
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership_enc",
        "verified_income",
    ]