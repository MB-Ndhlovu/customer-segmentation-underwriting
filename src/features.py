"""Feature engineering for customer segmentation."""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer RFM, behavioral, and stability features.
    Returns DataFrame with original columns plus engineered features.
    """
    df = df.copy()

    # RFM-style features
    df["income_per_year"] = df["income"] / (df["employment_years"] + 1)
    df["credit_per_age"] = df["credit_score"] / df["age"]

    # Behavioral features
    df["loan_density"] = df["loan_history_count"] / (df["age"] - 18 + 1)
    df["has_history"] = (df["loan_history_count"] > 0).astype(int)

    # Stability features
    df["employment_stability"] = df["employment_years"] / df["age"]
    df["income_stability"] = df["verified_income"] * df["credit_score"] / 100

    # Debt burden flags
    df["high_dti"] = (df["debt_to_income"] > 0.36).astype(int)
    df["low_credit"] = (df["credit_score"] < 580).astype(int)

    return df


FEATURE_COLS = [
    "income", "credit_score", "employment_years", "debt_to_income",
    "loan_history_count", "age", "home_ownership", "verified_income",
    "income_per_year", "credit_per_age", "loan_density", "has_history",
    "employment_stability", "income_stability", "high_dti", "low_credit",
]