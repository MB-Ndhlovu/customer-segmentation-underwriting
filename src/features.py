"""Feature engineering for customer segmentation."""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RFM, behavioral, and stability feature sets.

    RFM analogue (no actual timestamps, so we derive proxy metrics):
      - F = loan_history_count (frequency of credit usage)
      - M = income (monetary capacity)
      - R = debt_to_income inverted (lower DTI = better recency/behavior)

    Behavioral:
      - credit_score normalised
      - income per employment year (career trajectory)
      - verified_income flag

    Stability:
      - employment_years / age  (career stability ratio)
      - home_ownership flag
    """
    X = pd.DataFrame()

    # RFM-analogues
    X["frequency"] = df["loan_history_count"]
    X["monetary"] = df["income"]
    X["recency_proxy"] = 1 - df["debt_to_income"]  # higher = better

    # Behavioral
    X["credit_normalised"] = (df["credit_score"] - 300) / 550  # [0,1] scale
    X["income_per_emp_year"] = df["income"] / (df["employment_years"] + 1)
    X["verified_income"] = df["verified_income"]

    # Stability
    X["stability_ratio"] = df["employment_years"] / (df["age"] - 17 + 1)
    X["home_ownership"] = df["home_ownership"]

    return X


def add_original_features(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Append raw original features alongside engineered ones."""
    for col in ["income", "credit_score", "employment_years",
                "debt_to_income", "loan_history_count", "age",
                "home_ownership", "verified_income"]:
        X[col] = df[col]
    return X