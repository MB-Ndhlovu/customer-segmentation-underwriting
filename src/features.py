"""Feature engineering for customer underwriting data."""
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer RFM, behavioral, and stability features."""
    X = pd.DataFrame()

    X["income"] = df["income"]
    X["credit_score"] = df["credit_score"]
    X["employment_years"] = df["employment_years"]
    X["debt_to_income"] = df["debt_to_income"]
    X["loan_history_count"] = df["loan_history_count"]
    X["age"] = df["age"]
    X["verified_income"] = df["verified_income"].astype(int)

    home_map = {"own": 2, "mortgage": 1, "rent": 0}
    X["home_ownership_score"] = df["home_ownership"].map(lambda h: home_map.get(h, 0))

    # RFM-style features
    X["income_per_age"] = df["income"] / df["age"].clip(lower=1)
    X["employment_stability"] = df["employment_years"] / df["age"].clip(lower=1)

    # Behavioral features
    X["credit_to_income_ratio"] = df["credit_score"] / df["income"].clip(lower=1) * 1000
    X["loan_density"] = df["loan_history_count"] / df["employment_years"].clip(lower=0.5)

    # Stability features
    X["debt_burden_flag"] = (df["debt_to_income"] > 0.35).astype(int)
    X["verified_income_flag"] = df["verified_income"].astype(int)
    X["home_ownership_flag"] = (df["home_ownership"] == "own").astype(int)

    return X