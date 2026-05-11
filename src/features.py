import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_loader import FEATURE_COLS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer RFM, behavioral, and stability features."""

    X = df[FEATURE_COLS].copy()

    # RFM-adjacent features
    X["income_per_employment_year"] = X["income"] / (X["employment_years"] + 1)
    X["credit_per_age"] = X["credit_score"] / X["age"]

    # Behavioral features
    X["loan_density"] = X["loan_history_count"] / (X["age"] - 17 + 1)  # loan rate since adulthood
    X["verified_income_flag"] = X["verified_income"]
    X["homeowner_flag"] = X["home_ownership_status"]

    # Stability features
    X["employment_stability"] = X["employment_years"] / X["age"]
    X["dti_risk"] = (X["debt_to_income"] > 0.36).astype(int)  # threshold for high DTI
    X["credit_utilization_proxy"] = (X["credit_score"] < 620).astype(int)

    return X


def scale_features(X: pd.DataFrame) -> tuple:
    """Standardise features and return scaler for later use."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler