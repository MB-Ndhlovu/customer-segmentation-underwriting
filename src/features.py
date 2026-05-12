"""Feature engineering for customer segmentation."""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive RFM, behavioral, and stability features from raw customer data."""
    X = df.copy()

    # Stability features
    X["emp_income_ratio"] = X["employment_years"] / (X["income"] / 10000 + 1)
    X["credit_per_age"] = X["credit_score"] / (X["age"] - 17)
    X["DTI_stability"] = (0.35 - X["debt_to_income"]).clip(lower=0)

    # Behavioral features
    X["loan_density"] = X["loan_history_count"] / (X["age"] - 17 + 1)
    X["income_per_loan"] = X["income"] / (X["loan_history_count"] + 1)
    X["credit_utilization_proxy"] = (X["debt_to_income"] * X["income"]) / (X["credit_score"] + 1)

    # Verified income flags
    X["verified_strong"] = ((X["verified_income"] == 1) & (X["income"] > 60000)).astype(int)
    X["verified_with_credit"] = ((X["verified_income"] == 1) & (X["credit_score"] > 680)).astype(int)

    return X


def get_feature_names() -> list:
    """Return list of features used for clustering/classification."""
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
        "emp_income_ratio",
        "credit_per_age",
        "DTI_stability",
        "loan_density",
        "income_per_loan",
        "credit_utilization_proxy",
        "verified_strong",
        "verified_with_credit",
    ]


if __name__ == "__main__":
    from data_loader import generate_customer_data

    df = generate_customer_data(5000)
    X = build_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(X.describe().round(3))