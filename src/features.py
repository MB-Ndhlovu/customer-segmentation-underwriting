"""
Feature engineering for customer segmentation.
RFM features: Recency, Frequency, Monetary (adapted for lending context)
Behavioral features: credit utilization patterns, loan behavior
Stability features: employment tenure, income verification, home ownership
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw customer data.
    Returns DataFrame with all features ready for clustering.
    """
    f = pd.DataFrame()

    # --- RFM-adjacent features ---
    # Frequency: loan_history_count is already a proxy for frequency
    f["loan_frequency"] = df["loan_history_count"]

    # Monetary: income proxies spending capacity
    f["income_capacity"] = df["income"]

    # Recency proxy: younger borrowers with short employment = newer customers
    f["recency_proxy"] = 1 / (df["age"] + 1)

    # --- Behavioral features ---
    # Credit risk indicator
    f["credit_risk_score"] = 850 - df["credit_score"]

    # Debt burden
    f["debt_burden"] = df["debt_to_income"]

    # Loan density (loans per year of employment)
    f["loan_density"] = df["loan_history_count"] / (df["employment_years"] + 0.1)

    # Income stability ratio
    f["income_stability"] = df["verified_income"] * np.log1p(df["income"])

    # --- Stability features ---
    f["employment_stability"] = np.log1p(df["employment_years"])

    f["home_ownership_flag"] = df["home_ownership"]

    f["verified_income_flag"] = df["verified_income"]

    # Combined stability score
    f["stability_score"] = (
        f["home_ownership_flag"] * 0.3 +
        f["verified_income_flag"] * 0.3 +
        (f["employment_stability"] / 5) * 0.4
    ).clip(0, 1)

    # Age bracket feature
    f["age_bracket"] = pd.cut(df["age"], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]).astype(int)

    # Income tier
    f["income_tier"] = pd.cut(
        df["income"],
        bins=[0, 35000, 60000, 100000, np.inf],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return f


def get_feature_names() -> list:
    """Return list of all engineered feature names."""
    return [
        "loan_frequency", "income_capacity", "recency_proxy",
        "credit_risk_score", "debt_burden", "loan_density",
        "income_stability", "employment_stability", "home_ownership_flag",
        "verified_income_flag", "stability_score", "age_bracket", "income_tier"
    ]