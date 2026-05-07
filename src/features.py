"""Feature engineering for customer segmentation."""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to customer dataframe."""
    X = df.copy()

    # --- RFM-style features ---
    # Recency proxy: younger customers in high-credit cohort -> higher recency
    X["credit_age_ratio"] = X["credit_score"] / (X["age"] + 1)

    # Frequency proxy: loan history count (already in dataset)
    X["loan_frequency_score"] = X["loan_history_count"] / (X["age"] - 17 + 1)  # normalized by possible borrowing years

    # Monetary proxy: income per employment year (income stability)
    X["income_per_emp_year"] = X["income"] / (X["employment_years"] + 0.5)

    # --- Behavioral features ---
    # Debt burden index
    X["debt_burden_index"] = X["debt_to_income"] * X["loan_history_count"]

    # Credit utilization estimate (proxy via credit score buckets)
    X["credit_utilization_proxy"] = (X["credit_score"] - 500) / 350  # 500-850 scale

    # High DTI flag
    X["high_dti_flag"] = (X["debt_to_income"] > 0.35).astype(int)

    # Many loans flag
    X["many_loans_flag"] = (X["loan_history_count"] > 3).astype(int)

    # --- Stability features ---
    # Employment stability score
    X["employment_stability"] = X["employment_years"] / (X["age"] - 17 + 1)

    # Income verification bonus
    X["verified_income_bonus"] = X["verified_income"] * X["income"] / 50000

    # Home ownership bonus (proxy for stability)
    X["homeowner"] = X["home_ownership_status"]

    # Young and high credit trajectory (Rising Prime signal)
    X["young_prime_signal"] = ((X["age"] < 35) & (X["credit_score"] > 700)).astype(int)

    # --- Final feature vector ---
    feature_cols = [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age", "home_ownership_status", "verified_income",
        "credit_age_ratio", "loan_frequency_score", "income_per_emp_year",
        "debt_burden_index", "credit_utilization_proxy", "high_dti_flag",
        "many_loans_flag", "employment_stability", "verified_income_bonus",
        "young_prime_signal",
    ]
    return X[feature_cols]


def get_feature_names():
    return [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age", "home_ownership_status", "verified_income",
        "credit_age_ratio", "loan_frequency_score", "income_per_emp_year",
        "debt_burden_index", "credit_utilization_proxy", "high_dti_flag",
        "many_loans_flag", "employment_stability", "verified_income_bonus",
        "young_prime_signal",
    ]