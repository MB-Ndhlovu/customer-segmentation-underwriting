import numpy as np
import pandas as pd

def compute_rfm_features(df):
    """RFM-style features derived from credit behaviour."""
    df = df.copy()
    df["income_per_loan"] = df["income"] / (df["loan_history_count"] + 1)
    df["credit_per_age"] = df["credit_score"] / df["age"]
    df["emp_stability"] = df["employment_years"] / (df["age"] - 18 + 1)
    return df

def compute_behavioral_features(df):
    """Loan history intensity and income robustness."""
    df = df.copy()
    df["loan_density"] = df["loan_history_count"] / (df["age"] - 18 + 1)
    df["income_credit_product"] = df["income"] * df["credit_score"] / 1e6
    df["high_dti_flag"] = (df["debt_to_income"] > 0.40).astype(int)
    return df

def compute_stability_features(df):
    """Employment and income verification stability."""
    df = df.copy()
    df["verified_income_flag"] = df["verified_income"]
    df["homeowner_verified"] = ((df["home_ownership"] == 1) & (df["verified_income"] == 1)).astype(int)
    df["tenure_score"] = np.minimum(df["employment_years"] / 10.0, 1.0)
    return df

def build_features(df):
    df = compute_rfm_features(df)
    df = compute_behavioral_features(df)
    df = compute_stability_features(df)
    return df