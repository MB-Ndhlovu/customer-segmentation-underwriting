import numpy as np
import pandas as pd


def compute_rfm_features(df):
    """RFM-style features derived from credit behavior."""
    df = df.copy()

    # Recency proxy: inverse of loan recency (assumed uniform here as no timeline)
    # Use credit score as a proxy for engagement quality
    df["recency_score"] = (df["credit_score"] - 300) / (850 - 300)

    # Frequency: loan history normalized
    df["frequency_score"] = df["loan_history_count"] / df["loan_history_count"].max()

    # Monetary: income per year of employment (stability proxy)
    df["monetary_score"] = df["income"] / (df["employment_years"] + 1)

    return df


def compute_behavioral_features(df):
    """Behavioral risk indicators."""
    df = df.copy()

    # High DTI flag
    df["high_dti_flag"] = (df["debt_to_income"] > 0.36).astype(int)

    # Credit utilization proxy (lower credit score relative to income = higher utilization)
    df["credit_to_income_ratio"] = df["credit_score"] / (df["income"] / 1000 + 1)

    # Income stability index
    df["income_stability"] = df["employment_years"] / (df["age"] - 18 + 1)
    df["income_stability"] = df["income_stability"].clip(0, 1)

    # Loan density (loans per year of employment)
    df["loan_density"] = df["loan_history_count"] / (df["employment_years"] + 0.5)

    return df


def compute_stability_features(df):
    """Employment and residency stability features."""
    df = df.copy()

    # Employment tenure bucket
    df["employment_tenure_bucket"] = pd.cut(
        df["employment_years"],
        bins=[-1, 2, 5, 10, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Income per age (young but high income = exceptional)
    df["income_age_ratio"] = df["income"] / df["age"]

    # Home + verified income combo
    df["home_verified_combo"] = df["home_ownership_status"] * df["verified_income"]

    return df


def build_features(df):
    """Apply all feature engineering."""
    df = compute_rfm_features(df)
    df = compute_behavioral_features(df)
    df = compute_stability_features(df)
    return df