"""Feature engineering for underwriting customer data."""
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to customer DataFrame."""
    df = df.copy()

    df["income_per_employment_year"] = (df["income"] / (df["employment_years"] + 1)).round(2)
    df["loan_per_year"] = (df["loan_history_count"] / (df["age"] - 17)).round(4)
    df["credit_to_dti_ratio"] = (df["credit_score"] / (df["debt_to_income"] * 100 + 1)).round(4)

    dti_quintiles = pd.qcut(df["debt_to_income"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    df["dti_risk_tier"] = dti_quintiles.astype(float)

    employment_bins = [0, 2, 5, 10, 20, 100]
    employment_labels = [1, 2, 3, 4, 5]
    df["employment_stability_tier"] = pd.cut(
        df["employment_years"], bins=employment_bins, labels=employment_labels, right=False
    ).astype(float).fillna(1)

    credit_bins = [0, 580, 670, 740, 800, 900]
    credit_labels = [1, 2, 3, 4, 5]
    df["credit_tier"] = pd.cut(
        df["credit_score"], bins=credit_bins, labels=credit_labels, right=False
    ).astype(float).fillna(1)

    income_per_age = df["income"] / df["age"]
    df["income_growth_proxy"] = (income_per_age / income_per_age.mean()).round(4)

    df["combined_risk_score"] = (
        (850 - df["credit_score"]) / 100 * 0.35
        + df["debt_to_income"] * 0.30
        + (1 / (df["employment_years"] + 1)) * 0.20
        + df["loan_history_count"] * 0.15
    ).round(4)

    return df