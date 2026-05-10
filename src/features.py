"""Feature engineering for customer segmentation: RFM, behavioral, stability features."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "income",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "loan_history_count",
    "age",
    "home_ownership",
    "verified_income",
]

# --- RFM-style features ---
def rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # income per employment year — financial stability ratio
    df["income_per_emp_year"] = (df["income"] / (df["employment_years"] + 0.01)).round(2)
    # credit score normalised to 0-1
    df["credit_normalised"] = ((df["credit_score"] - 300) / (850 - 300)).round(4)
    return df


# --- Behavioral features ---
def behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # loan burden: how many loans relative to age (young person with many loans = risk)
    df["loan_burden"] = (df["loan_history_count"] / (df["age"] - 17 + 1)).round(4)
    # income stability proxy: employment years / age (higher = more stable career)
    df["career_stability"] = (df["employment_years"] / (df["age"] - 17 + 1)).round(4)
    # verified income flag multiplies credit quality
    df["verified_credit_quality"] = (df["credit_normalised"] * df["verified_income"]).round(4)
    return df


# --- Stability features ---
def stability_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # DTI risk buckets (high DTI = unstable)
    df["dti_risk"] = (df["debt_to_income"] * 100).round(2)
    # home ownership as stability anchor
    df["home_stability_anchor"] = df["home_ownership"]
    return df


# --- Derived features (no external data needed) ---
def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = rfm_features(df)
    df = behavioral_features(df)
    df = stability_features(df)
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return engineered feature matrix for clustering."""
    df = derive_features(df)
    feature_names = FEATURE_COLS + [
        "income_per_emp_year",
        "credit_normalised",
        "loan_burden",
        "career_stability",
        "verified_credit_quality",
        "dti_risk",
        "home_stability_anchor",
    ]
    return df[feature_names]


def scale_features(X: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler