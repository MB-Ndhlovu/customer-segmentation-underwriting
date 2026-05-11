import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def build_features(df):
    df = df.copy()

    # RFM-style features
    df["income_per_year_of_employment"] = df["income"] / (df["employment_years"] + 1)
    df["loan_density"] = df["loan_history_count"] / (df["age"] - 17)  # normalized by working life span
    df["income_per_age"] = df["income"] / df["age"]

    # Behavioral features
    df["credit_per_income"] = df["credit_score"] / (df["income"] / 1_000_000)
    df["active_borrower"] = (df["loan_history_count"] > 2).astype(int)
    df["high_loan_density"] = (df["loan_density"] > 0.15).astype(int)

    # Stability features
    df["income_stability_score"] = (
        (df["employment_years"] / (df["age"] - 17)) * 0.5 +
        (df["verified_income"].astype(float)) * 0.5
    )
    df["homeowner"] = (df["home_ownership"] == "own").astype(int)
    df["long_tenure"] = (df["employment_years"] >= 5).astype(int)

    return df


def prepare_for_clustering(df, feature_cols):
    X = df[feature_cols].copy()

    # Encode home_ownership
    le = LabelEncoder()
    X["home_ownership"] = le.fit_transform(X["home_ownership"])

    # Encode verified_income
    X["verified_income"] = X["verified_income"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, X, scaler