"""Feature engineering for customer segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[
        [
            "income",
            "credit_score",
            "employment_years",
            "debt_to_income",
            "loan_history_count",
            "age",
            "home_ownership",
            "verified_income",
        ]
    ].copy()

    # RFM-style features
    X["income_per_employment_year"] = df["income"] / (df["employment_years"] + 1)
    X["income_per_age"] = df["income"] / df["age"]
    X["credit_per_year_of_history"] = df["credit_score"] / (df["loan_history_count"] + 1)

    # Behavioral features
    X["loan_density"] = df["loan_history_count"] / (df["age"] - 18 + 1)  # loans per year since adulthood
    X["income_stability_proxy"] = df["verified_income"] * df["employment_years"]

    # Stability features
    X["debt_burden_score"] = df["debt_to_income"] * df["loan_history_count"]
    X["credit_to_debt_ratio"] = df["credit_score"] / (df["debt_to_income"] * 100 + 1)

    return X


def scale_features(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return pd.DataFrame(Xs, columns=X.columns), scaler