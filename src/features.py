"""Feature engineering for customer segmentation."""

import numpy as np
import pandas as pd


def compute_rfm_features(df):
    """RFM-style features derived from income and loan history."""
    return pd.DataFrame({
        "income_per_employment_year": df["income"] / (df["employment_years"] + 0.5),
        "loan_density": df["loan_history_count"] / (df["age"] - 17 + 1),  # loans per eligible year
        "income_stability_ratio": df["income"] / (df["employment_years"] * 1000 + 1),
    })


def compute_behavioral_features(df):
    """Behavioral signals from debt and credit."""
    return pd.DataFrame({
        "dti_credit_product": df["debt_to_income"] * df["loan_history_count"],
        "credit_per_age": df["credit_score"] / df["age"],
        "verified_income_weight": df["verified_income"] * np.log1p(df["income"]),
        "ownership_income": df["home_ownership"] * np.log1p(df["income"]),
    })


def compute_stability_features(df):
    """Employment and residence stability indicators."""
    return pd.DataFrame({
        "employment_stability": df["employment_years"] / (df["age"] - 17 + 1),
        "tenure_income_interaction": df["employment_years"] * np.log1p(df["income"]),
        "verified_tenure": df["verified_income"] * df["employment_years"],
        "age_employment_gap": df["age"] - df["employment_years"] - 16,
    })


def build_feature_matrix(df, feature_cols):
    """Build full feature matrix including engineered features."""
    base = df[feature_cols].copy()

    rfm = compute_rfm_features(df)
    behavioral = compute_behavioral_features(df)
    stability = compute_stability_features(df)

    X = pd.concat([base, rfm, behavioral, stability], axis=1)
    return X


def get_all_feature_names():
    base = [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age", "home_ownership", "verified_income",
    ]
    engineered = [
        "income_per_employment_year", "loan_density", "income_stability_ratio",
        "dti_credit_product", "credit_per_age", "verified_income_weight", "ownership_income",
        "employment_stability", "tenure_income_interaction", "verified_tenure", "age_employment_gap",
    ]
    return base + engineered


if __name__ == "__main__":
    from data_loader import generate_customer_data, get_feature_columns
    df = generate_customer_data(5000)
    X = build_feature_matrix(df, get_feature_columns())
    print(f"Feature matrix shape: {X.shape}")
    print(X.describe())