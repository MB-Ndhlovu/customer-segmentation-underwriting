import numpy as np
import pandas as pd


def build_features(df):
    """
    Engineer RFM, behavioral, and stability features from raw customer data.

    RFM-style:
      - income_per_employment_year
      - credit_score_per_age_decade

    Behavioral:
      - loan_density (loans per employment year)
      - debt_burden_score (DTI * loan_count)

    Stability:
      - employment_stability (years / age_ratio)
      - home_income_interaction
    """
    X = df.copy()

    # RFM-style features
    X["income_per_emp_year"] = X["income"] / (X["employment_years"] + 0.5)
    X["credit_per_age_decade"] = X["credit_score"] / ((X["age"] + 1) / 10)

    # Behavioral
    X["loan_density"] = X["loan_history_count"] / (X["employment_years"] + 0.5)
    X["debt_burden_score"] = X["debt_to_income"] * (X["loan_history_count"] + 1)

    # Stability
    X["employment_stability"] = X["employment_years"] / (X["age"] + 1)
    X["home_income_interaction"] = X["home_ownership"] * np.log1p(X["income"])

    # Verification signal
    X["verification_bonus"] = X["verified_income"] * X["credit_score"] / 850

    return X


def get_feature_columns():
    """Columns used for clustering and classification."""
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
        "income_per_emp_year",
        "credit_per_age_decade",
        "loan_density",
        "debt_burden_score",
        "employment_stability",
        "home_income_interaction",
        "verification_bonus",
    ]
