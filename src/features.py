import pandas as pd
import numpy as np

FEATURE_COLS = ["income", "credit_score", "employment_years",
                "debt_to_income", "loan_history_count", "age",
                "home_ownership", "verified_income"]

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}

def build_features(df):
    X = df[FEATURE_COLS].copy()

    # RFM-style features
    X["income_per_loan"] = df["income"] / (df["loan_history_count"] + 1)
    X["income_per_age"] = df["income"] / (df["age"] + 1)

    # Behavioral
    X["credit_per_employment_year"] = df["credit_score"] / (df["employment_years"] + 1)
    X["loan_density"] = df["loan_history_count"] / (df["age"] - 17 + 1)
    X["DTI_credit_interaction"] = df["debt_to_income"] * df["credit_score"] / 100

    # Stability
    X["employment_age_ratio"] = df["employment_years"] / (df["age"] - 17 + 1)
    X["income_stability"] = df["verified_income"] * (1 - df["debt_to_income"])
    X["ownership_verified"] = df["home_ownership"] * df["verified_income"]

    return X

def get_feature_names():
    base = FEATURE_COLS
    derived = ["income_per_loan", "income_per_age", "credit_per_employment_year",
               "loan_density", "DTI_credit_interaction", "employment_age_ratio",
               "income_stability", "ownership_verified"]
    return base + derived

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data(500)
    X = build_features(df)
    print(X.shape)
    print(X.columns.tolist())
