import pandas as pd
import numpy as np

def compute_features(df):
    X = df[["income", "credit_score", "employment_years",
            "debt_to_income", "loan_history_count", "age",
            "home_ownership", "verified_income"]].copy()

    X["income_per_loan"] = df["income"] / (df["loan_history_count"] + 1)
    X["credit_per_year"] = df["credit_score"] / (df["employment_years"] + 1)
    X["debt_burden"] = df["debt_to_income"] * df["loan_history_count"]
    X["stability_score"] = df["employment_years"] * df["home_ownership"] + df["verified_income"]
    X["credit_age_ratio"] = df["credit_score"] / (df["age"] - 17)
    X["income_stability"] = df["income"] * (df["employment_years"] + 1)

    return X

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    df = generate_synthetic_data()
    X = compute_features(df)
    print("Features shape:", X.shape)
    print(X.describe())