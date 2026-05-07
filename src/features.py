import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["income_per_employment_year"] = df["income"] / (df["employment_years"] + 0.01)

    df["income_age_ratio"] = df["income"] / (df["age"] + 1)

    df["credit_x_income"] = df["credit_score"] * df["income"] / 100000

    df["employment_stability"] = df["employment_years"] / (df["age"] - 18 + 1)

    df["debt_burden"] = df["debt_to_income"] * (1 - df["home_ownership"] * 0.2)

    df["verified_flag"] = df["verified_income"]

    df["loan_density"] = df["loan_history_count"] / (df["age"] - 17)

    feature_cols = [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
        "income_per_employment_year",
        "income_age_ratio",
        "credit_x_income",
        "employment_stability",
        "debt_burden",
        "loan_density",
    ]

    return df[feature_cols]