import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RFM-style features
    df["income_per_employment_year"] = (df["income"] / (df["employment_years"] + 1)).round(2)

    # Behavioral features
    df["loan_density"] = (df["loan_history_count"] / (df["age"] - 17 + 1)).round(4)
    df["credit_per_age"] = (df["credit_score"] / df["age"]).round(4)

    # Stability features
    df["employment_stability"] = (df["employment_years"] / (df["age"] - 17 + 1)).round(4)
    df["debt_burden_flag"] = (df["debt_to_income"] > 0.36).astype(int)

    return df


if __name__ == "__main__":
    from src.data_loader import generate_customer_data

    df = generate_customer_data(5000)
    df_feat = build_features(df)
    print(df_feat.head())
    print(df_feat.columns.tolist())
