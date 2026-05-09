import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
    - RFM features: income-to-credit ratio, credit utilization proxy
    - Behavioral features: loan density (loan_history/age), DTI risk flag
    - Stability features: employment stability, home ownership weight
    """
    X = df.copy()

    # RFM-inspired features
    X["income_credit_ratio"] = X["income"] / (X["credit_score"] + 1)
    X["income_per_employment_year"] = X["income"] / (X["employment_years"] + 1)

    # Behavioral
    X["loan_density"] = X["loan_history_count"] / (X["age"] - 17 + 1)  # normalized by working years
    X["high_dti"] = (X["debt_to_income"] > 0.36).astype(int)
    X["credit_score_binned"] = pd.cut(X["credit_score"], bins=[0, 580, 620, 670, 740, 850],
                                       labels=[0, 1, 2, 3, 4]).astype(float).fillna(0)

    # Stability
    X["employment_stability"] = X["employment_years"] * (1 - X["high_dti"])
    X["home_income_verified"] = X["home_ownership"] * X["verified_income"]
    X["verified_income_flag"] = X["verified_income"]

    # Interaction terms
    X["dti_x_loan_count"] = X["debt_to_income"] * X["loan_history_count"]
    X["age_x_employment"] = X["age"] * X["employment_years"]

    return X


def get_feature_columns() -> list:
    return [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age", "home_ownership", "verified_income",
        "income_credit_ratio", "income_per_employment_year", "loan_density",
        "high_dti", "credit_score_binned", "employment_stability",
        "home_income_verified", "dti_x_loan_count", "age_x_employment",
    ]


if __name__ == "__main__":
    from data_loader import generate_customer_data
    df = generate_customer_data()
    X = build_features(df)
    print(X.head())
    print("Shape:", X.shape)