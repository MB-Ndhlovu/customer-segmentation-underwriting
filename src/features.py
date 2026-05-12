import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build engineered features for segmentation:
    - RFM features: Recency, Frequency, Monetary proxies
    - Behavioral features: debt burden, credit utilization proxy
    - Stability features: employment tenure, income stability proxy
    """
    X = df.copy()

    # RFM features (loan_application_recency as proxy for recency)
    # Frequency: loan_history_count already represents historical frequency
    # Monetary: income as monetary value

    # Behavioral: debt burden score
    X["debt_burden_score"] = X["debt_to_income"] * X["loan_history_count"]

    # Behavioral: credit risk score (lower credit_score = higher risk)
    X["credit_risk_score"] = (850 - X["credit_score"]) / 550  # normalized 0-1

    # Stability: employment stability (longer tenure = more stable)
    X["employment_stability"] = np.where(
        X["employment_years"] < 2, 0,
        np.where(X["employment_years"] < 5, 1,
                 np.where(X["employment_years"] < 10, 2, 3))
    )

    # Stability: income stability (verified income = stable)
    X["income_stability"] = X["verified_income"] * 0.7 + X["home_ownership"] * 0.3

    # Combined risk indicators
    X["combined_risk"] = (
        X["credit_risk_score"] * 0.4 +
        X["debt_to_income"] * 0.3 +
        (1 - X["employment_stability"] / 3) * 0.3
    )

    # Affordability ratio
    X["affordability_ratio"] = X["income"] / (X["debt_to_income"] * X["income"] + 1)

    return X


def get_feature_columns() -> list:
    """Return the list of features used for clustering."""
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
    ]


if __name__ == "__main__":
    from data_loader import generate_customer_data

    df = generate_customer_data()
    X = build_features(df)
    print("Engineered features shape:", X.shape)
    print(X.describe())
