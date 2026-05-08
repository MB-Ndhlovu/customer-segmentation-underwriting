"""Feature engineering for customer segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create RFM, behavioral, and stability features from raw application data."""
    X = df[[
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_history_count", "age",
        "home_ownership", "verified_income"
    ]].copy()

    # RFM-equivalent features (proxy for recency/frequency via available signals)
    # Recency proxy: credit_score trend (higher score = more recent activity)
    X["credit_score_norm"] = X["credit_score"] / 850

    # Frequency proxy: loan history density (loans per year of credit history)
    X["loan_frequency"] = X["loan_history_count"] / (X["employment_years"] + 0.5)

    # Monetary proxy: income per age (proxy for earnings trajectory)
    X["income_per_age"] = X["income"] / X["age"]

    # Behavioral features
    # Credit utilization signal (high loan count + low score = over-leveraged)
    X["leverage_signal"] = X["loan_history_count"] * (1 - X["credit_score_norm"])

    # DTI severity (quadratic to weight high DTI customers)
    X["dti_severity"] = X["debt_to_income"] ** 1.5

    # Stability features
    # Employment stability ratio (longer tenure = more stable)
    X["employment_stability"] = X["employment_years"] / (X["age"] - 18 + 1)

    # Home ownership as stability signal
    X["home_stability"] = X["home_ownership"]

    # Income verification boost
    X["verified_income_flag"] = X["verified_income"]

    # Affordability score (inverse of DTI, normalized)
    X["affordability"] = (0.50 - X["debt_to_income"]) / 0.50
    X["affordability"] = X["affordability"].clip(0, 1)

    return X


def scale_features(X: pd.DataFrame) -> tuple:
    """Standardize features and return scaler for later inverse-transform."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


if __name__ == "__main__":
    from data_loader import generate_customer_data

    df = generate_customer_data()
    X = build_features(df)
    print("Feature columns:", X.columns.tolist())
    print("\nFeature stats:")
    print(X.describe().T[["mean", "std", "min", "max"]])