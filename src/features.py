import pandas as pd
import numpy as np


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM-style features derived from the customer attributes.
    """
    f = df.copy()

    # Frequency proxy — loan_history_count normalized
    f["loan_frequency"] = f["loan_history_count"] / (f["loan_history_count"].max() + 1)

    # Recency proxy — younger employment = more recent start (proxy for new customer)
    f["employment_recency"] = 1 / (f["employment_years"] + 1)

    # Monetary proxy — income per year of employment (stability-adjusted income)
    f["income_stability_adj"] = f["income"] / (f["employment_years"] + 1)

    return f


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Behavioral signals from application data.
    """
    f = df.copy()

    # Debt burden severity
    f["high_dti_flag"] = (f["debt_to_income"] > 0.40).astype(int)

    # Credit utilization signal (credit score bands)
    f["credit_band"] = pd.cut(f["credit_score"],
                              bins=[0, 620, 680, 740, 850],
                              labels=[0, 1, 2, 3]).astype(int)

    # Loan history intensity
    f["high_loan_history"] = (f["loan_history_count"] > 4).astype(int)

    # Verified income reliability signal
    f["income_verified"] = f["verified_income"]

    return f


def build_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Employment and residence stability features.
    """
    f = df.copy()

    # Employment stability score
    f["employment_stable"] = (f["employment_years"] >= 3).astype(int)

    # Young and employed — rising prime indicator
    f["young_professional"] = ((f["age"] < 35) & (f["employment_years"] >= 2)).astype(int)

    # Established homeowner
    f["established_homeowner"] = ((f["age"] >= 40) & (f["home_ownership"] == 1)).astype(int)

    # Income-to-age ratio (earnings trajectory)
    f["income_age_ratio"] = f["income"] / f["age"]

    return f


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    f = build_rfm_features(df)
    f = build_behavioral_features(f)
    f = build_stability_features(f)
    return f


def get_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the full feature set (original + engineered) used for clustering.
    """
    f = compute_all_features(df)

    engineered = [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age", "home_ownership", "verified_income",
        "loan_frequency", "employment_recency", "income_stability_adj",
        "high_dti_flag", "credit_band", "high_loan_history", "income_verified",
        "employment_stable", "young_professional", "established_homeowner",
        "income_age_ratio"
    ]

    return f[engineered]


if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    df = generate_synthetic_data()
    f = get_engineered_features(df)
    print(f.shape)
    print(f.describe())