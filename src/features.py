import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recency, Frequency, Monetary features derived from loan behaviour."""
    # Monetary: income per loan application (proxy)
    df = df.copy()
    df["monetary_income"] = df["income"] / (df["loan_history_count"] + 1)
    # Frequency: loan density — loans per year of employment
    df["loan_frequency"] = df["loan_history_count"] / (df["employment_years"] + 0.5)
    # Recency proxy: age relative to employment — newer workers have shorter history
    df["tenure_recency"] = df["employment_years"] / (df["age"] - 18 + 1)
    return df


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural risk indicators from application patterns."""
    df = df.copy()
    # Debt burden score
    df["debt_burden_score"] = df["debt_to_income"] * df["loan_history_count"]
    # Income stability proxy: verified + homeownership
    df["income_stability"] = df["verified_income"] + df["home_ownership"]
    # Credit utilisation proxy from score band
    df["credit_band"] = pd.cut(df["credit_score"], bins=[0, 580, 670, 740, 850], labels=[0, 1, 2, 3]).astype(float).fillna(0)
    return df


def build_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Employment and residence stability features."""
    df = df.copy()
    # Employment stability: longer tenure = lower risk
    df["emp_stability"] = np.where(df["employment_years"] > 5, 2,
                        np.where(df["employment_years"] > 2, 1, 0))
    # Young borrower risk: age < 25 and high DTI
    df["young_borrower_risk"] = ((df["age"] < 25) & (df["debt_to_income"] > 0.3)).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering stages."""
    df = build_rfm_features(df)
    df = build_behavioral_features(df)
    df = build_stability_features(df)
    return df


def get_feature_columns() -> list:
    """Return list of all features used for clustering / classification."""
    return [
        "income", "credit_score", "employment_years", "debt_to_income",
        "loan_history_count", "age", "home_ownership", "verified_income",
        "monetary_income", "loan_frequency", "tenure_recency",
        "debt_burden_score", "income_stability", "credit_band",
        "emp_stability", "young_borrower_risk",
    ]


def scale_features(df: pd.DataFrame, scaler=None) -> tuple:
    """Scale features; returns scaled DataFrame and fitted scaler."""
    cols = get_feature_columns()
    X = df[cols].values
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler
