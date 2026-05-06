import pandas as pd
import numpy as np


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recency, Frequency, Monetary proxies using available columns."""
    feat = pd.DataFrame(index=df.index)

    feat["income_per_loan"] = df["income"] / (df["loan_history_count"] + 1)
    feat["loan_density"] = df["loan_history_count"] / (df["age"] - 17 + 1)
    feat["income_to_age"] = df["income"] / (df["age"] + 1)

    return feat


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioural signals from credit and loan history."""
    feat = pd.DataFrame(index=df.index)

    feat["credit_utilization_proxy"] = (df["debt_to_income"] * df["income"]) / (df["credit_score"] + 1)
    feat["has_loan_history"] = (df["loan_history_count"] > 0).astype(int)
    feat["high_loan_burden"] = (df["debt_to_income"] > 0.35).astype(int)
    feat["stable_income_flag"] = (df["verified_income"] == 1).astype(int)

    return feat


def build_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Employment and residential stability indicators."""
    feat = pd.DataFrame(index=df.index)

    feat["employment_stability"] = df["employment_years"] / (df["age"] - 17 + 1)
    feat["is_homeowner"] = (df["home_ownership"] >= 1).astype(int)
    feat["young_high_earner"] = ((df["age"] < 30) & (df["income"] > 70000)).astype(int)
    feat["senior_estabilished"] = ((df["age"] > 50) & (df["employment_years"] > 10)).astype(int)

    return feat


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all feature groups into a single feature DataFrame."""
    features = pd.concat(
        [
            build_rfm_features(df),
            build_behavioral_features(df),
            build_stability_features(df),
        ],
        axis=1,
    )
    return features


def get_feature_names() -> list:
    return [
        "income_per_loan",
        "loan_density",
        "income_to_age",
        "credit_utilization_proxy",
        "has_loan_history",
        "high_loan_burden",
        "stable_income_flag",
        "employment_stability",
        "is_homeowner",
        "young_high_earner",
        "senior_estabilished",
    ]