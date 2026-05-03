"""Feature engineering for customer segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """RFM-style features: Recency (proxy via age), Frequency (loan history), Monetary (income)."""
    features = pd.DataFrame()

    # Monetary — income per year of employment (proxy for earning power)
    features["income_per_tenure"] = df["income"] / (df["employment_years"] + 1)

    # Frequency — loan history density
    features["loan_density"] = df["loan_history_count"] / (df["age"] - 17)

    # Monetary — income stability proxy (verified income boost)
    features["income_verified_flag"] = df["verified_income"]

    return features


def build_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Behavioral features: debt burden, credit utilization proxies."""
    features = pd.DataFrame()

    # Debt burden score
    features["debt_burden_score"] = df["debt_to_income"] * 100

    # Credit score normalized band
    features["credit_band"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 850],
        labels=[0, 1, 2, 3],
    ).astype(float)

    # Home ownership as stability proxy
    features["is_homeowner"] = (df["home_ownership"] >= 1).astype(int)

    return features


def build_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Employment and residence stability features."""
    features = pd.DataFrame()

    # Employment tenure ratio
    features["tenure_ratio"] = df["employment_years"] / (df["age"] - 17)

    # Long tenure flag
    features["long_tenure_flag"] = (df["employment_years"] >= 5).astype(int)

    # Young high-earner indicator
    features["young_high_earner"] = ((df["age"] < 30) & (df["income"] > 70000)).astype(int)

    return features


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all feature groups into a single DataFrame."""
    rfm = build_rfm_features(df)
    beh = build_behavioral_features(df)
    stab = build_stability_features(df)

    features = pd.concat([rfm, beh, stab], axis=1)

    # Ensure no NaN from division
    features = features.fillna(0)

    return features


def scale_features(X: pd.DataFrame) -> np.ndarray:
    """Standardize features for clustering."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)