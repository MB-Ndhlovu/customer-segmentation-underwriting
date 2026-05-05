import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_rfm_features(df):
    """RFM: Recency (age proxy), Frequency (loan history), Monetary (income)."""
    recency = 65 - df["age"]
    frequency = df["loan_history_count"]
    monetary = df["income"] / 100000
    return pd.DataFrame({"recency": recency, "frequency": frequency, "monetary": monetary})


def compute_behavioral_features(df):
    """Debt burden, credit utilization proxy, income stability proxy."""
    debt_burden = df["debt_to_income"]
    credit_utilization = 1 - (df["credit_score"] - 300) / 550
    income_stability = (df["verified_income"] * 0.5 + (df["employment_years"] > 3).astype(float) * 0.5)
    return pd.DataFrame({
        "debt_burden": debt_burden,
        "credit_utilization": credit_utilization,
        "income_stability": income_stability,
    })


def compute_stability_features(df):
    """Employment stability, homeownership stability, income level."""
    emp_stability = np.minimum(df["employment_years"] / 10, 1.0)
    home_stability = df["home_ownership"].astype(float)
    income_level = df["income"] / 200000
    return pd.DataFrame({
        "emp_stability": emp_stability,
        "home_stability": home_stability,
        "income_level": income_level,
    })


def build_feature_matrix(df, scaler=None, fit=True):
    """Combine all feature groups into a scaled feature matrix."""
    rfm = compute_rfm_features(df)
    beh = compute_behavioral_features(df)
    stab = compute_stability_features(df)
    features = pd.concat([rfm, beh, stab], axis=1)
    if fit:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
    else:
        scaled = scaler.transform(features)
    return scaled, scaler