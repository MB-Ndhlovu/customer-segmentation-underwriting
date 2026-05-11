"""
Feature engineering for customer segmentation.
Transforms raw customer data into features for clustering and classification.
"""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for clustering and classification.
    Returns DataFrame with all numeric features.
    """
    f = df.copy()

    # ── RFM-like financial health features ──────────────────────────────────
    f["dti_risk_score"] = f["debt_to_income"] * 10          # higher DTI → higher risk
    f["income_per_loan"] = f["income"] / (f["loan_history_count"] + 1)

    # ── Behavioral features ─────────────────────────────────────────────────
    f["emp_stability"] = np.log1p(f["employment_years"])   # log-smoothed tenure
    f["loan_density"] = f["loan_history_count"] / (f["age"] - 17)  # loans per eligible year
    f["verified_flag"] = f["verified_income"].astype(float)

    # ── Stability / capitalization features ─────────────────────────────────
    f["home_ownership_encoded"] = f["home_ownership"].map({
        "own": 3, "mortgage": 2, "rent": 1, "other": 0
    }).astype(float)

    f["income_stability"] = f["verified_flag"] * f["emp_stability"]

    # ── Derived risk indicators ─────────────────────────────────────────────
    f["credit_dti_interaction"] = f["credit_score"] / 100 * (1 - f["debt_to_income"])

    return f


def get_feature_columns() -> list:
    """
    Returns the list of feature column names used for clustering / classification.
    """
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership_encoded",
        "verified_income",
        # derived
        "dti_risk_score",
        "income_per_loan",
        "emp_stability",
        "loan_density",
        "income_stability",
        "credit_dti_interaction",
    ]