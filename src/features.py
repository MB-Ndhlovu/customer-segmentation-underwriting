"""Feature engineering: RFM, behavioral, and stability features."""
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the customer DataFrame."""
    f = df.copy()

    # RFM-adjacent: income per employment year (recency proxy = employment stability)
    f["income_per_tenure"] = f["income"] / (f["employment_years"] + 0.5)

    # Behavioral: loan density (loan count per year of credit history)
    f["loan_density"] = f["loan_history_count"] / (f["employment_years"] + 0.5)

    # Stability: DTI stability score (inverse DTI * tenure interaction)
    f["dti_stability"] = (1 / (f["debt_to_income"] + 0.01)) * np.log1p(f["employment_years"])

    # Credit quality: normalized credit score bucket
    f["credit_bucket"] = pd.cut(
        f["credit_score"],
        bins=[0, 580, 670, 740, 850],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # Income tier
    f["income_tier"] = pd.cut(
        f["income"],
        bins=[0, 40_000, 75_000, 110_000, 1_000_000],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # Verified income flag (already binary 0/1)

    return f

def get_feature_columns() -> list:
    """Columns to use for clustering / classification."""
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership_enc",
        "verified_income",
    ]