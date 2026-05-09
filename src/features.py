"""Feature engineering: RFM, behavioural, and stability features."""

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["income_per_employment_year"] = (df["income"] / (df["employment_years"] + 1)).round(2)

    df["credit_to_income_ratio"] = (df["credit_score"] / (df["income"] / 1000)).round(4)

    df["loan_density"] = (df["loan_history_count"] / (df["age"] - 17)).clip(0, 1).round(4)

    df["stability_score"] = (
        (df["employment_years"] * 0.3)
        + (df["verified_income"] * 0.3)
        + (df["home_ownership"].isin([1, 2]).astype(float) * 0.2)
        + ((df["credit_score"] - 300) / 550 * 0.2)
    ).round(4)

    df["debt_burden_flag"] = (df["debt_to_income"] > 0.36).astype(int)

    df["credit_band"] = pd.cut(
        df["credit_score"],
        bins=[0, 580, 670, 740, 850],
        labels=[0, 1, 2, 3],
    ).astype(int)

    return df


FEATURE_COLS = [
    "income",
    "credit_score",
    "employment_years",
    "debt_to_income",
    "loan_history_count",
    "age",
    "home_ownership",
    "verified_income",
    "income_per_employment_year",
    "credit_to_income_ratio",
    "loan_density",
    "stability_score",
    "debt_burden_flag",
    "credit_band",
]