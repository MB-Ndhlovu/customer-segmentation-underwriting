import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for segmentation:
      - RFM features
      - Behavioral features
      - Stability features
    """
    df = df.copy()

    # RFM-style features (Recency not available in static synthetic data,
    # so we proxy with credit_score and income level bins)

    # Recency proxy: credit_score normalised (higher = more "recent" credit activity)
    df["credit_score_norm"] = (df["credit_score"] - 300) / (850 - 300)

    # Frequency proxy: loan_history_count per year of employment
    df["loan_frequency"] = df["loan_history_count"] / (df["employment_years"] + 0.5)

    # Monetary proxy: income-to-age ratio (earnings efficiency)
    df["income_per_age"] = df["income"] / (df["age"] + 1)

    # Behavioral: debt burden score
    df["debt_burden_score"] = df["debt_to_income"] * (1 + df["loan_history_count"] * 0.1)

    # Behavioral: verified stability
    df["verified_stability"] = (
        df["verified_income"] * 0.4 + df["home_ownership"] * 0.6
    )

    # Stability: employment consistency (longer = more stable)
    df["employment_stability"] = np.tanh(df["employment_years"] / 10)

    # Stability: income reliability (verified income boosts reliability)
    df["income_reliability"] = df["verified_income"] * 0.5 + (1 - df["debt_to_income"]) * 0.5

    # Interaction: credit_score * employment_stability
    df["credit_employment_interaction"] = df["credit_score_norm"] * df["employment_stability"]

    return df


if __name__ == "__main__":
    from data_loader import generate_synthetic_data

    df = generate_synthetic_data()
    df_feat = build_features(df)
    print("Feature columns:", df_feat.columns.tolist())
    print(df_feat.describe())