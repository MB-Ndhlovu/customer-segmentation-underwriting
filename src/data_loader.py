"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# Segment definitions for synthetic generation
# We generate 4 clusters and label them after KMeans, but seed so shapes are distinct
SEGMENT_PARAMS = {
    0: {  # Mass Market
        "income": (35000, 75000),
        "credit_score": (580, 700),
        "employment_years": (0.5, 6),
        "debt_to_income": (0.15, 0.35),
        "loan_history_count": (1, 5),
        "age": (22, 45),
        "home_ownership": 0.3,  # 30% own
        "verified_income": 0.6,
    },
    1: {  # Rising Prime
        "income": (65000, 120000),
        "credit_score": (680, 760),
        "employment_years": (3, 12),
        "debt_to_income": (0.10, 0.28),
        "loan_history_count": (2, 7),
        "age": (28, 50),
        "home_ownership": 0.55,
        "verified_income": 0.75,
    },
    2: {  # Established Prime
        "income": (90000, 200000),
        "credit_score": (740, 850),
        "employment_years": (7, 30),
        "debt_to_income": (0.05, 0.20),
        "loan_history_count": (3, 10),
        "age": (35, 65),
        "home_ownership": 0.85,
        "verified_income": 0.92,
    },
    3: {  # Subprime High-Risk
        "income": (18000, 45000),
        "credit_score": (300, 600),
        "employment_years": (0, 3),
        "debt_to_income": (0.30, 0.60),
        "loan_history_count": (0, 4),
        "age": (18, 38),
        "home_ownership": 0.1,
        "verified_income": 0.35,
    },
}


def _draw(param_range, size):
    low, high = param_range
    return np.random.uniform(low, high, size)


def generate_customer_data(n=5000):
    """Generate synthetic customer data with 4 distinct segments."""
    # Balanced generation across segments
    per_segment = n // 4
    records = []

    for seg_id, params in SEGMENT_PARAMS.items():
        for _ in range(per_segment):
            income = _draw(params["income"], 1)[0]
            credit_score = int(_draw(params["credit_score"], 1)[0])
            employment_years = round(_draw(params["employment_years"], 1)[0], 2)
            debt_to_income = round(_draw(params["debt_to_income"], 1)[0], 4)
            loan_history_count = int(np.random.choice(
                range(params["loan_history_count"][0], params["loan_history_count"][1] + 1)
            ))
            age = int(_draw(params["age"], 1)[0])
            home_ownership = 1 if np.random.random() < params["home_ownership"] else 0
            verified_income = 1 if np.random.random() < params["verified_income"] else 0

            records.append({
                "income": income,
                "credit_score": credit_score,
                "employment_years": employment_years,
                "debt_to_income": debt_to_income,
                "loan_history_count": loan_history_count,
                "age": age,
                "home_ownership": home_ownership,
                "verified_income": verified_income,
                "_true_segment": seg_id,
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)

    # Add label column (will be overwritten by KMeans in practice)
    df["segment_label"] = df["_true_segment"]

    return df


def get_feature_columns():
    return [
        "income",
        "credit_score",
        "employment_years",
        "debt_to_income",
        "loan_history_count",
        "age",
        "home_ownership",
        "verified_income",
    ]


def scale_features(df, feature_cols=None, scaler=None):
    if feature_cols is None:
        feature_cols = get_feature_columns()
    X = df[feature_cols].values
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


if __name__ == "__main__":
    df = generate_customer_data(5000)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nSegment distribution:\n{df['_true_segment'].value_counts().sort_index()}")