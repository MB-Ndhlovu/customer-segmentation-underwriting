"""Synthetic customer dataset for underwriting segmentation."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

def generate_customer_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic customer data with 4 distinct underwriting segments."""
    np.random.seed(seed)

    # Segment definitions for realistic clustering
    # Each segment = [income_mean, credit_score_mean, employment_years_mean,
    #                 debt_to_income_mean, loan_history_count_mean, age_mean,
    #                 home_ownership_prob (rent=0, own=1), verified_income_prob]
    segment_params = {
        0: {  # Mass Market — moderate income, fair credit, standard employment
            "income": (55000, 15000),
            "credit_score": (660, 50),
            "employment_years": (5, 3),
            "debt_to_income": (0.28, 0.10),
            "loan_history_count": (2, 1.5),
            "age": (35, 8),
            "home_ownership": 0.35,
            "verified_income": 0.60,
        },
        1: {  # Rising Prime — growing income, improving credit, stable employment
            "income": (85000, 20000),
            "credit_score": (720, 40),
            "employment_years": (7, 3),
            "debt_to_income": (0.22, 0.08),
            "loan_history_count": (1, 1.2),
            "age": (38, 7),
            "home_ownership": 0.50,
            "verified_income": 0.80,
        },
        2: {  # Established Prime — high income, excellent credit, long tenure
            "income": (130000, 35000),
            "credit_score": (780, 35),
            "employment_years": (12, 5),
            "debt_to_income": (0.15, 0.06),
            "loan_history_count": (0.5, 0.8),
            "age": (45, 8),
            "home_ownership": 0.75,
            "verified_income": 0.95,
        },
        3: {  # Subprime High-Risk — low income, poor credit, high DTI, many prior loans
            "income": (32000, 10000),
            "credit_score": (580, 45),
            "employment_years": (3, 2),
            "debt_to_income": (0.42, 0.12),
            "loan_history_count": (5, 2),
            "age": (30, 7),
            "home_ownership": 0.20,
            "verified_income": 0.35,
        },
    }

    segment_labels = []
    data_rows = []

    # Distribute samples evenly across 4 segments
    per_segment = n_samples // 4
    for seg_id, params in segment_params.items():
        for _ in range(per_segment):
            income = max(15000, np.random.normal(params["income"][0], params["income"][1]))
            credit_score = min(850, max(300, np.random.normal(params["credit_score"][0], params["credit_score"][1])))
            employment_years = max(0, np.random.normal(params["employment_years"][0], params["employment_years"][1]))
            debt_to_income = max(0.0, min(0.95, np.random.normal(params["debt_to_income"][0], params["debt_to_income"][1])))
            loan_history_count = max(0, int(np.random.normal(params["loan_history_count"][0], params["loan_history_count"][1])))
            age = max(18, min(80, np.random.normal(params["age"][0], params["age"][1])))
            home_ownership = 1 if np.random.random() < params["home_ownership"] else 0
            verified_income = 1 if np.random.random() < params["verified_income"] else 0

            data_rows.append({
                "income": round(income, 2),
                "credit_score": round(credit_score),
                "employment_years": round(employment_years, 2),
                "debt_to_income": round(debt_to_income, 4),
                "loan_history_count": loan_history_count,
                "age": round(age),
                "home_ownership": home_ownership,
                "verified_income": verified_income,
            })
            segment_labels.append(seg_id)

    df = pd.DataFrame(data_rows)
    df["segment_true"] = segment_labels

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nSegment distribution:\n{df['segment_true'].value_counts().sort_index()}")