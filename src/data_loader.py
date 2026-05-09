"""
Synthetic customer dataset generator for underwriting segmentation.
Generates 5000 rows with features that naturally cluster into 4 segments.
"""

import numpy as np
import pandas as pd


def generate_customer_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    df = pd.DataFrame(index=range(n))

    # 4 distinct segments with clear separation
    # Segment 0: Mass Market       - middle income, fair credit, moderate stability
    # Segment 1: Rising Prime       - above avg income, good credit, growing stability
    # Segment 2: Established Prime  - high income, excellent credit, high stability
    # Segment 3: Subprime High-Risk - low income, poor credit, low stability
    segment_sizes = [1200, 1400, 1100, 1300]  # sum = 5000

    income       = np.array([])
    credit_score = np.array([])
    employment_years = np.array([])
    debt_to_income   = np.array([])
    loan_history_count = np.array([])
    age           = np.array([])
    home_ownership   = np.array([])
    verified_income  = np.array([])
    segment_label    = np.array([], dtype=int)

    seg_configs = [
        # (income_mean, income_std, credit_mean, credit_std, emp_years_mean, dti_mean, dti_std, loan_hist_mean, age_mean, age_std, home_own_prob, verified_prob)
        (50000, 10000,  660, 55,  4.0, 0.28, 0.07, 2.5, 33, 7,  0.35, 0.45),  # Mass Market
        (78000, 15000,  720, 48,  6.0, 0.22, 0.06, 3.0, 37, 7,  0.55, 0.70),  # Rising Prime
        (115000, 35000, 775, 42,  9.0, 0.16, 0.05, 4.0, 44, 8,  0.80, 0.90), # Established Prime
        (26000,  7000,  570, 50,  1.8, 0.44, 0.10, 5.5, 26, 5,  0.12, 0.18), # Subprime High-Risk
    ]

    for seg_id, (size, cfg) in enumerate(zip(segment_sizes, seg_configs)):
        inc_mean, inc_std, cs_mean, cs_std, emp_mean, dti_mean, dti_std, \
            lh_mean, age_mean, age_std, ho_prob, vi_prob = cfg

        income       = np.concatenate([income, np.random.normal(inc_mean, inc_std, size)])
        credit_score = np.concatenate([credit_score, np.random.normal(cs_mean, cs_std, size)])
        employment_years = np.concatenate([employment_years, np.random.exponential(emp_mean, size)])
        debt_to_income   = np.concatenate([debt_to_income, np.random.normal(dti_mean, dti_std, size)])
        loan_history_count = np.concatenate([loan_history_count, np.random.poisson(lh_mean, size)])
        age           = np.concatenate([age, np.random.normal(age_mean, age_std, size)])
        home_ownership   = np.concatenate([home_ownership, np.random.choice([0,1], size, p=[1-ho_prob, ho_prob])])
        verified_income  = np.concatenate([verified_income, np.random.choice([0,1], size, p=[1-vi_prob, vi_prob])])
        segment_label    = np.concatenate([segment_label, np.full(size, seg_id)])

    df["income"] = np.clip(income, 15000, 300000).round(2)
    df["credit_score"] = np.clip(credit_score, 500, 850).astype(int)
    df["employment_years"] = np.clip(employment_years, 0, 40).round(2)
    df["debt_to_income"] = np.clip(debt_to_income, 0.05, 0.75).round(4)
    df["loan_history_count"] = np.clip(loan_history_count, 0, 15).astype(int)
    df["age"] = np.clip(age, 18, 70).astype(int)
    df["home_ownership"] = home_ownership.astype(int)
    df["verified_income"] = verified_income.astype(int)
    df["segment_label"] = segment_label

    # Shuffle
    idx = np.random.permutation(n)
    for col in df.columns:
        df[col] = df[col].values[idx]

    return df


def get_feature_columns() -> list:
    return [
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_history_count", "age",
        "home_ownership", "verified_income"
    ]


def get_segment_names() -> dict:
    return {
        0: "Mass Market",
        1: "Rising Prime",
        2: "Established Prime",
        3: "Subprime High-Risk"
    }


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.describe())
    print("\nSegment distribution:")
    print(df["segment_label"].value_counts().sort_index())
    for seg in range(4):
        sub = df[df["segment_label"] == seg]
        print(f"\nSegment {seg}:")
        print(f"  Income mean: {sub['income'].mean():.0f}")
        print(f"  Credit score mean: {sub['credit_score'].mean():.1f}")
        print(f"  DTI mean: {sub['debt_to_income'].mean():.3f}")