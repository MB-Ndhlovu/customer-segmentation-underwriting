"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd

np.random.seed(42)

# Segment parameters: (income_mean, credit_score_mean, employment_years_mean,
#                      debt_to_income_mean, loan_history_count_mean, age_mean,
#                      home_ownership_prob, verified_income_prob, weight)
SEGMENT_PARAMS = {
    0: (55000, 670, 4.0, 0.28, 2.0, 35, 0.50, 0.60, 0.35),  # Mass Market
    1: (72000, 720, 3.5, 0.22, 1.5, 28, 0.30, 0.70, 0.25),  # Rising Prime
    2: (95000, 780, 8.0, 0.18, 1.0, 45, 0.80, 0.90, 0.25),  # Established Prime
    3: (32000, 590, 2.0, 0.40, 4.0, 30, 0.20, 0.35, 0.15),  # Subprime High-Risk
}


def _draw_from_segment(seg_idx, n):
    params = SEGMENT_PARAMS[seg_idx]
    (income_mean, cs_mean, emp_mean, dti_mean,
     lhc_mean, age_mean, ho_prob, vi_prob, _) = params

    income = np.random.normal(income_mean, income_mean * 0.25, n).clip(15000, 250000)
    credit_score = np.random.normal(cs_mean, 60, n).clip(500, 850).astype(int)
    employment_years = np.random.exponential(emp_mean, n).clip(0, 35)
    debt_to_income = np.random.beta(3, 8, n) * 0.55 + 0.05
    loan_history_count = np.random.poisson(max(lhc_mean - 0.5, 0.5), n).clip(0, 12)
    age = np.random.normal(age_mean, 8, n).clip(18, 75).astype(int)
    home_ownership_status = (np.random.random(n) < ho_prob).astype(int)
    verified_income = (np.random.random(n) < vi_prob).astype(int)

    return pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "loan_history_count": loan_history_count,
        "age": age,
        "home_ownership_status": home_ownership_status,
        "verified_income": verified_income,
        "_segment_hint": seg_idx,
    })


def load_customer_data(n=5000):
    """Generate n synthetic customer records."""
    weights = [SEGMENT_PARAMS[i][8] for i in range(4)]
    labels = np.random.choice(4, size=n, p=weights)

    chunks = []
    for seg_idx in range(4):
        count = int(np.sum(labels == seg_idx))
        if count > 0:
            chunks.append(_draw_from_segment(seg_idx, count))

    df = pd.concat(chunks, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.drop(columns=["_segment_hint"])


if __name__ == "__main__":
    df = load_customer_data()
    print(f"Generated {len(df)} rows")
    print(df.describe())
    print("\nSegment preview (approximate labels via heuristic):")
    print(df.head())