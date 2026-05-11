import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)


def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic customer dataset with features relevant to loan underwriting."""

    # Pre-define segment parameters to ensure 4 distinct clusters emerge after KMeans
    segment_params = [
        # (income_mean, income_std, credit_mean, credit_std, emp_mean, emp_std, dti_mean, dti_std, loan_count_mean, loan_count_std, age_mean, age_std, home_owner_prob, verified_prob)
        # Segment 0: Mass Market
        (45000, 12000, 650, 60, 5.0, 2.0, 0.25, 0.10, 2.0, 1.5, 35, 8, 0.40, 0.55),
        # Segment 1: Rising Prime
        (75000, 18000, 720, 50, 2.5, 1.5, 0.35, 0.12, 3.0, 2.0, 30, 5, 0.35, 0.70),
        # Segment 2: Established Prime
        (120000, 30000, 780, 40, 10.0, 3.0, 0.18, 0.08, 4.0, 2.5, 45, 8, 0.75, 0.90),
        # Segment 3: Subprime High-Risk
        (28000, 8000, 580, 70, 1.5, 1.0, 0.55, 0.15, 1.0, 1.0, 28, 5, 0.15, 0.30),
    ]

    rows = []
    for seg_id, params in enumerate(segment_params):
        inc_m, inc_s, cr_m, cr_s, emp_m, emp_s, dti_m, dti_s, lc_m, lc_s, age_m, age_s, ho_p, v_p = params
        n_seg = n // 4
        income = np.random.normal(inc_m, inc_s, n_seg).clip(10000, 500000)
        credit_score = np.random.normal(cr_m, cr_s, n_seg).clip(300, 850)
        employment_years = np.random.exponential(emp_m, n_seg).clip(0, 40)
        debt_to_income = np.random.beta(2, 6, n_seg) * (dti_m * 3) + 0.05
        debt_to_income = debt_to_income.clip(0.05, 0.9)
        loan_history_count = np.random.poisson(max(1, lc_m), n_seg).clip(0, 20)
        age = np.random.normal(age_m, age_s, n_seg).clip(18, 75)
        home_ownership = np.random.binomial(1, ho_p, n_seg)
        verified_income = np.random.binomial(1, v_p, n_seg)

        for i in range(n_seg):
            rows.append({
                "income": round(income[i], 2),
                "credit_score": round(credit_score[i]),
                "employment_years": round(employment_years[i], 2),
                "debt_to_income": round(debt_to_income[i], 4),
                "loan_history_count": int(loan_history_count[i]),
                "age": int(age[i]),
                "home_ownership": int(home_ownership[i]),
                "verified_income": int(verified_income[i]),
                "true_segment": seg_id,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


FEATURE_COLS = [
    "income", "credit_score", "employment_years", "debt_to_income",
    "loan_history_count", "age", "home_ownership", "verified_income"
]


def load_data(n: int = 5000) -> pd.DataFrame:
    return generate_customer_data(n)


if __name__ == "__main__":
    df = load_data()
    print(df.describe())
    print("\nTrue segment distribution:")
    print(df["true_segment"].value_counts().sort_index())
