import numpy as np
import pandas as pd

def generate_customer_data(n=5000, seed=42):
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    income = rng.lognormal(10.5, 0.7, n)
    credit_score = rng.normal(680, 80, n).clip(300, 850).astype(int)
    employment_years = rng.exponential(6, n).clip(0, 40)
    debt_to_income = rng.beta(2, 5, n) * 0.45
    loan_history_count = rng.poisson(2.5, n)
    age = rng.normal(38, 12, n).clip(18, 75).astype(int)
    home_ownership = rng.choice([0, 1], n, p=[0.35, 0.65])
    verified_income = rng.choice([0, 1], n, p=[0.25, 0.75])

    df = pd.DataFrame({
        "income": np.round(income, 2),
        "credit_score": credit_score,
        "employment_years": np.round(employment_years, 2),
        "debt_to_income": np.round(debt_to_income, 4),
        "loan_history_count": loan_history_count,
        "age": age,
        "home_ownership": home_ownership,
        "verified_income": verified_income,
    })
    return df