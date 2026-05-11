import numpy as np
import pandas as pd

np.random.seed(42)

def build_dataset(n=5000):
    """Generate synthetic customer dataset with 4 natural segment archetypes."""
    segment_weights = [0.35, 0.25, 0.25, 0.15]

    income_bounds = [(35000, 90000), (45000, 120000), (90000, 250000), (18000, 50000)]
    credit_bounds = [(580, 699), (650, 749), (720, 850), (480, 620)]
    emp_years_bounds = [(2, 15), (0, 5), (5, 30), (0, 6)]
    dti_bounds = [(0.15, 0.35), (0.10, 0.30), (0.05, 0.25), (0.40, 0.70)]
    loan_count_bounds = [(1, 5), (0, 3), (0, 2), (3, 10)]
    age_bounds = [(25, 55), (22, 40), (30, 60), (20, 45)]
    home_owner_probs = [0.55, 0.65, 0.85, 0.25]
    verified_income_probs = [0.70, 0.60, 0.90, 0.35]

    records = []
    for i in range(n):
        seg = np.random.choice(4, p=segment_weights)
        records.append({
            "income": round(np.random.uniform(*income_bounds[seg]), 2),
            "credit_score": np.random.randint(*credit_bounds[seg]),
            "employment_years": round(np.random.uniform(*emp_years_bounds[seg]), 2),
            "debt_to_income": round(np.random.uniform(*dti_bounds[seg]), 4),
            "loan_history_count": np.random.randint(*loan_count_bounds[seg]),
            "age": np.random.randint(*age_bounds[seg]),
            "home_ownership": 1 if np.random.random() < home_owner_probs[seg] else 0,
            "verified_income": 1 if np.random.random() < verified_income_probs[seg] else 0,
        })

    df = pd.DataFrame(records)
    df["segment_label"] = -1
    return df