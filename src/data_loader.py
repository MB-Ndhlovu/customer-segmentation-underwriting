import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


def generate_synthetic_data(n=5000):
    """Generate synthetic customer data with 4 distinct underwriting segments."""
    segment_probs = [0.30, 0.35, 0.20, 0.15]
    segments = np.random.choice(4, size=n, p=segment_probs)

    data = {
        "age": np.zeros(n, dtype=int),
        "income": np.zeros(n),
        "credit_score": np.zeros(n, dtype=int),
        "employment_years": np.zeros(n),
        "debt_to_income": np.zeros(n),
        "loan_history_count": np.zeros(n, dtype=int),
        "home_ownership_status": np.zeros(n, dtype=int),
        "verified_income": np.zeros(n, dtype=int),
    }

    base_params = {
        0: {  # Mass Market
            "age": (22, 35),
            "income": (18000, 45000),
            "credit_score": (520, 640),
            "employment_years": (0.5, 6),
            "debt_to_income": (0.25, 0.45),
            "loan_history_count": (0, 3),
            "home_ownership": (0.10,),
            "verified_income": (0.40,),
        },
        1: {  # Rising Prime
            "age": (25, 42),
            "income": (45000, 90000),
            "credit_score": (620, 720),
            "employment_years": (3, 12),
            "debt_to_income": (0.15, 0.35),
            "loan_history_count": (1, 5),
            "home_ownership": (0.30,),
            "verified_income": (0.70,),
        },
        2: {  # Established Prime
            "age": (32, 58),
            "income": (75000, 180000),
            "credit_score": (700, 840),
            "employment_years": (7, 30),
            "debt_to_income": (0.10, 0.28),
            "loan_history_count": (2, 8),
            "home_ownership": (0.85,),
            "verified_income": (0.95,),
        },
        3: {  # Subprime High-Risk
            "age": (20, 48),
            "income": (12000, 40000),
            "credit_score": (450, 600),
            "employment_years": (0, 4),
            "debt_to_income": (0.35, 0.65),
            "loan_history_count": (1, 7),
            "home_ownership": (0.05,),
            "verified_income": (0.20,),
        },
    }

    for seg in range(4):
        mask = segments == seg
        p = base_params[seg]
        n_seg = mask.sum()

        data["age"][mask] = np.random.randint(p["age"][0], p["age"][1] + 1, n_seg)
        data["income"][mask] = np.random.uniform(p["income"][0], p["income"][1], n_seg)
        data["credit_score"][mask] = np.random.randint(p["credit_score"][0], p["credit_score"][1] + 1, n_seg)
        data["employment_years"][mask] = np.round(np.random.uniform(p["employment_years"][0], p["employment_years"][1], n_seg), 1)
        data["debt_to_income"][mask] = np.round(np.random.uniform(p["debt_to_income"][0], p["debt_to_income"][1], n_seg), 4)
        data["loan_history_count"][mask] = np.random.randint(p["loan_history_count"][0], p["loan_history_count"][1] + 1, n_seg)
        data["home_ownership_status"][mask] = (np.random.random(n_seg) < p["home_ownership"][0]).astype(int)
        data["verified_income"][mask] = (np.random.random(n_seg) < p["verified_income"][0]).astype(int)

    df = pd.DataFrame(data)
    df["segment_label"] = segments
    return df


def load_data(n=5000):
    df = generate_synthetic_data(n)
    return df