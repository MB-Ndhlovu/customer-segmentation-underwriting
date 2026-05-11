import numpy as np
import pandas as pd

np.random.seed(42)

SEGMENT_PARAMS = {
    0: {
        "income": (28000, 55000),
        "credit_score": (580, 680),
        "employment_years": (1, 6),
        "debt_to_income": (0.15, 0.38),
        "loan_history_count": (0, 4),
        "age": (20, 34),
        "home_ownership": 0.25,
        "verified_income": 0.60,
    },
    1: {
        "income": (55000, 90000),
        "credit_score": (660, 740),
        "employment_years": (4, 12),
        "debt_to_income": (0.10, 0.30),
        "loan_history_count": (2, 7),
        "age": (27, 42),
        "home_ownership": 0.50,
        "verified_income": 0.82,
    },
    2: {
        "income": (90000, 200000),
        "credit_score": (720, 840),
        "employment_years": (8, 30),
        "debt_to_income": (0.05, 0.22),
        "loan_history_count": (4, 15),
        "age": (35, 62),
        "home_ownership": 0.80,
        "verified_income": 0.95,
    },
    3: {
        "income": (18000, 42000),
        "credit_score": (480, 610),
        "employment_years": (0, 3),
        "debt_to_income": (0.35, 0.65),
        "loan_history_count": (0, 2),
        "age": (18, 38),
        "home_ownership": 0.10,
        "verified_income": 0.35,
    },
}

SEGMENT_WEIGHTS = [0.35, 0.28, 0.20, 0.17]

def _generate_segment(n, params):
    income = np.random.uniform(*params["income"], size=n)
    credit_score = np.random.uniform(*params["credit_score"], size=n)
    employment_years = np.random.uniform(*params["employment_years"], size=n)
    debt_to_income = np.random.uniform(*params["debt_to_income"], size=n)
    loan_history_count = np.random.randint(params["loan_history_count"][0], params["loan_history_count"][1] + 1, size=n)
    age = np.random.randint(params["age"][0], params["age"][1] + 1, size=n)
    home_ownership = (np.random.random(size=n) < params["home_ownership"]).astype(int)
    verified_income = (np.random.random(size=n) < params["verified_income"]).astype(int)
    return pd.DataFrame({
        "income": income,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "debt_to_income": debt_to_income,
        "loan_history_count": loan_history_count,
        "age": age,
        "home_ownership": home_ownership,
        "verified_income": verified_income,
    })

def load_data(n=5000):
    segment_counts = np.random.multinomial(n, SEGMENT_WEIGHTS)
    dfs = []
    for label, count in enumerate(segment_counts):
        df = _generate_segment(count, SEGMENT_PARAMS[label])
        df["_true_segment"] = label
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return data

if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print(df.shape)
    print(df["_true_segment"].value_counts().sort_index())
