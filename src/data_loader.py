import numpy as np
import pandas as pd


def generate_customer_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # Segment probabilities for realistic distribution
    # Mass Market: 40%, Rising Prime: 25%, Established Prime: 20%, Subprime High-Risk: 15%
    segment_probs = [0.40, 0.25, 0.20, 0.15]
    segments = np.random.choice(4, size=n, p=segment_probs)

    data = {
        "income": np.zeros(n),
        "credit_score": np.zeros(n),
        "employment_years": np.zeros(n),
        "debt_to_income": np.zeros(n),
        "loan_history_count": np.zeros(n),
        "age": np.zeros(n),
        "home_ownership": np.zeros(n),  # 0=rent, 1=own
        "verified_income": np.zeros(n),   # 0=unverified, 1=verified
    }

    # Segment 0: Mass Market
    mask_0 = segments == 0
    n_0 = mask_0.sum()
    data["income"][mask_0] = np.random.normal(55000, 15000, n_0)
    data["credit_score"][mask_0] = np.random.normal(660, 50, n_0)
    data["employment_years"][mask_0] = np.random.exponential(4, n_0)
    data["debt_to_income"][mask_0] = np.random.normal(0.30, 0.10, n_0)
    data["loan_history_count"][mask_0] = np.random.poisson(2, n_0)
    data["age"][mask_0] = np.random.normal(35, 8, n_0)
    data["home_ownership"][mask_0] = np.random.choice([0, 1], size=n_0, p=[0.55, 0.45])
    data["verified_income"][mask_0] = np.random.choice([0, 1], size=n_0, p=[0.40, 0.60])

    # Segment 1: Rising Prime
    mask_1 = segments == 1
    n_1 = mask_1.sum()
    data["income"][mask_1] = np.random.normal(85000, 20000, n_1)
    data["credit_score"][mask_1] = np.random.normal(720, 40, n_1)
    data["employment_years"][mask_1] = np.random.exponential(5, n_1)
    data["debt_to_income"][mask_1] = np.random.normal(0.25, 0.08, n_1)
    data["loan_history_count"][mask_1] = np.random.poisson(3, n_1)
    data["age"][mask_1] = np.random.normal(32, 6, n_1)
    data["home_ownership"][mask_1] = np.random.choice([0, 1], size=n_1, p=[0.35, 0.65])
    data["verified_income"][mask_1] = np.random.choice([0, 1], size=n_1, p=[0.15, 0.85])

    # Segment 2: Established Prime
    mask_2 = segments == 2
    n_2 = mask_2.sum()
    data["income"][mask_2] = np.random.normal(130000, 35000, n_2)
    data["credit_score"][mask_2] = np.random.normal(780, 35, n_2)
    data["employment_years"][mask_2] = np.random.exponential(10, n_2)
    data["debt_to_income"][mask_2] = np.random.normal(0.18, 0.06, n_2)
    data["loan_history_count"][mask_2] = np.random.poisson(5, n_2)
    data["age"][mask_2] = np.random.normal(45, 8, n_2)
    data["home_ownership"][mask_2] = np.random.choice([0, 1], size=n_2, p=[0.10, 0.90])
    data["verified_income"][mask_2] = np.random.choice([0, 1], size=n_2, p=[0.05, 0.95])

    # Segment 3: Subprime High-Risk
    mask_3 = segments == 3
    n_3 = mask_3.sum()
    data["income"][mask_3] = np.random.normal(32000, 10000, n_3)
    data["credit_score"][mask_3] = np.random.normal(580, 45, n_3)
    data["employment_years"][mask_3] = np.random.exponential(2, n_3)
    data["debt_to_income"][mask_3] = np.random.normal(0.45, 0.12, n_3)
    data["loan_history_count"][mask_3] = np.random.poisson(4, n_3)
    data["age"][mask_3] = np.random.normal(28, 7, n_3)
    data["home_ownership"][mask_3] = np.random.choice([0, 1], size=n_3, p=[0.80, 0.20])
    data["verified_income"][mask_3] = np.random.choice([0, 1], size=n_3, p=[0.70, 0.30])

    df = pd.DataFrame(data)

    # Clip to realistic bounds
    df["income"] = df["income"].clip(15000, 500000)
    df["credit_score"] = df["credit_score"].clip(300, 850).round(0)
    df["employment_years"] = df["employment_years"].clip(0, 50).round(1)
    df["debt_to_income"] = df["debt_to_income"].clip(0, 1.0).round(3)
    df["loan_history_count"] = df["loan_history_count"].clip(0, 20).round(0).astype(int)
    df["age"] = df["age"].clip(18, 80).round(0).astype(int)
    df["home_ownership"] = df["home_ownership"].astype(int)
    df["verified_income"] = df["verified_income"].astype(int)

    return df


if __name__ == "__main__":
    df = generate_customer_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nSegment distribution (pre-clustering): N/A - data is random mix")
