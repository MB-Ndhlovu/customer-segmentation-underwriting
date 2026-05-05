import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # Four segments with distinct distributions
    # 0: Mass Market (standard applicants)
    # 1: Rising Prime (young, good income trajectory)
    # 2: Established Prime (stable, high credit)
    # 3: Subprime High-Risk (high DTI, poor credit)

    sizes = [n_samples // 4] * 4
    sizes[0] += n_samples - sum(sizes)  # balance rounding

    data = []

    # Mass Market — moderate across the board
    for _ in range(sizes[0]):
        income = int(np.random.normal(55000, 15000))
        credit_score = int(np.random.normal(670, 60))
        employment_years = np.random.exponential(4)
        debt_to_income = np.random.beta(2, 5) * 0.45
        loan_history_count = np.random.poisson(2)
        age = int(np.random.normal(35, 8))
        home_ownership = np.random.choice([0, 1], p=[0.55, 0.45])
        verified_income = np.random.choice([0, 1], p=[0.3, 0.7])
        data.append([income, credit_score, employment_years, debt_to_income,
                     loan_history_count, age, home_ownership, verified_income])

    # Rising Prime — young, good income, low DTI
    for _ in range(sizes[1]):
        income = int(np.random.normal(72000, 18000))
        credit_score = int(np.random.normal(720, 50))
        employment_years = np.random.exponential(3)
        debt_to_income = np.random.beta(1.5, 6) * 0.35
        loan_history_count = np.random.poisson(1)
        age = int(np.random.normal(27, 4))
        home_ownership = np.random.choice([0, 1], p=[0.7, 0.3])
        verified_income = np.random.choice([0, 1], p=[0.15, 0.85])
        data.append([income, credit_score, employment_years, debt_to_income,
                     loan_history_count, age, home_ownership, verified_income])

    # Established Prime — older, high credit, stable
    for _ in range(sizes[2]):
        income = int(np.random.normal(95000, 22000))
        credit_score = int(np.random.normal(770, 45))
        employment_years = np.random.exponential(10)
        debt_to_income = np.random.beta(1, 7) * 0.30
        loan_history_count = np.random.poisson(3)
        age = int(np.random.normal(45, 7))
        home_ownership = np.random.choice([0, 1], p=[0.2, 0.8])
        verified_income = np.random.choice([0, 1], p=[0.05, 0.95])
        data.append([income, credit_score, employment_years, debt_to_income,
                     loan_history_count, age, home_ownership, verified_income])

    # Subprime High-Risk — lower credit, high DTI, unstable
    for _ in range(sizes[3]):
        income = int(np.random.normal(38000, 12000))
        credit_score = int(np.random.normal(580, 55))
        employment_years = np.random.exponential(2)
        debt_to_income = np.random.beta(3, 3) * 0.55
        loan_history_count = np.random.poisson(5)
        age = int(np.random.normal(32, 8))
        home_ownership = np.random.choice([0, 1], p=[0.8, 0.2])
        verified_income = np.random.choice([0, 1], p=[0.6, 0.4])
        data.append([income, credit_score, employment_years, debt_to_income,
                     loan_history_count, age, home_ownership, verified_income])

    cols = ["income", "credit_score", "employment_years", "debt_to_income",
            "loan_history_count", "age", "home_ownership", "verified_income"]
    df = pd.DataFrame(data, columns=cols)

    # Clamp to realistic bounds
    df["income"] = df["income"].clip(lower=15000, upper=250000)
    df["credit_score"] = df["credit_score"].clip(lower=500, upper=850)
    df["employment_years"] = df["employment_years"].clip(lower=0, upper=45)
    df["debt_to_income"] = df["debt_to_income"].clip(lower=0.01, upper=0.65)
    df["loan_history_count"] = df["loan_history_count"].clip(lower=0, upper=20).astype(int)
    df["age"] = df["age"].clip(lower=18, upper=75).astype(int)
    df["home_ownership"] = df["home_ownership"].astype(int)
    df["verified_income"] = df["verified_income"].astype(int)

    # True labels for validation (not used in training)
    df["true_segment"] = [0] * sizes[0] + [1] * sizes[1] + [2] * sizes[2] + [3] * sizes[3]

    return df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    feature_cols = ["income", "credit_score", "employment_years", "debt_to_income",
                   "loan_history_count", "age", "home_ownership", "verified_income"]
    return df[feature_cols].values


def scale_features(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


if __name__ == "__main__":
    df = generate_synthetic_data()
    print(df.describe())
    print("\nSegment distribution:")
    print(df["true_segment"].value_counts().sort_index())