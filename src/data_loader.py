"""Generate 5000 synthetic customer records for underwriting segmentation."""
import numpy as np
import pandas as pd

np.random.seed(42)

SEGMENT_NAMES = {
    0: "Mass Market",
    1: "Rising Prime",
    2: "Established Prime",
    3: "Subprime High-Risk",
}

def generate_customer_data(n: int = 5000) -> pd.DataFrame:
    """Generate n synthetic customer records with realistic underwriting features."""
    segments = []
    per_segment = n // 4

    # Segment 0 – Mass Market: moderate income, average credit, stable employment
    s0 = {
        "income": np.random.normal(55_000, 15_000, per_segment),
        "credit_score": np.random.normal(660, 60, per_segment),
        "employment_years": np.random.exponential(5, per_segment) + 1,
        "debt_to_income": np.random.normal(0.28, 0.10, per_segment),
        "loan_history_count": np.random.poisson(2, per_segment),
        "age": np.random.randint(25, 55, per_segment),
        "home_ownership": np.random.choice(["rent", "own", "own", "rent"], per_segment),
        "verified_income": np.random.choice([True, False], per_segment, p=[0.65, 0.35]),
    }

    # Segment 1 – Rising Prime: growing income, strong credit, moderate tenure
    s1 = {
        "income": np.random.normal(85_000, 20_000, per_segment),
        "credit_score": np.random.normal(720, 50, per_segment),
        "employment_years": np.random.exponential(6, per_segment) + 2,
        "debt_to_income": np.random.normal(0.22, 0.08, per_segment),
        "loan_history_count": np.random.poisson(3, per_segment),
        "age": np.random.randint(28, 50, per_segment),
        "home_ownership": np.random.choice(["rent", "own", "own", "own"], per_segment),
        "verified_income": np.random.choice([True, False], per_segment, p=[0.80, 0.20]),
    }

    # Segment 2 – Established Prime: high income, excellent credit, long history
    s2 = {
        "income": np.random.normal(130_000, 35_000, per_segment),
        "credit_score": np.random.normal(780, 40, per_segment),
        "employment_years": np.random.exponential(10, per_segment) + 5,
        "debt_to_income": np.random.normal(0.18, 0.07, per_segment),
        "loan_history_count": np.random.poisson(4, per_segment),
        "age": np.random.randint(35, 60, per_segment),
        "home_ownership": np.random.choice(["own", "own", "own", "mortgage"], per_segment),
        "verified_income": np.random.choice([True, False], per_segment, p=[0.92, 0.08]),
    }

    # Segment 3 – Subprime High-Risk: low income, poor credit, short tenure, high DTI
    s3 = {
        "income": np.random.normal(32_000, 10_000, per_segment),
        "credit_score": np.random.normal(580, 55, per_segment),
        "employment_years": np.random.exponential(2, per_segment) + 0.5,
        "debt_to_income": np.random.normal(0.42, 0.15, per_segment),
        "loan_history_count": np.random.poisson(5, per_segment),
        "age": np.random.randint(20, 50, per_segment),
        "home_ownership": np.random.choice(["rent", "rent", "rent", "own"], per_segment),
        "verified_income": np.random.choice([True, False], per_segment, p=[0.30, 0.70]),
    }

    dfs = []
    for seg_data in [s0, s1, s2, s3]:
        dfs.append(pd.DataFrame(seg_data))

    df = pd.concat(dfs, ignore_index=True)

    # Clip unrealistic negatives
    df["income"] = df["income"].clip(lower=5_000)
    df["credit_score"] = df["credit_score"].clip(lower=300, upper=850)
    df["debt_to_income"] = df["debt_to_income"].clip(lower=0.01)
    df["employment_years"] = df["employment_years"].clip(lower=0.1)
    df["loan_history_count"] = df["loan_history_count"].clip(lower=0)
    df["age"] = df["age"].clip(lower=18)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Encode home_ownership
    df["home_ownership_enc"] = df["home_ownership"].map({"rent": 0, "own": 1, "mortgage": 2})
    df["verified_income"] = df["verified_income"].astype(int)

    return df

if __name__ == "__main__":
    df = generate_customer_data()
    print(f"Generated {len(df)} records")
    print(df.describe())