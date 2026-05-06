import numpy as np
import pandas as pd

def generate_synthetic_data(n=5000, seed=42):
    np.random.seed(seed)
    
    segments = {
        0: dict(name="Mass Market",        size=0.30, income=(30000, 65000), credit=(620, 720), emp=(1, 10), dti=(0.10, 0.35), loans=(0, 3), age=(22, 45), home=(0.4, 0.7), verified=(0.3, 0.7)),
        1: dict(name="Rising Prime",        size=0.30, income=(55000, 95000), credit=(680, 780), emp=(2, 12), dti=(0.15, 0.40), loans=(1, 5), age=(26, 50), home=(0.5, 0.8), verified=(0.5, 0.9)),
        2: dict(name="Established Prime",   size=0.25, income=(80000, 150000), credit=(740, 850), emp=(5, 30), dti=(0.10, 0.30), loans=(2, 8), age=(32, 60), home=(0.7, 1.0), verified=(0.7, 1.0)),
        3: dict(name="Subprime High-Risk",  size=0.15, income=(18000, 45000), credit=(500, 640), emp=(0, 5), dti=(0.30, 0.65), loans=(3, 12), age=(18, 40), home=(0.1, 0.4), verified=(0.1, 0.4)),
    }

    rows = []
    for label, info in segments.items():
        n_seg = int(n * info["size"])
        for _ in range(n_seg):
            income = np.random.uniform(*info["income"])
            credit_score = int(np.random.normal(np.mean(info["credit"]), (info["credit"][1] - info["credit"][0]) / 4))
            credit_score = max(300, min(850, credit_score))
            employment_years = max(0, np.random.uniform(*info["emp"]))
            debt_to_income = np.random.uniform(*info["dti"])
            loan_history_count = np.random.randint(info["loans"][0], info["loans"][1] + 1)
            age = np.random.randint(info["age"][0], info["age"][1] + 1)
            home_ownership = int(np.random.random() < np.random.uniform(*info["home"]))
            verified_income = int(np.random.random() < np.random.uniform(*info["verified"]))
            rows.append({
                "income": round(income, 2),
                "credit_score": credit_score,
                "employment_years": round(employment_years, 2),
                "debt_to_income": round(debt_to_income, 4),
                "loan_history_count": loan_history_count,
                "age": age,
                "home_ownership": home_ownership,
                "verified_income": verified_income,
                "segment_label": label,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    print(f"Generated {len(df)} rows")
    print(df["segment_label"].value_counts().sort_index())
    print(df.describe())