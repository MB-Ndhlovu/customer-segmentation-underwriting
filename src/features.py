import numpy as np
import pandas as pd

def build_features(df):
    X = df[[
        "income", "credit_score", "employment_years",
        "debt_to_income", "loan_history_count", "age",
        "home_ownership", "verified_income"
    ]].copy()
    return X

def build_rfm(df):
    rfm = pd.DataFrame(index=df.index)
    rfm["recency"] = np.random.exponential(180, len(df))
    rfm["frequency"] = df["loan_history_count"] + 1
    rfm["monetary"] = df["income"] / 50000
    return rfm

def build_behavioral(df):
    beh = pd.DataFrame(index=df.index)
    beh["loan_density"] = df["loan_history_count"] / (df["age"] - 17 + 1)
    beh["dtl_ratio"] = df["debt_to_income"]
    return beh

def build_stability(df):
    stab = pd.DataFrame(index=df.index)
    stab["emp_stability"] = df["employment_years"] / (df["age"] - 17 + 1)
    stab["verified_flag"] = df["verified_income"]
    stab["homeowner_flag"] = df["home_ownership"]
    return stab