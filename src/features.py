import numpy as np
import pandas as pd

def build_features(df):
    """Build RFM, behavioral, and stability features."""
    feat = pd.DataFrame(index=df.index)

    # RFM-style proxies
    feat['income_log'] = np.log1p(df['income'])
    feat['credit_score_norm'] = (df['credit_score'] - 300) / 550  # 0-1 scale
    feat['employment_log'] = np.log1p(df['employment_years'])

    # Behavioral signals
    feat['loan_density'] = df['loan_history_count'] / (df['age'] - 18 + 1)  # loans per eligible year
    feat['debt_burden'] = df['debt_to_income']

    # Stability signals
    feat['job_stability'] = df['employment_years'] / (df['age'] - 18 + 1)
    feat['income_stability'] = feat['income_log'] * (1 - feat['debt_burden'])  # high income, low debt = stable

    # Interaction features
    feat['credit_x_income'] = feat['credit_score_norm'] * feat['income_log']
    feat['credit_x_stability'] = feat['credit_score_norm'] * feat['job_stability']

    return feat

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    df = generate_synthetic_data()
    X = build_features(df)
    print(X.describe())