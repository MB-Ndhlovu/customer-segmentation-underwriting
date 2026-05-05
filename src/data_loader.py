"""Synthetic customer dataset generator for underwriting segmentation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

SEGMENT_CONFIGS = {
    0: {  # Mass Market
        'income': ('uniform', 25000, 65000),
        'credit_score': ('uniform', 580, 700),
        'employment_years': ('uniform', 1, 10),
        'debt_to_income': ('uniform', 0.15, 0.40),
        'loan_history_count': ('uniform', 0, 4),
        'age': ('uniform', 22, 45),
        'home_ownership': ('categorical', [0.6, 0.4]),  # Rent, Own
        'verified_income': ('categorical', [0.5, 0.5]),  # Verified, Not Verified
    },
    1: {  # Rising Prime
        'income': ('uniform', 55000, 95000),
        'credit_score': ('uniform', 680, 780),
        'employment_years': ('uniform', 3, 15),
        'debt_to_income': ('uniform', 0.10, 0.30),
        'loan_history_count': ('uniform', 1, 5),
        'age': ('uniform', 28, 50),
        'home_ownership': ('categorical', [0.5, 0.5]),
        'verified_income': ('categorical', [0.7, 0.3]),
    },
    2: {  # Established Prime
        'income': ('uniform', 80000, 180000),
        'credit_score': ('uniform', 740, 850),
        'employment_years': ('uniform', 8, 30),
        'debt_to_income': ('uniform', 0.05, 0.25),
        'loan_history_count': ('uniform', 2, 7),
        'age': ('uniform', 35, 60),
        'home_ownership': ('categorical', [0.2, 0.8]),
        'verified_income': ('categorical', [0.85, 0.15]),
    },
    3: {  # Subprime High-Risk
        'income': ('uniform', 18000, 40000),
        'credit_score': ('uniform', 450, 600),
        'employment_years': ('uniform', 0, 5),
        'debt_to_income': ('uniform', 0.35, 0.65),
        'loan_history_count': ('uniform', 3, 10),
        'age': ('uniform', 20, 40),
        'home_ownership': ('categorical', [0.85, 0.15]),
        'verified_income': ('categorical', [0.3, 0.7]),
    },
}

SEGMENT_WEIGHTS = [0.35, 0.25, 0.20, 0.20]  # Probability of each segment


def generate_customer_data(n=5000):
    """Generate synthetic customer dataset with realistic underwriting features."""
    segment_labels = np.random.choice(4, size=n, p=SEGMENT_WEIGHTS)
    
    data = {
        'income': np.zeros(n),
        'credit_score': np.zeros(n),
        'employment_years': np.zeros(n),
        'debt_to_income': np.zeros(n),
        'loan_history_count': np.zeros(n),
        'age': np.zeros(n),
        'home_ownership': np.zeros(n),
        'verified_income': np.zeros(n),
        'segment_label': segment_labels,
    }
    
    for seg_id, config in SEGMENT_CONFIGS.items():
        mask = segment_labels == seg_id
        count = mask.sum()
        
        for feat, (dist_type, *params) in config.items():
            if dist_type == 'uniform':
                low, high = params
                data[feat][mask] = np.random.uniform(low, high, count)
            elif dist_type == 'categorical':
                probs = params[0]
                data[feat][mask] = np.random.choice([0, 1], size=count, p=probs)
    
    # Add slight noise for realism
    for col in ['income', 'credit_score', 'employment_years', 'debt_to_income', 'loan_history_count', 'age']:
        data[col] += np.random.normal(0, data[col].std() * 0.02, n)
        data[col] = np.clip(data[col], 0, None)
    
    df = pd.DataFrame(data)
    
    # Round appropriate columns
    df['income'] = df['income'].round(2)
    df['credit_score'] = df['credit_score'].round(0).astype(int)
    df['employment_years'] = df['employment_years'].round(1)
    df['debt_to_income'] = df['debt_to_income'].round(4)
    df['loan_history_count'] = df['loan_history_count'].round(0).astype(int)
    df['age'] = df['age'].round(0).astype(int)
    df['home_ownership'] = df['home_ownership'].astype(int)
    df['verified_income'] = df['verified_income'].astype(int)
    df['segment_label'] = df['segment_label'].astype(int)
    
    return df


if __name__ == '__main__':
    df = generate_customer_data()
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSegment distribution:\n{df['segment_label'].value_counts().sort_index()}")