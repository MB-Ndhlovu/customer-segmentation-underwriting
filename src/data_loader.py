import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

def generate_customer_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    # 4 cluster centers representing the 4 segments
    # Mass Market (0): moderate income, average credit
    # Rising Prime (1): growing income, improving credit
    # Established Prime (2): high income, excellent credit
    # Subprime High-Risk (3): low income, poor credit, high DTI

    centers = {
        0: {'income': 55000, 'credit_score': 680, 'employment_years': 5,
            'debt_to_income': 0.25, 'loan_history_count': 2, 'age': 35, 'home_ownership': 0.5, 'verified_income': 0.7},
        1: {'income': 75000, 'credit_score': 720, 'employment_years': 7,
            'debt_to_income': 0.20, 'loan_history_count': 3, 'age': 38, 'home_ownership': 0.6, 'verified_income': 0.85},
        2: {'income': 120000, 'credit_score': 790, 'employment_years': 12,
            'debt_to_income': 0.15, 'loan_history_count': 4, 'age': 45, 'home_ownership': 0.9, 'verified_income': 0.98},
        3: {'income': 28000, 'credit_score': 580, 'employment_years': 2,
            'debt_to_income': 0.45, 'loan_history_count': 5, 'age': 28, 'home_ownership': 0.2, 'verified_income': 0.4},
    }

    blobs, true_labels = make_blobs(
        n_samples=n_samples,
        centers=[list(centers[i].values()) for i in range(4)],
        cluster_std=[8000, 10000, 15000, 6000],
        random_state=random_state,
        n_features=8
    )

    feature_names = ['income', 'credit_score', 'employment_years', 'debt_to_income',
                     'loan_history_count', 'age', 'home_ownership', 'verified_income']

    df = pd.DataFrame(blobs, columns=feature_names)

    # Clip and correct unrealistic values
    df['income'] = df['income'].clip(15000, 300000)
    df['credit_score'] = df['credit_score'].clip(500, 850)
    df['employment_years'] = df['employment_years'].clip(0, 40).round()
    df['debt_to_income'] = df['debt_to_income'].clip(0.0, 0.9)
    df['loan_history_count'] = df['loan_history_count'].clip(0, 15).round()
    df['age'] = df['age'].clip(18, 75).round()
    df['home_ownership'] = df['home_ownership'].clip(0, 1)
    df['verified_income'] = df['verified_income'].clip(0, 1)

    df['segment_label'] = true_labels

    # Map labels to meaningful names
    segment_names = {0: 'Mass Market', 1: 'Rising Prime', 2: 'Established Prime', 3: 'Subprime High-Risk'}
    df['segment_name'] = df['segment_label'].map(segment_names)

    return df

if __name__ == '__main__':
    df = generate_customer_data()
    print(df.head())
    print(df.shape)
    print(df['segment_name'].value_counts())