import numpy as np
import pandas as pd

def compute_rfm_features(df):
    """RFM features based on loan history and behavior"""
    features = pd.DataFrame(index=df.index)
    features['loan_frequency'] = df['loan_history_count'] / (df['age'] - 18 + 1)
    features['income_per_employment_year'] = df['income'] / (df['employment_years'] + 1)
    features['credit_per_age'] = df['credit_score'] / (df['age'] - 18 + 1)
    return features

def compute_behavioral_features(df):
    """Behavioral features derived from raw inputs"""
    features = pd.DataFrame(index=df.index)
    features['debt_burden'] = df['debt_to_income'] * df['income']
    features['employment_stability'] = df['employment_years'] / (df['age'] - 18 + 1)
    features['credit_to_income_ratio'] = df['credit_score'] / (df['income'] / 10000)
    features['loan_density'] = df['loan_history_count'] / (df['employment_years'] + 0.5)
    return features

def compute_stability_features(df):
    """Stability features for underwriting"""
    features = pd.DataFrame(index=df.index)
    features['income_stability_score'] = (
        df['verified_income'] * 0.4 +
        (df['home_ownership'] * 0.3) +
        (df['employment_years'] / 20 * 0.3)
    )
    features['credit_quality_indicator'] = df['credit_score'] / 850
    features['debt_capacity'] = 1 - df['debt_to_income']
    return features

def build_feature_matrix(df):
    """Combine all feature sets with original features"""
    rfm = compute_rfm_features(df)
    behavioral = compute_behavioral_features(df)
    stability = compute_stability_features(df)

    feature_df = pd.concat([df[['income', 'credit_score', 'employment_years',
                                 'debt_to_income', 'loan_history_count', 'age',
                                 'home_ownership', 'verified_income']], rfm, behavioral, stability], axis=1)
    return feature_df

if __name__ == "__main__":
    from data_loader import generate_synthetic_data
    df = generate_synthetic_data()
    features = build_feature_matrix(df)
    print(f"Feature matrix shape: {features.shape}")
    print(features.describe())