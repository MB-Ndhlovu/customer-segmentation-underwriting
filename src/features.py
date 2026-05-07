import numpy as np

def compute_rfm_features(df):
    """Recency, Frequency, Monetary proxies from static features."""
    recency_proxy = (df['age'].max() - df['age']) / 50  # older = lower recency score
    frequency_proxy = df['loan_history_count'] / df['loan_history_count'].max()
    monetary_proxy = df['income'] / df['income'].max()
    return recency_proxy, frequency_proxy, monetary_proxy

def compute_behavioral_features(df):
    """Behavioral signals from existing columns."""
    credit_utilization = (850 - df['credit_score']) / 350  # higher when score is lower
    loan_density = df['loan_history_count'] / (df['age'] - 18 + 1)  # loans per eligible year
    employment_stability = df['employment_years'] / (df['age'] - 18 + 1)  # tenure ratio
    return credit_utilization, loan_density, employment_stability

def compute_stability_features(df):
    """Stability signals."""
    income_stability = df['verified_income'] * (df['income'] / df['income'].mean())
    home_ownership_bonus = df['home_ownership'] * 0.5
    tenure_score = np.tanh(df['employment_years'] / 10)  # saturating score
    return income_stability, home_ownership_bonus, tenure_score

def build_feature_matrix(df):
    recency, frequency, monetary = compute_rfm_features(df)
    credit_util, loan_density, emp_stability = compute_behavioral_features(df)
    income_stab, home_bonus, tenure = compute_stability_features(df)

    features = np.column_stack([
        df['income'].values,
        df['credit_score'].values,
        df['employment_years'].values,
        df['debt_to_income'].values,
        df['loan_history_count'].values,
        df['age'].values,
        df['home_ownership'].values,
        df['verified_income'].values,
        recency.values,
        frequency.values,
        monetary.values,
        credit_util.values,
        loan_density.values,
        emp_stability.values,
        income_stab.values,
        home_bonus.values,
        tenure.values
    ])

    feature_names = [
        'income', 'credit_score', 'employment_years', 'debt_to_income',
        'loan_history_count', 'age', 'home_ownership', 'verified_income',
        'recency_proxy', 'frequency_proxy', 'monetary_proxy',
        'credit_utilization', 'loan_density', 'employment_stability',
        'income_stability', 'home_ownership_bonus', 'tenure_score'
    ]
    return features, feature_names