# Customer Segmentation for Underwriting

A machine learning pipeline that segments loan applicants into 4 distinct risk categories using KMeans clustering, then trains a supervised classifier for real-time segment prediction.

## Segments

| Segment | Description | Risk Profile |
|---------|-------------|--------------|
| **Mass Market** | Largest segment (~45%), moderate income/credit | Standard underwriting |
| **Rising Prime** | Growing borrowers with improving credit | Favorable terms eligible |
| **Established Prime** | High income, strong credit history | Premium products |
| **Subprime High-Risk** | Low credit scores, high DTI, unstable employment | Enhanced scrutiny required |

## Architecture

```
run_pipeline.py
├── src/data_loader.py   — Synthetic dataset (5000 rows)
├── src/features.py      — RFM, behavioral, stability features
├── src/segment.py       — KMeans clustering + profiling
├── src/classify.py      — RandomForest supervised classifier
└── reports/
    └── segmentation_results.json
```

## Features

**Raw Features:** income, credit_score, employment_years, debt_to_income, loan_history_count, age, home_ownership, verified_income

**Engineered Features:**
- RFM: loan_frequency, income_per_employment_year, credit_per_age
- Behavioral: debt_burden, employment_stability, credit_to_income_ratio, loan_density
- Stability: income_stability_score, credit_quality_indicator, debt_capacity

## Results

- **Dataset:** 5,000 synthetic customers across 4 segments
- **Clustering:** KMeans (k=4) with StandardScaler normalization
- **Classifier:** RandomForest (100 estimators, ~95%+ accuracy)
- **Target:** segment_label (0-3 from KMeans → actual segment name)

## Usage

```bash
cd /home/workspace/Projects/customer-segmentation-underwriting
pip install -r requirements.txt
python run_pipeline.py
```

## Business Impact

- Enables real-time segment prediction at loan application intake
- Supports differentiated underwriting strategies per segment
- Identifies high-risk applicants early in the pipeline
- Targets premium products to established prime customers