# Customer Segmentation for Underwriting

## Overview
Unsupervised clustering + supervised classification pipeline for borrower risk segmentation in life insurance underwriting. Identifies 4 distinct customer segments that map to actionable lending risk tiers.

## Methodology
1. **Synthetic Data Generation** (`src/data_loader.py`): 5000 borrower profiles with realistic feature distributions across 4 clusters
2. **Feature Engineering** (`src/features.py`): RFM, behavioral, and stability features
3. **Clustering** (`src/segment.py`): KMeans with Elbow method + silhouette analysis to determine optimal k=4
4. **Classification** (`src/classify.py`): RandomForest trained on cluster labels for real-time segment prediction

## Customer Segments
| Label | Segment | Risk Profile |
|-------|---------|-------------|
| 0 | Mass Market | Entry-level borrowers, moderate income, standard credit |
| 1 | Rising Prime | Growing income, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, strong financial footprint |
| 3 | Subprime High-Risk | Elevated DTI, limited credit history, higher default risk |

## Key Results
- **Silhouette Score**: ~0.58 (strong cluster separation)
- **RandomForest Accuracy**: ~96% on held-out test set
- **Features Used**: income, credit_score, employment_years, debt_to_income, loan_history_count, age, home_ownership, verified_income

## Business Impact
- Enables risk-appropriate pricing at point of application
- Reduces underwriter review load via automated segment assignment
- Supports actuarial pricing models with data-driven segment assumptions

## Files
```
customer-segmentation-underwriting/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── reports/
│   └── segmentation_results.json
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── features.py
    ├── segment.py
    └── classify.py
```