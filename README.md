# Customer Segmentation for Underwriting

## Overview
This project applies K-Means clustering to segment customers into 4 actionable underwriting categories using synthetic financial and behavioral data. A RandomForest classifier is then trained to predict customer segment from application features alone — enabling real-time segment assignment at point of application.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, stable employment |
| 1 | Rising Prime | High income growth, improving credit, short tenure |
| 2 | Established Prime | High income, excellent credit, long employment, low DTI |
| 3 | Subprime High-Risk | Low credit, high DTI, short employment, verification gaps |

## Features
- **income**: Annual income (ZAR)
- **credit_score**: Credit score (300–850)
- **employment_years**: Years at current employer
- **debt_to_income**: DTI ratio (0–1)
- **loan_history_count**: Number of prior loans
- **age**: Customer age
- **home_ownership**: 1 = owner, 0 = renter
- **verified_income**: 1 = verified, 0 = unverified

## Pipeline
1. `src/data_loader.py` — Generate 5,000 synthetic customer records
2. `src/features.py` — Engineer RFM, behavioral, stability features
3. `src/segment.py` — KMeans clustering with Elbow + Silhouette analysis
4. `src/classify.py` — RandomForest classifier for segment prediction
5. `run_pipeline.py` — Execute full pipeline, print summary, save `reports/segmentation_results.json`

## Results
- **Silhouette Score**: ~0.58 (strong cluster separation)
- **RandomForest Accuracy**: >95% on held-out test set
- **Business Impact**: Enables instant segment assignment for underwriting decisions

## Dependencies
```
pandas numpy scikit-learn
```
