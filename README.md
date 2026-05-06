# Customer Segmentation for Underwriting

**Project 3 of the ML pipeline** — Unsupervised clustering + supervised classification for credit risk segmentation.

## Overview

This project implements a customer segmentation pipeline for lenders. It uses KMeans clustering to identify 4 distinct borrower segments, then trains a RandomForest classifier to predict segment membership from application features alone.

## Segments

| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low-mid income, average credit, short employment |
| 1 | Rising Prime | Mid income, improving credit, stable employment |
| 2 | Established Prime | High income, strong credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, many past loans |

## Pipeline

```
data_loader.py  →  features.py  →  segment.py  →  classify.py
     (5000 rows)       (RFM,           (KMeans,         (RandomForest,
                      behavioral,      silhouette,      predict segment)
                      stability)       elbow)
```

## Results

- **Silhouette Score**: ~0.42 (4 clusters)
- ** supervised Accuracy**: ~91% (RandomForest on cluster labels)
- **Features used**: income, credit_score, employment_years, debt_to_income, loan_history_count, age, home_ownership, verified_income

## Business Impact

- Underwriters can instantly classify new applicants into risk tiers
- Segment-level pricing and policy rules become data-driven
- Reduces manual underwriting workload for low-risk segments

## Files

- `src/data_loader.py` — Synthetic dataset generator (5000 rows)
- `src/features.py` — Feature engineering (RFM, behavioral, stability)
- `src/segment.py` — KMeans clustering + evaluation
- `src/classify.py` — RandomForest classifier on cluster labels
- `run_pipeline.py` — End-to-end execution + JSON report
- `reports/segmentation_results.json` — Pipeline output