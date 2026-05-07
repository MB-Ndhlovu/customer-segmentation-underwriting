# Customer Segmentation for Underwriting

## Overview
This project builds an unsupervised–supervised hybrid segmentation pipeline for credit underwriting. It generates 5,000 synthetic customer records, clusters them into 4 meaningful segments using KMeans, and trains a RandomForest classifier to predict segment membership from application features alone.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, stable employment |
| 1 | Rising Prime | High income, strong credit, long tenure, low DTI |
| 2 | Established Prime | Very high income, excellent credit, homeowners, verified income |
| 3 | Subprime High-Risk | Low income, poor credit, short tenure, high DTI |

## Pipeline
1. `src/data_loader.py` — Generates synthetic customer data (5,000 rows)
2. `src/features.py` — Computes RFM, behavioral, and stability features
3. `src/segment.py` — KMeans clustering with Elbow + Silhouette analysis
4. `src/classify.py` — RandomForest classifier trained on cluster labels
5. `run_pipeline.py` — Orchestrates the full pipeline, saves `reports/segmentation_results.json`

## Business Impact
- Segment labels can be used as underwriting tiers
- The classifier enables instant segment prediction at loan application time
- Risk-based pricing and credit policy can be guided by segment profiles

## Results Summary
See `reports/segmentation_results.json` for cluster centroids, silhouette scores, and classifier metrics.