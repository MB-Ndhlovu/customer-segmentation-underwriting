# Customer Segmentation for Underwriting

## Overview
This project builds a customer segmentation pipeline for lending underwriting using unsupervised clustering (KMeans) followed by supervised classification (RandomForest). The goal is to segment loan applicants into meaningful risk tiers that a lender would actually use in practice.

## Segments
| Label | Segment | Description |
|-------|---------|-------------|
| 0 | Mass Market | Low income, average credit, high DTI, young renters |
| 1 | Rising Prime | Medium income, growing credit, moderate DTI, stable employment |
| 2 | Established Prime | High income, excellent credit, low DTI, homeowners, verified income |
| 3 | Subprime High-Risk | Variable income, poor credit, very high DTI, unstable history |

## Methodology
1. **Synthetic Data Generation**: 5,000 customer records with realistic distributions across income, credit score, employment, debt, and loan history.
2. **Feature Engineering**: RFM features, behavioral signals, stability metrics.
3. **Clustering**: KMeans with elbow method and silhouette analysis to determine 4 clusters.
4. **Classification**: RandomForest trained on cluster labels to predict segment from application features.

## Files
- `src/data_loader.py` — Synthetic dataset generation
- `src/features.py` — Feature engineering
- `src/segment.py` — KMeans clustering, profiling
- `src/classify.py` — RandomForest classifier
- `run_pipeline.py` — Full pipeline execution

## Business Impact
- Risk-based pricing per segment
- Faster underwriting decisions via classification model
- Clear segment profiles for portfolio management