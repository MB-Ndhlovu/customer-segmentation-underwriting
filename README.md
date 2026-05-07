# Customer Segmentation for Underwriting

## Overview
Unsupervised KMeans clustering + supervised classification pipeline for customer segmentation in life insurance underwriting. Produces 4 actionable segments from synthetic applicant data.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, stable employment |
| 1 | Rising Prime | Growing income, strong credit, moderate employment tenure |
| 2 | Established Prime | High income, excellent credit, long employment history |
| 3 | Subprime High-Risk | Low income, poor credit, short employment, high DTI |

## Pipeline
1. **Data Generation** (`src/data_loader.py`) — 5000 synthetic customers
2. **Feature Engineering** (`src/features.py`) — RFM, behavioral, stability features
3. **Clustering** (`src/segment.py`) — KMeans (k=4), silhouette analysis, Elbow method
4. **Classification** (`src/classify.py`) — RandomForest on cluster labels
5. **run_pipeline.py** — Orchestrates full pipeline

## Results
- Clustering silhouette score and cluster profiles saved to `reports/segmentation_results.json`
- Classification model provides segment inference from raw application features

## Dependencies
```
pandas
numpy
scikit-learn
```