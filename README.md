# Customer Segmentation for Underwriting

## Overview
Machine learning pipeline that segments loan applicants into 4 risk categories using KMeans clustering and a supervised classifier for real-time prediction.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low-mid income, average credit, young |
| 1 | Rising Prime | Moderate income, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, verified income issues |

## Pipeline
1. `src/data_loader.py` — Generates 5000 synthetic customer records
2. `src/features.py` — RFM, behavioral, and stability feature engineering
3. `src/segment.py` — KMeans clustering with Elbow + Silhouette analysis
4. `src/classify.py` — RandomForest classifier trained on cluster labels
5. `run_pipeline.py` — Executes full pipeline, outputs summary

## Business Impact
- Risk-adjusted lending decisions at point of application
- Segment-level loss expectations for pricing
- Explainable segment assignment via feature importance

## Results
See `reports/segmentation_results.json` for cluster profiles, silhouette score, and classifier metrics.