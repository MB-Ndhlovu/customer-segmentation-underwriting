# Customer Segmentation for Underwriting

## Overview
Unsupervised KMeans clustering + supervised RandomForest classification to segment loan applicants into 4 risk tiers for underwriting decisions.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low income, mid credit, short employment |
| 1 | Rising Prime | Growing income, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low credit, high DTI, thin file |

## Pipeline
1. Generate 5000-row synthetic customer dataset
2. Feature engineering (RFM, behavioral, stability)
3. KMeans clustering (k=4) with Elbow + Silhouette validation
4. RandomForest classifier trained on cluster labels
5. Segment profiling and business impact summary

## Business Impact
- Instant segment prediction from application features
- Risk-adjusted pricing support
- Underwriting workflow automation

## Results
See `reports/segmentation_results.json` for cluster profiles and model metrics.