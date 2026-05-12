# Customer Segmentation for Underwriting

## Overview
Machine learning pipeline that segments loan applicants into 4 risk tiers using KMeans clustering, then trains a supervised classifier to predict segment membership from application features alone.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low-to-medium income, average credit, standard employment |
| 1 | Rising Prime | Growing income, improving credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, unstable employment |

## Pipeline
1. `src/data_loader.py` — Generates 5,000 synthetic customer records
2. `src/features.py` — Builds RFM, behavioral, and stability feature sets
3. `src/segment.py` — KMeans clustering with Elbow + Silhouette analysis
4. `src/classify.py` — RandomForestClassifier trained on cluster labels
5. `run_pipeline.py` — Orchestrates the full pipeline

## Results
- **Silhouette Score**: ~0.45 (good cluster separation)
- **Classification Accuracy**: >92% on held-out test set
- **Segment distribution**: Realistic tier spread across 4 groups

## Business Impact
- Risk-adjusted pricing by segment
- Faster preliminary underwriting decisions
- Reduced manual review burden for low-risk applicants