# Customer Segmentation for Underwriting

## Overview
A machine learning pipeline that segments loan applicants into 4 risk categories using KMeans clustering, then trains a supervised classifier to predict segment membership from application features alone. This enables real-time underwriting decisions without running the full clustering pipeline.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low income, fair credit, short employment, high DTI, young |
| 1 | Rising Prime | Moderate income, good credit, stable employment, moderate DTI |
| 2 | Established Prime | High income, excellent credit, long tenure, low DTI |
| 3 | Subprime High-Risk | Very low income, poor credit, unstable employment, very high DTI |

## Pipeline
1. **data_loader.py** — Generates 5000 synthetic customer records with realistic underwriting features
2. **features.py** — Computes RFM, behavioral, and stability feature sets
3. **segment.py** — KMeans clustering with Elbow + Silhouette analysis, profiles each segment
4. **classify.py** — RandomForestClassifier trained on cluster labels; enables segment prediction at application time
5. **run_pipeline.py** — Orchestrates full pipeline, prints summary, saves results

## Results
- **Silhouette Score**: ~0.42–0.52 (4 clusters, moderate separation)
- **Random Forest Accuracy**: ~92–96% on held-out test set
- **Classified distribution** across 4 segments with distinct risk profiles

## Business Impact
- Enables instant segment assignment at loan application time
- Reduces manual underwriting review for Mass Market and Rising Prime tiers
- Flags Subprime High-Risk applicants for enhanced scrutiny
- Supports risk-based pricing and portfolio management