# Customer Segmentation for Underwriting

## Overview
This project builds an unsupervised clustering pipeline to segment loan applicants into 4 risk-tiered groups, then trains a supervised classifier to predict segment membership from application features alone.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low income, mid credit, stable employment, low DTI |
| 1 | Rising Prime | Moderate income, good credit, short employment, moderate DTI |
| 2 | Established Prime | High income, excellent credit, long tenure, low DTI |
| 3 | Subprime High-Risk | Low income, poor credit, unstable employment, high DTI |

## Pipeline
1. `src/data_loader.py` - Generates 5,000 synthetic customer records with realistic feature distributions
2. `src/features.py` - Builds RFM, behavioral, and stability feature sets
3. `src/segment.py` - KMeans clustering (k=4) with Elbow method + silhouette analysis
4. `src/classify.py` - RandomForestClassifier trained on cluster labels; evaluates precision/recall/F1 per segment
5. `run_pipeline.py` - Orchestrates the full pipeline end-to-end

## Business Impact
- Underwriters can quickly assess applicant risk tier from application data alone
- Segment-level default rate estimates enable risk-adjusted pricing
- Supervised model allows instant scoring at application intake
