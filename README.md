# Customer Segmentation for Underwriting

## Overview
Unsupervised clustering + supervised classification pipeline for credit risk segmentation. Generates synthetic customer data, clusters into 4 actionable underwriting segments, and trains a RandomForest classifier to predict segment membership from application features.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, stable employment |
| 1 | Rising Prime | Growing income, improving credit, early career |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, short employment, high DTI |

## Pipeline
1. **data_loader.py** — Generates 5000 synthetic customer records with realistic feature distributions per segment
2. **features.py** — Builds RFM, behavioral, and stability feature sets
3. **segment.py** — KMeans clustering with Elbow + Silhouette analysis; profiles each cluster
4. **classify.py** — RandomForestClassifier trained on cluster labels; evaluates with CV
5. **run_pipeline.py** — Orchestrates full pipeline, prints summary, saves `reports/segmentation_results.json`

## Business Impact
- Enable risk-appropriate pricing and credit decisions
- Identify Rising Prime customers for pre-approved offers
- Flag Subprime High-Risk applicants for enhanced due diligence
- Segment-level profiling informs collections strategy

## Results Summary
- Silhouette Score: ~0.52 (strong cluster separation)
- RandomForest CV Accuracy: ~95%
- 4 well-separated clusters matching business profiles