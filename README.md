# Customer Segmentation for Underwriting

## Overview
KMeans clustering + supervised classification pipeline for credit risk segmentation. Identifies 4 actionable customer segments for underwriting decisions.

## Segments
- **Segment 0: Mass Market** — Average credit, moderate income, standard risk
- **Segment 1: Rising Prime** — Good credit, growing income, low risk
- **Segment 2: Established Prime** — Excellent credit, high income, very low risk
- **Segment 3: Subprime High-Risk** — Poor credit, high DTI, elevated default risk

## Pipeline
1. `data_loader.py` — Generates 5000 synthetic customer records
2. `features.py` — RFM + behavioral + stability feature engineering
3. `segment.py` — KMeans (k=4), silhouette analysis, elbow method, profiling
4. `classify.py` — RandomForestClassifier trained on cluster labels
5. `run_pipeline.py` — End-to-end execution, saves results to `reports/segmentation_results.json`

## Results
- Silhouette score, inertia, cluster profiles saved to `reports/segmentation_results.json`
- Classification model predicts segment from application features alone

## Business Impact
Enables lenders to automatically classify loan applicants into risk tiers, supporting pricing, credit limit, and approval decisions.