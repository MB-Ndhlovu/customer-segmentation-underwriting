# Customer Segmentation for Underwriting

## Overview
This project builds a customer segmentation system for loan underwriting using unsupervised learning (KMeans clustering) combined with supervised classification. The pipeline enables lenders to categorize loan applicants into actionable risk segments.

## Segments
The model identifies 4 distinct customer segments:

| Segment | Profile |
|---------|---------|
| **0 — Mass Market** | Moderate income, average credit, standard employment |
| **1 — Rising Prime** | Growing income, improving credit, stable employment |
| **2 — Established Prime** | High income, strong credit, long tenure, verified assets |
| **3 — Subprime High-Risk** | Low income, poor credit, short employment, high DTI |

## Pipeline
1. **Data Generation** — Synthetic dataset of 5,000 customers with realistic financial profiles
2. **Feature Engineering** — RFM, behavioral, and stability features
3. **Segmentation** — KMeans clustering with Elbow method and silhouette analysis to determine optimal k
4. **Classification** — RandomForestClassifier trained on cluster labels for fast segment prediction

## Results
- Silhouette Score: ~0.XX (varies by run)
- Best K determined via Elbow method
- Supervised model accuracy: ~XX% on held-out test set

## Business Impact
- Enables real-time applicant screening
- Supports risk-based pricing
- Identifies growth segments (Rising Prime)
- Flags high-risk applicants (Subprime) for manual review

## Files
- `src/data_loader.py` — Synthetic data generator
- `src/features.py` — Feature engineering pipeline
- `src/segment.py` — KMeans clustering + evaluation
- `src/classify.py` — RandomForestClassifier on cluster labels
- `run_pipeline.py` — Orchestrates the full pipeline
- `reports/segmentation_results.json` — Saved metrics and segment profiles