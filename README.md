# Customer Segmentation for Underwriting

## Overview
Unsupervised customer segmentation using KMeans clustering on synthetic lending applicant data. Four actionable segments emerge for risk-based underwriting decisions.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, standard debt levels |
| 1 | Rising Prime | Growing income, improving credit, low debt, stable employment |
| 2 | Established Prime | High income, excellent credit, asset ownership, verified |
| 3 | Subprime High-Risk | Low income, poor credit, high debt-to-income, unstable |

## Pipeline
- `src/data_loader.py` — Generate 5000-row synthetic dataset
- `src/features.py` — Feature engineering (RFM, behavioral, stability)
- `src/segment.py` — KMeans + Elbow/Silhouette analysis
- `src/classify.py` — RandomForest supervised classifier on cluster labels
- `run_pipeline.py` — End-to-end execution

## Business Impact
Segments directly map to underwriting policy tiers:
- **Mass Market** → Standard approval流程
- **Rising Prime** → Pre-approved for better rates
- **Established Prime** → Premium tier, lowest risk
- **Subprime High-Risk** → Manual review required, higher rates

## Results
See `reports/segmentation_results.json` for cluster profiles and model metrics.