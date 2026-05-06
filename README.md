# Customer Segmentation for Underwriting

## Overview
This project applies unsupervised learning (K-Means clustering) to segment loan applicants into 4 actionable risk tiers, then trains a supervised classifier to predict segment membership from application features alone.

**Segments:**
- `0 — Mass Market`: Moderate income, decent credit, average stability
- `1 — Rising Prime`: Growing income, improving credit, short employment but upward trajectory
- `2 — Established Prime`: High income, excellent credit, long tenure, low risk
- `3 — Subprime High-Risk`: Low income, poor credit, high DTI, unstable employment

## Tech Stack
- Python 3.12
- pandas, numpy
- scikit-learn (KMeans, RandomForest, preprocessing, metrics)
- faker (synthetic data generation)

## Files
```
src/
  data_loader.py   — Generate 5000-row synthetic dataset
  features.py      — RFM, behavioral, stability feature engineering
  segment.py       — KMeans clustering, silhouette/elbow analysis, profiling
  classify.py      — RandomForestClassifier trained on cluster labels
run_pipeline.py   — End-to-end pipeline orchestration
reports/
  segmentation_results.json — Profile summaries + model metrics
```

## Business Impact
- **Mass Market**: Standard products, moderate rates
- **Rising Prime**: Growth-tier lending, pre-approved upgrades
- **Established Prime**: Premium products, lowest rates
- **Subprime High-Risk**: Require co-signers, higher rates, or decline

## Usage
```bash
pip install -r requirements.txt
python run_pipeline.py
```