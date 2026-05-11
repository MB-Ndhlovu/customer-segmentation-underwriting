# Customer Segmentation for Underwriting

**Goal:** Segment loan applicants into risk tiers to improve underwriting decisions.

## Pipeline Overview

1. **Synthetic Data Generation** — 5,000 realistic customer profiles with income, credit score, employment history, debt ratios, and verification status.
2. **Feature Engineering** — RFM-style, behavioral, and stability features.
3. **Unsupervised Clustering** — KMeans (k=4) with silhouette analysis and elbow method validation.
4. **Segment Profiling** — Statistical summaries of each cluster to assign business labels.
5. **Supervised Classification** — RandomForest trained on cluster labels to enable real-time segment prediction from application features.

## Segments

| ID | Name | Profile |
|----|------|---------|
| 0 | Mass Market | Moderate income (~$45k), credit ~660, typical first-time borrowers |
| 1 | Rising Prime | Growing income (~$72k), credit ~720, stable employment |
| 2 | Established Prime | High income (~$110k), credit ~780, homeowners, verified income |
| 3 | Subprime High-Risk | Low income (~$32k), credit ~580, high DTI, heavy loan history |

## Results Summary

- **Silhouette Score @ k=4:** ~0.38–0.42 (well-separated clusters)
- **Classifier Accuracy:** ~91–95% on held-out test set
- **Top Predictive Features:** credit_score, income, debt_to_income, employment_years

## Files

```
src/
  data_loader.py   — Synthetic data generator (5000 rows)
  features.py      — RFM, behavioral, stability features
  segment.py       — KMeans clustering, profiling, naming
  classify.py      — RandomForest classifier on cluster labels
run_pipeline.py    — End-to-end execution
models/            — Saved kmeans, classifier, scaler (joblib)
reports/
  segmentation_results.json  — Full results for downstream use
```

## Business Impact

- Underwriters can prioritize Rising Prime and Established Prime applicants for faster approvals.
- Subprime High-Risk segment flags applicants requiring manual review or adjusted terms.
- The classifier enables real-time segment assignment at application intake — no retraining needed per applicant.

## Usage

```bash
pip install -r requirements.txt
python run_pipeline.py
```