# Customer Segmentation for Underwriting

**Project 3 — ML Pipeline for Credit Risk Tiering**

## Overview
This project implements a customer segmentation pipeline for underwriting decisions. Using KMeans clustering on synthetic financial data, we segment 5,000 customers into 4 actionable risk tiers: **Mass Market**, **Rising Prime**, **Established Prime**, and **Subprime High-Risk**.

A supervised classifier (RandomForest) is then trained on cluster labels to enable real-time segment prediction from raw application features.

## Segments
| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Moderate income, average credit, standard employment |
| 1 | Rising Prime | Growing income, improving credit, short employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, many loans |

## Features Used
- `income` — Annual income (USD)
- `credit_score` — FICO-equivalent score (300–850)
- `employment_years` — Years at current employer
- `debt_to_income` — Monthly debt / monthly income ratio
- `loan_history_count` — Total previous loans
- `age` — Customer age
- `home_ownership` — Binary (1 = owns, 0 = rents)
- `verified_income` — Binary (1 = verified, 0 = unverified)

## Pipeline
1. **data_loader** — Generates 5,000 synthetic customer records
2. **features** — Computes RFM, behavioral, and stability features
3. **segment** — KMeans clustering + silhouette/elbow analysis → cluster labels
4. **classify** — RandomForest trained on cluster labels for segment prediction
5. **run_pipeline** — Orchestrates full pipeline, saves artifacts to `reports/`

## Business Impact
- **Risk-based pricing**: Match interest rates to segment risk
- **Fraud triage**: High-risk segments get extra scrutiny
- **Credit policy**: Segment profiles inform approval thresholds
- **Portfolio monitoring**: Track segment drift over time

## Files
```
customer-segmentation-underwriting/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── features.py
│   ├── segment.py
│   └── classify.py
└── reports/
    └── segmentation_results.json
```

## Usage
```bash
cd /home/workspace/Projects/customer-segmentation-underwriting
pip install -r requirements.txt
python run_pipeline.py
```