# Customer Segmentation for Underwriting

**Project 3** — Machine learning pipeline for customer risk segmentation using KMeans clustering and supervised classification.

## Overview

This project segments loan applicants into 4 distinct risk tiers for underwriting decisions:
- **Mass Market** — Low-to-moderate risk, broad customer base
- **Rising Prime** — Growing creditworthiness, upward trajectory
- **Established Prime** — High credit quality, stable borrowers
- **Subprime High-Risk** — Elevated default risk, requires careful evaluation

## Pipeline

```
data_loader.py   → Synthetic customer data (5000 rows)
     ↓
features.py      → RFM, behavioral, stability feature engineering
     ↓
segment.py       → KMeans clustering + Elbow/Silhouette analysis
     ↓
classify.py      → RandomForest classifier trained on cluster labels
     ↓
run_pipeline.py  → Orchestrates full pipeline, saves results
```

## Business Impact

- **Underwriting efficiency** — Instant segment prediction for new applications
- **Risk-adjusted pricing** — Segment-specific loan terms and rates
- **Portfolio monitoring** — Track segment drift over time
- **Regulatory compliance** — Transparent, auditable scoring logic

## Results Summary

| Segment | Description | Approx. Size |
|---------|-------------|--------------|
| 0 | Mass Market | ~35% |
| 1 | Rising Prime | ~25% |
| 2 | Established Prime | ~20% |
| 3 | Subprime High-Risk | ~20% |

> Run `python run_pipeline.py` to generate the full segmentation report.