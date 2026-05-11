# Customer Segmentation for Underwriting

**Project 3 — BSc Mathematical Sciences | Actuarial & Quantitative Finance Track**

---

## Overview

This project builds a **customer segmentation system** for loan underwriting using **KMeans clustering** on synthetic customer data, followed by a **supervised classification model** that predicts segment membership from application features alone.

The goal: enable lenders to automatically classify loan applicants into risk tiers that map to real underwriting decisions.

---

## Segments

| Label | Name | Profile |
|-------|------|---------|
| 0 | **Mass Market** | Young borrowers, low-to-moderate income, short credit history, moderate DTI |
| 1 | **Rising Prime** | Mid-career, growing income, good credit, low DTI, stable employment |
| 2 | **Established Prime** | High income, excellent credit, long tenure, low DTI, asset-rich |
| 3 | **Subprime High-Risk** | High DTI, thin credit file, employment instability, verified income concerns |

---

## Architecture

```
data_loader.py    -> Synthetic 5000-row dataset
features.py       -> RFM, behavioral, stability feature engineering
segment.py        -> KMeans + Elbow + Silhouette + profiling
classify.py       -> RandomForestClassifier on cluster labels
run_pipeline.py   -> Orchestrates full pipeline
```

---

## Business Impact

- Segment predictions enable **instant risk triage** at loan application
- Supervised model allows **real-time scoring** without re-running clustering
- Four-tier system maps directly to: *approve with rate, refer, decline, subprime bucket*

---

## Tech Stack

- Python 3.12
- pandas, numpy, scikit-learn
- matplotlib (visualization)
