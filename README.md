# Customer Segmentation for Underwriting

**Project 3 of the Actuarial AI Pipeline**

---

## Overview

This project builds an unsupervised + supervised pipeline to segment loan applicants into risk tiers for underwriting decisions. It uses KMeans clustering on engineered features to identify 4 distinct customer segments, then trains a RandomForest classifier to predict segment membership from application features alone.

---

## Segments

| Label | Segment Name | Profile |
|-------|--------------|---------|
| 0 | Mass Market | Low-to-medium income, average credit, short tenure, moderate DTI |
| 1 | Rising Prime | Medium income, improving credit, mid career, low DTI |
| 2 | Established Prime | High income, excellent credit, long tenure, very low DTI |
| 3 | Subprime High-Risk | Low income, poor credit, unstable employment, high DTI |

---

## Pipeline

```
Synthetic Data → Feature Engineering → KMeans Clustering → Segment Profiling
                                                           → RandomForest Classifier (supervised)
```

**Features used:**
- `income` — annual income
- `credit_score` — VantageScore-style (300–850)
- `employment_years` — tenure at current employer
- `debt_to_income` — monthly debt payments / gross income
- `loan_history_count` — number of prior loans
- `age` — applicant age
- `home_ownership` — 0=rent, 1=own
- `verified_income` — binary

---

## Results Summary

- **Silhouette Score:** ~0.45–0.55 (varies by run — synthetic data)
- **KMeans k=4** chosen via elbow method + silhouette validation
- **RandomForest accuracy:** 97–99% on held-out test set
- Segment distribution: roughly 25% each (balanced synthetic generation)

---

## Business Impact

- Underwriters can pre-screen applications by segment before full review
- High-risk segment flagged for manual underwriting
- Prime segments eligible for automated approval workflows
- Segment labels feed downstream credit policy rules

---

## Files

```
.
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # 5000-row synthetic dataset generator
│   ├── features.py          # RFM, behavioral, stability features
│   ├── segment.py           # KMeans + elbow + silhouette + profiling
│   └── classify.py         # RandomForest supervised classifier
└── reports/
    └── segmentation_results.json
```

---

## Running

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Output: `reports/segmentation_results.json`