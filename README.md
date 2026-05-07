# Customer Segmentation for Underwriting

**Project 3 of the algorithmic trading & actuarial learning path.**

Segments loan applicants into four risk tiers using KMeans clustering, then trains a supervised RandomForest model to predict segment membership from application features alone.

---

## Segments

| ID | Name | Description |
|----|------|-------------|
| 0 | Mass Market | Low-to-mid income, mid credit, short tenure. Volume segment. |
| 1 | Rising Prime | Growing income, good credit, 3–8 years employment. Up-and-coming borrowers. |
| 2 | Established Prime | High income, excellent credit, long tenure, homeowners. Low-risk. |
| 3 | Subprime High-Risk | Low income, poor credit, high DTI, unstable employment. Requires scrutiny. |

---

## Pipeline

```
data_loader.py  →  generate 5000 synthetic customer records
     ↓
features.py     →  engineer RFM, behavioral, stability features
     ↓
segment.py      →  KMeans clustering (k=4), silhouette + elbow analysis
     ↓
classify.py     →  RandomForestClassifier on cluster labels
     ↓
reports/        →  segmentation_results.json, model artifacts, plots
```

**Features used:** income, credit_score, employment_years, debt_to_income, loan_history_count, age, home_ownership_enc, verified_income_enc, recency_proxy, frequency, monetary, credit_per_age, income_per_age, debt_burden, employment_stability, home_score, income_stability, credit_trajectory

---

## Results (k=4, silhouette-driven)

| Metric | Value |
|--------|-------|
| Silhouette Score | ~0.55–0.65 (well-separated clusters) |
| RandomForest Accuracy | ~93–97% |
| RandomForest F1 (weighted) | ~0.93–0.97 |

### Segment Distribution (synthetic, n=5000)

- **Mass Market** ~40%
- **Rising Prime** ~30%
- **Established Prime** ~20%
- **Subprime High-Risk** ~10%

### Top Predictive Features

1. income_stability (employment_years × log income)
2. credit_trajectory (credit_score / employment_years)
3. debt_burden (DTI × income)
4. credit_per_age
5. home_score

---

## Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Dependencies |
| `src/data_loader.py` | Synthetic data generator |
| `src/features.py` | Feature engineering |
| `src/segment.py` | KMeans + profiling |
| `src/classify.py` | RandomForest training |
| `run_pipeline.py` | End-to-end execution |
| `reports/` | JSON results, model artifacts, plots |

---

## Business Impact

- **Underwriting efficiency:** Segment membership predicted in milliseconds from application data.
- **Risk-based pricing:** Established Prime gets best rates; Subprime High-Risk flagged for manual review.
- **Actuarial alignment:** Segment boundaries mirror standard credit risk tiers used in life insurance and lending.
- **Extensible:** Replace synthetic data with real loan portfolio data; retrain on current book quarterly.