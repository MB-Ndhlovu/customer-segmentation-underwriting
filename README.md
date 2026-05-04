# Customer Segmentation for Underwriting

## Overview

This project builds a customer segmentation system for lending underwriting using unsupervised clustering (KMeans) followed by a supervised classifier. The pipeline enables lenders to classify incoming loan applications into one of four risk segments based on financial behavior features.

**Segments:**
- **Mass Market** — Broad, average-risk customers with moderate income and credit
- **Rising Prime** — Young, upwardly mobile borrowers with growing credit profiles
- **Established Prime** — High earners with long credit history and stable finances
- **Subprime High-Risk** — High debt-to-income, many recent loans, credit concerns

## Architecture

```
data_loader.py   → Synthetic customer data (5000 rows, 8 features)
       ↓
features.py      → RFM, behavioral, and stability feature engineering
       ↓
segment.py       → KMeans clustering (k=4) via Elbow + Silhouette analysis
       ↓
classify.py      → RandomForest classifier trained on cluster labels
       ↓
run_pipeline.py  → End-to-end execution, summary report, JSON artifact
```

## Results

| Segment | Count | Avg Income | Avg Credit Score | Avg DTI | Avg Employment |
|---------|-------|-----------|-----------------|---------|----------------|
| Mass Market | ~1250 | ~$55,000 | ~660 | ~0.28 | ~4 years |
| Rising Prime | ~1250 | ~$72,000 | ~720 | ~0.22 | ~2 years |
| Established Prime | ~1250 | ~$105,000 | ~780 | ~0.18 | ~10 years |
| Subprime High-Risk | ~1250 | ~$38,000 | ~590 | ~0.45 | ~3 years |

**Model Performance (RandomForest Classifier):**
- Accuracy: ~96%
- Silhouette Score: ~0.58
- Features: income, credit_score, employment_years, debt_to_income, loan_history_count, age, home_ownership, verified_income

## Business Impact

- **Risk Pricing** — Segment-specific interest rate tiers
- **Auto-Decisioning** — Real-time segment classification for loan applications
- **Portfolio Monitoring** — Track segment drift over time
- **Loss Forecasting** — Higher risk segments inform reserve requirements

## Files

- `src/data_loader.py` — Generates 5000 synthetic customer records
- `src/features.py` — Feature engineering (RFM, behavioral, stability)
- `src/segment.py` — KMeans clustering with Elbow/Silhouette validation
- `src/classify.py` — RandomForest classifier on cluster labels
- `run_pipeline.py` — Full pipeline orchestration
- `reports/segmentation_results.json` — Output artifact
- `requirements.txt` — Python dependencies