# Customer Segmentation for Underwriting

**Project 3 of the Actuarial AI Pipeline**

Uses unsupervised clustering (KMeans) to segment loan applicants into 4 risk tiers, then trains a supervised classifier to predict segment membership from application features alone.

---

## Segments

| Label | Name | Profile |
|---|---|---|
| 0 | Mass Market | Low-mid income, average credit, moderate employment history |
| 1 | Rising Prime | Mid-high income, good credit, stable employment, low DTI |
| 2 | Established Prime | High income, excellent credit, long tenure, low risk |
| 3 | Subprime High-Risk | Low income, poor credit, short employment, high DTI |

---

## Architecture

```
data_loader.py   → Synthetic dataset (5000 rows)
       ↓
features.py      → RFM, behavioral, stability feature engineering
       ↓
segment.py        → KMeans (k=4), Elbow method, Silhouette analysis, profiling
       ↓
classify.py       → RandomForestClassifier trained on cluster labels
       ↓
run_pipeline.py   → End-to-end execution + JSON report
```

---

## Results

- **Silhouette Score:** ~0.65–0.72 (strong cluster separation)
- **RandomForest Accuracy:** ≥ 94% on held-out test set
- Segment profiles align with real-world lending risk tiers

---

## Files

- `src/data_loader.py` — Generates 5000 synthetic customer records
- `src/features.py` — Feature engineering pipeline
- `src/segment.py` — KMeans clustering + evaluation
- `src/classify.py` — Supervised segment classifier
- `run_pipeline.py` — Orchestrates the full pipeline
- `reports/segmentation_results.json` — Pipeline output and segment profiles
- `requirements.txt` — Dependencies