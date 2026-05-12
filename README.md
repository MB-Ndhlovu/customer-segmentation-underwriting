# Customer Segmentation for Underwriting

**Project 3 of the ML Portfolio — Sefako Makgatho Health Sciences University**

---

## Overview

This project builds a **customer segmentation system** for loan underwriting using unsupervised learning (KMeans clustering) followed by a supervised classifier for production deployment.

A lender can use 4 distinct segments to:
- Price risk appropriately
- Set credit limits
- Trigger additional verification
- Approve/reject applications with explainable reasoning

---

## Architecture

```
run_pipeline.py
    src/data_loader.py   → generate 5000 synthetic customer records
    src/features.py     → RFM + behavioral + stability features
    src/segment.py       → KMeans clustering, silhouette analysis, elbow method
    src/classify.py      → RandomForestClassifier trained on cluster labels
reports/
    segmentation_results.json
```

---

## Segments

| Label | Name | Profile |
|-------|------|---------|
| 0 | Mass Market | Low income, decent credit, short history |
| 1 | Rising Prime | Growing income, good credit, stable employment |
| 2 | Established Prime | High income, excellent credit, long tenure |
| 3 | Subprime High-Risk | Low credit, high DTI, verified income issues |

---

## Results Summary

- **Silhouette Score**: 0.XX (cluster quality)
- **4 Segments** produced by KMeans (k=4)
- **RandomForest Classifier** trained on segment labels — accuracy: ~XX%
- Segment profiles exported to `reports/segmentation_results.json`

---

## Business Impact

- Replace gut-feel underwriting with data-driven segmentation
- Each segment maps to a risk tier with distinct handling rules
- Classifier enables real-time segment prediction from application data alone

---

## Tech Stack

- Python 3.12
- pandas, numpy
- scikit-learn (KMeans, RandomForest, silhouette_score, StandardScaler)
- faker (synthetic data generation)