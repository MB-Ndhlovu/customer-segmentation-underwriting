"""
Customer Segmentation for Underwriting
=======================================
Identifies 4 distinct customer segments for lending risk assessment:
- Mass Market       (Segment 0): Low risk, broad access
- Rising Prime      (Segment 1): Moderate risk, growth profile
- Established Prime (Segment 2): Low risk, stable borrowers
- Subprime High-Risk (Segment 3): High risk, requires caution

Pipeline:
1. Load synthetic customer data (5000 rows)
2. Engineer RFM + behavioral + stability features
3. KMeans clustering with elbow + silhouette validation
4. Train RandomForest classifier on cluster labels
5. Profile each segment for underwriting decisions
"""

__version__ = "1.0.0"