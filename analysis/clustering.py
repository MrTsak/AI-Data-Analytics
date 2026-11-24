"""K-means patient clustering with stable, data-driven labels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = ["age", "bmi", "HbA1c_level"]
RISK_LABELS = ["Lowest risk", "Middle risk", "Highest risk"]


@dataclass
class ClusterResult:
    labels: np.ndarray
    centers: pd.DataFrame
    sizes: list
    diabetes_rate: list
    descriptions: list


def cluster_patients(raw: pd.DataFrame, seed: int = 42, k: int = 3) -> ClusterResult:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(raw[CLUSTER_FEATURES])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled)
    centers = scaler.inverse_transform(km.cluster_centers_)
    # K-means label numbers are arbitrary; reorder by mean HbA1c (then age) so cluster 0
    # is always the lowest-HbA1c group.
    hba1c, age = CLUSTER_FEATURES.index("HbA1c_level"), CLUSTER_FEATURES.index("age")
    order = np.lexsort((centers[:, age], centers[:, hba1c]))
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    labels = remap[km.labels_]
    centers_df = pd.DataFrame(centers[order], columns=CLUSTER_FEATURES)

    y = raw["diabetes"].to_numpy()
    sizes = [int((labels == i).sum()) for i in range(k)]
    rates = [float(y[labels == i].mean()) for i in range(k)]
    rank = np.argsort(np.argsort(rates))
    descriptions = []
    for i in range(k):
        c = centers_df.iloc[i]
        tag = RISK_LABELS[0] if rank[i] == 0 else RISK_LABELS[-1] if rank[i] == k - 1 else RISK_LABELS[1]
        descriptions.append(
            f"{tag}: avg age {c['age']:.0f}, BMI {c['bmi']:.1f}, HbA1c {c['HbA1c_level']:.1f}% "
            f"— diabetes rate {rates[i] * 100:.1f}%")
    return ClusterResult(labels, centers_df, sizes, rates, descriptions)
