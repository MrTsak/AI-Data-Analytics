"""Train and evaluate the two diabetes classifiers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.data import Dataset

RF_NAME, LR_NAME = "Random Forest", "Logistic Regression"
MODEL_NAMES = [RF_NAME, LR_NAME]
SEED = 42


@dataclass
class ModelResults:
    rf: RandomForestClassifier
    lr: Pipeline
    X_test: pd.DataFrame
    y_test: pd.Series
    metrics: dict
    best: str


def _roc_points(y, probs, n=101) -> dict:
    fpr, tpr, _ = roc_curve(y, probs)
    grid = np.linspace(0, 1, n)
    tpr_i = np.interp(grid, fpr, tpr)
    tpr_i[0] = 0.0
    return {"fpr": grid.round(4).tolist(), "tpr": tpr_i.round(4).tolist()}


def _evaluate(model, X, y) -> dict:
    pred = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "auc": float(roc_auc_score(y, probs)),
        "confusion": confusion_matrix(y, pred).tolist(),
        "roc": _roc_points(y, probs),
    }


def train_models(ds: Dataset) -> ModelResults:
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=0.2, random_state=SEED, stratify=ds.y)
    rf = RandomForestClassifier(n_estimators=150, max_depth=7, min_samples_split=8,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000,
                                   random_state=SEED)),
    ])
    lr.fit(X_train, y_train)
    metrics = {RF_NAME: _evaluate(rf, X_test, y_test), LR_NAME: _evaluate(lr, X_test, y_test)}
    best = max(MODEL_NAMES, key=lambda n: metrics[n]["auc"])
    return ModelResults(rf, lr, X_test, y_test, metrics, best)
