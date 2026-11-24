"""Run the analysis pipeline and export everything the browser dashboard needs to web/data/.

Usage:  python scripts/export_web.py [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.clustering import cluster_patients  # noqa: E402
from analysis.data import FEATURES, NUMERIC, SMOKING_LABELS, Dataset, load_dataset  # noqa: E402
from analysis.models import ModelResults, train_models  # noqa: E402

FILES = ["summary.json", "explore.json", "clusters.json", "models.json", "model_lr.json",
         "model_rf.json", "form.json", "parity.json"]
CONTINUOUS = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
BINS = 30
SCATTER = 3000
SEED = 42


def _r(x, nd=4):
    return [round(float(v), nd) for v in x]


def export_summary(ds: Dataset, res: ModelResults) -> dict:
    return {
        "n_original": ds.n_original, "n_duplicates": ds.n_duplicates,
        "n_other_gender": ds.n_other_gender, "n_rows": len(ds.raw),
        "positive_rate": round(float(ds.y.mean()), 4),
        "best": res.best, "best_auc": round(res.metrics[res.best]["auc"], 4),
    }


def export_explore(ds: Dataset) -> dict:
    df = ds.raw
    pos = df["diabetes"] == 1
    hists = {}
    for col in CONTINUOUS:
        edges = np.linspace(df[col].min(), df[col].max(), BINS + 1)
        hists[col] = {
            "edges": _r(edges, 3),
            "neg": np.histogram(df.loc[~pos, col], edges)[0].tolist(),
            "pos": np.histogram(df.loc[pos, col], edges)[0].tolist(),
        }
    cats = {}
    for col, labels in [("gender", None), ("smoking_history", SMOKING_LABELS),
                        ("hypertension", {0: "No", 1: "Yes"}), ("heart_disease", {0: "No", 1: "Yes"})]:
        values = sorted(df[col].unique(), key=lambda v: list(labels).index(v) if labels else v)
        cats[col] = {
            "labels": [labels[v] if labels else str(v) for v in values],
            "neg": [int(((df[col] == v) & ~pos).sum()) for v in values],
            "pos": [int(((df[col] == v) & pos).sum()) for v in values],
        }
    cols = NUMERIC + ["diabetes"]
    corr = df[cols].corr()
    return {"histograms": hists, "categories": cats,
            "correlation": {"labels": cols, "matrix": [_r(row, 3) for row in corr.to_numpy()]}}


def export_clusters(ds: Dataset) -> dict:
    c = cluster_patients(ds.raw, seed=SEED)
    df = ds.raw.assign(cluster=c.labels)
    sample = df.sample(n=min(SCATTER, len(df)), random_state=SEED)
    return {
        "centers": [{k: round(float(v), 2) for k, v in row.items()} for _, row in c.centers.iterrows()],
        "sizes": c.sizes, "diabetes_rate": _r(c.diabetes_rate),
        "descriptions": c.descriptions,
        "sample": [{"HbA1c_level": float(r.HbA1c_level), "bmi": float(r.bmi), "age": float(r.age),
                    "cluster": int(r.cluster), "diabetes": int(r.diabetes)}
                   for r in sample.itertuples()],
    }


def export_models(res: ModelResults) -> dict:
    clf = res.lr.named_steps["clf"]
    metrics = {n: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
               for n, m in res.metrics.items()}
    return {
        "metrics": metrics, "best": res.best,
        "importances": dict(zip(FEATURES, _r(res.rf.feature_importances_))),
        "lr_coefs": dict(zip(FEATURES, _r(clf.coef_[0]))),
    }


def export_lr(res: ModelResults) -> dict:
    scale, clf = res.lr.named_steps["scale"], res.lr.named_steps["clf"]
    return {"features": FEATURES, "mean": scale.mean_.tolist(), "scale": scale.scale_.tolist(),
            "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}


def export_rf(res: ModelResults) -> dict:
    trees = []
    for est in res.rf.estimators_:
        t = est.tree_
        value = t.value[:, 0, :]
        prob = value[:, 1] / value.sum(axis=1)
        leaf = t.children_left == -1
        trees.append({
            "f": np.where(leaf, -1, t.feature).tolist(),
            # Thresholds kept at full precision: sklearn compares float32(x) <= threshold.
            "t": np.where(leaf, 0, t.threshold).tolist(),
            "l": t.children_left.tolist(),
            "r": t.children_right.tolist(),
            "v": np.where(leaf, np.round(prob, 8), 0).tolist(),
        })
    return {"features": FEATURES, "trees": trees}


def rf_predict_from_json(model: dict, x) -> float:
    """Reference walker mirroring web/js/models.js predictRF."""
    # sklearn casts X to float32, then compares against the float64 threshold.
    x32 = np.asarray(x, dtype=np.float32).astype(np.float64).tolist()
    total = 0.0
    for tr in model["trees"]:
        n = 0
        while tr["f"][n] != -1:
            n = tr["l"][n] if x32[tr["f"][n]] <= tr["t"][n] else tr["r"][n]
        total += tr["v"][n]
    return total / len(model["trees"])


def export_form(ds: Dataset) -> dict:
    numeric = {}
    steps = {"age": 1, "bmi": 0.1, "HbA1c_level": 0.1, "blood_glucose_level": 1}
    for col in CONTINUOUS:
        s = ds.raw[col]
        numeric[col] = {"min": float(s.min()), "max": float(s.max()),
                        "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
                        "mean": round(float(s.mean()), 1), "step": steps[col]}
    return {"numeric": numeric,
            "smoking": [{"value": v, "label": label} for v, label in SMOKING_LABELS.items()]}


def export_parity(ds: Dataset, res: ModelResults) -> list:
    rng = np.random.default_rng(SEED)
    X = res.X_test.sample(n=190, random_state=SEED).copy()
    lo, hi = ds.X.min(), ds.X.max()
    extremes = [lo, hi, lo.where(np.arange(len(lo)) % 2 == 0, hi),
                hi.where(np.arange(len(hi)) % 2 == 0, lo), (lo + hi) / 2]
    X = pd.concat([X, pd.DataFrame(extremes)], ignore_index=True)
    # Five cases with a feature exactly on an RF split threshold (tests the <= branch).
    for i in range(5):
        tree = res.rf.estimators_[i].tree_
        row = res.X_test.iloc[int(rng.integers(len(res.X_test)))].copy()
        row.iloc[tree.feature[0]] = float(np.float32(tree.threshold[0]))
        X.loc[len(X)] = row
    X = X[FEATURES].astype(float)
    lr_p = res.lr.predict_proba(X)[:, 1]
    rf_p = res.rf.predict_proba(X)[:, 1]
    return [{"input": dict(zip(FEATURES, row)), "lr": float(a), "rf": float(b)}
            for row, a, b in zip(X.to_numpy().tolist(), lr_p, rf_p)]


def main(out_dir: str | Path = ROOT / "web" / "data") -> ModelResults:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset()
    res = train_models(ds)
    payloads = {
        "summary.json": export_summary(ds, res),
        "explore.json": export_explore(ds),
        "clusters.json": export_clusters(ds),
        "models.json": export_models(res),
        "model_lr.json": export_lr(res),
        "model_rf.json": export_rf(res),
        "form.json": export_form(ds),
        "parity.json": export_parity(ds, res),
    }
    for name, data in payloads.items():
        (out / name).write_text(json.dumps(data, separators=(",", ":")))
    return res


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else ROOT / "web" / "data"
    main(target)
    print(f"Wrote {len(FILES)} files to {target}")
