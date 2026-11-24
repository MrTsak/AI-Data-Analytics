import json

import numpy as np
import pytest

from scripts.export_web import FILES, main, rf_predict_from_json


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = tmp_path_factory.mktemp("web_data")
    ctx = main(d)
    return d, ctx


def _load(d, name):
    return json.loads((d / name).read_text())


def test_export_schema(out):
    d, _ = out
    assert sorted(p.name for p in d.iterdir()) == sorted(FILES)
    assert {"n_original", "n_duplicates", "n_other_gender", "n_rows", "positive_rate",
            "best", "best_auc"} <= _load(d, "summary.json").keys()
    ex = _load(d, "explore.json")
    assert set(ex["histograms"]) == {"age", "bmi", "HbA1c_level", "blood_glucose_level"}
    h = ex["histograms"]["bmi"]
    assert len(h["edges"]) == 31 and len(h["neg"]) == len(h["pos"]) == 30
    assert set(ex["categories"]) == {"gender", "smoking_history", "hypertension", "heart_disease"}
    assert len(ex["correlation"]["matrix"]) == len(ex["correlation"]["labels"])
    cl = _load(d, "clusters.json")
    assert len(cl["centers"]) == 3
    assert cl["grid"] and all(g["n"] >= 5 for g in cl["grid"])
    assert "sample" not in cl and sum(g["n"] for g in cl["grid"]) <= 96128
    assert {"metrics", "best", "importances", "lr_coefs"} <= _load(d, "models.json").keys()
    assert {"features", "mean", "scale", "coef", "intercept"} <= _load(d, "model_lr.json").keys()
    form = _load(d, "form.json")
    assert set(form["numeric"]) == {"age", "bmi", "HbA1c_level", "blood_glucose_level"}
    assert len(_load(d, "parity.json")) == 198


def test_rf_size_under_2mb(out):
    d, _ = out
    assert (d / "model_rf.json").stat().st_size < 2_000_000


def test_rf_json_matches_sklearn(out):
    d, ctx = out
    model = _load(d, "model_rf.json")
    X = ctx.X_test.iloc[:500]
    expected = ctx.rf.predict_proba(X)[:, 1]
    got = np.array([rf_predict_from_json(model, row) for row in X.to_numpy()])
    assert np.allclose(got, expected, atol=1e-6)


def test_export_handles_messy_replacement_dataset(tmp_path):
    import pandas as pd
    from tests.conftest import DATA

    df = pd.read_csv(DATA).iloc[:3000].copy()
    df.loc[df.index[:20], "gender"] = None
    df.loc[df.index[20:30], "smoking_history"] = "sometimes"
    df.loc[df.index[30:40], "hypertension"] = None
    df.loc[df.index[40:50], "bmi"] = None
    csv = tmp_path / "messy.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "out"
    main(out, data_path=csv)
    ex = _load(out, "explore.json")
    assert ex["categories"]["gender"]["labels"] == ["Female", "Male"]
    assert ex["categories"]["hypertension"]["labels"] == ["No", "Yes"]
    assert "sometimes" not in json.dumps(ex)
