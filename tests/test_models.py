import pytest

from analysis.data import load_dataset
from analysis.models import MODEL_NAMES, train_models
from tests.conftest import DATA


@pytest.fixture(scope="module")
def results():
    return train_models(load_dataset(DATA))


def test_metrics_sane(results):
    for name in MODEL_NAMES:
        m = results.metrics[name]
        assert m["auc"] > 0.9
        for k in ("accuracy", "precision", "recall", "f1", "auc"):
            assert 0 <= m[k] <= 1
        assert sum(map(sum, m["confusion"])) == len(results.y_test)


def test_best_is_max_auc(results):
    best = max(MODEL_NAMES, key=lambda n: results.metrics[n]["auc"])
    assert results.best == best


def test_roc_downsampled(results):
    for name in MODEL_NAMES:
        roc = results.metrics[name]["roc"]
        assert len(roc["fpr"]) == len(roc["tpr"]) <= 101
        assert roc["fpr"][0] == 0 and roc["tpr"][0] == 0
        assert roc["fpr"][-1] == 1 and roc["tpr"][-1] == 1
