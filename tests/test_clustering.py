import numpy as np
import pytest

from analysis.clustering import cluster_patients
from analysis.data import load_dataset
from tests.conftest import DATA


@pytest.fixture(scope="module")
def raw():
    return load_dataset(DATA).raw


def test_cluster_order_stable(raw):
    a = cluster_patients(raw, seed=0)
    b = cluster_patients(raw, seed=42)
    assert np.allclose(a.centers.values, b.centers.values, atol=0.05)
    assert a.centers["HbA1c_level"].is_monotonic_increasing


def test_descriptions_match_data(raw):
    c = cluster_patients(raw)
    assert len(c.descriptions) == 3 == len(c.sizes) == len(c.diabetes_rate)
    assert sum(c.sizes) == len(raw)
    for i, text in enumerate(c.descriptions):
        assert f"{c.diabetes_rate[i] * 100:.1f}%" in text
    ranked = np.argsort(c.diabetes_rate)
    assert c.descriptions[ranked[-1]].startswith("Highest risk")
    assert c.descriptions[ranked[0]].startswith("Lowest risk")


def test_clustering_tolerates_missing_values(raw):
    messy = raw.copy()
    messy.loc[messy.index[::100], "bmi"] = np.nan
    c = cluster_patients(messy)
    assert sum(c.sizes) == len(messy)
