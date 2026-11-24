import pandas as pd

from analysis.data import FEATURES, encode_patient, load_dataset
from tests.conftest import DATA

SMOKING_COLS = [f for f in FEATURES if f.startswith("smoking_")]


def test_drops_duplicates():
    ds = load_dataset(DATA)
    assert ds.n_original == 100_000
    assert ds.n_duplicates == 3854
    assert len(ds.raw) == 100_000 - 3854 - ds.n_other_gender
    assert not ds.raw.duplicated().any()


def test_excludes_other_gender():
    ds = load_dataset(DATA)
    assert ds.n_other_gender > 0
    assert set(ds.raw["gender"]) == {"Female", "Male"}


def test_features_one_hot():
    ds = load_dataset(DATA)
    assert list(ds.X.columns) == FEATURES
    assert (ds.X[SMOKING_COLS].sum(axis=1) <= 1).all()
    assert len(ds.X) == len(ds.y) == len(ds.raw)


def test_encode_patient_matches_X():
    ds = load_dataset(DATA)
    for i in [0, 5, 123, 4567]:
        row = ds.raw.iloc[i].to_dict()
        enc = encode_patient(row)
        assert list(enc.columns) == FEATURES
        pd.testing.assert_series_equal(
            enc.iloc[0].astype(float), ds.X.iloc[i].astype(float), check_names=False
        )


def test_load_handles_missing_values(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text(
        "gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes\n"
        "Female,50,0,0,never,,6.0,140,1\n"
        "Male,30,1,0,current,25.0,5.0,100,0\n"
    )
    ds = load_dataset(csv)
    assert not ds.X.isna().any().any()
    assert ds.X["bmi"].iloc[0] == 25.0
