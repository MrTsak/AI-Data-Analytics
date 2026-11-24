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


def test_load_keeps_continuous_gaps_for_split_time_filling(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text(
        "gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes\n"
        "Female,50,0,0,never,,6.0,140,1\n"
        "Male,30,1,0,current,25.0,5.0,100,0\n"
    )
    ds = load_dataset(csv)
    # Continuous gaps are filled later from training rows only (see train_models).
    assert ds.X["bmi"].isna().sum() == 1
    assert ds.X.drop(columns=["bmi"]).notna().all().all()


def test_excludes_missing_or_unknown_gender(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text(
        "gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes\n"
        ",50,0,0,never,30,6.0,140,1\n"
        "Unknown,30,1,0,current,25.0,5.0,100,0\n"
        "Male,60,1,0,former,28.0,6.5,180,1\n"
    )
    ds = load_dataset(csv)
    assert len(ds.raw) == 1 and ds.n_other_gender == 2


def test_missing_target_dropped_and_binary_stays_binary(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text(
        "gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level,diabetes\n"
        "Female,50,0,0,never,30,6.0,140,\n"
        "Male,30,,0,current,25.0,5.0,100,0\n"
        "Male,60,1,0,former,28.0,6.5,180,1\n"
        "Female,40,0,1,never,22.0,5.5,120,0\n"
    )
    ds = load_dataset(csv)
    assert len(ds.raw) == 3
    assert set(ds.raw["diabetes"]) <= {0, 1}
    assert set(ds.X["hypertension"]) <= {0.0, 1.0}
