"""Load and clean the diabetes dataset into model-ready features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "sample.csv"

NUMERIC = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
BINARY = ["hypertension", "heart_disease"]
CONTINUOUS = [c for c in NUMERIC if c not in BINARY]

# Raw smoking value -> one-hot column. "No Info" is the baseline (all zeros).
SMOKING_LEVELS = {
    "never": "smoking_never",
    "former": "smoking_former",
    "current": "smoking_current",
    "not current": "smoking_not_current",
    "ever": "smoking_ever",
}
SMOKING_LABELS = {
    "No Info": "No info",
    "never": "Never smoked",
    "former": "Former smoker",
    "current": "Current smoker",
    "not current": "Not currently smoking",
    "ever": "Has smoked",
}

FEATURES = ["gender_male", *NUMERIC, *SMOKING_LEVELS.values()]


@dataclass
class Dataset:
    raw: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    n_original: int
    n_duplicates: int
    n_other_gender: int


def _encode(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["gender_male"] = (df["gender"] == "Male").astype(int)
    for col in NUMERIC:
        out[col] = df[col].astype(float)
    for raw_value, col in SMOKING_LEVELS.items():
        out[col] = (df["smoking_history"] == raw_value).astype(int)
    return out[FEATURES]


def load_dataset(path: str | Path = DEFAULT_PATH) -> Dataset:
    df = pd.read_csv(path)
    n_original = len(df)
    n_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()
    other = df["gender"] == "Other"
    n_other = int(other.sum())
    df = df[~other].copy()
    # Rows without an outcome can't be used for training or evaluation.
    df = df.dropna(subset=["diabetes"])
    df["diabetes"] = df["diabetes"].astype(int)
    for col in BINARY:
        df[col] = df[col].fillna(df[col].mode().iloc[0]).astype(int)
    df[CONTINUOUS] = df[CONTINUOUS].fillna(df[CONTINUOUS].mean())
    df["smoking_history"] = df["smoking_history"].where(df["smoking_history"].isin(SMOKING_LABELS), "No Info")
    df = df.reset_index(drop=True)
    return Dataset(
        raw=df,
        X=_encode(df),
        y=df["diabetes"].astype(int),
        n_original=n_original,
        n_duplicates=n_duplicates,
        n_other_gender=n_other,
    )


def encode_patient(patient: dict) -> pd.DataFrame:
    """Encode one patient given raw values (gender 'Female'/'Male', raw smoking value)."""
    return _encode(pd.DataFrame([patient]))
