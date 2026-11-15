"""Deterministic preprocessing and stratified holdout; no holdout-driven imputation."""
from __future__ import annotations
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import (DATA_PATH, OUTPUT_DIR, TARGET, FEATURE_COLUMNS,
                     CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, SEED)
from .data_register import validate_dataset


def clean_data(df):
    """Apply canonical labels and remove identifiers and post-contact features.

    Numeric missingness is retained here and imputed using *training folds only*
    within the modeling pipeline, avoiding information leakage.
    """
    df = df.copy()
    df["Gender"] = (df["Gender"].astype("string").str.strip()
                    .replace({"Fe Male": "Female"}))
    for column in (set(CATEGORICAL_COLUMNS) - {"CityTier"}):
        df[column] = df[column].astype("string").str.strip().replace({"": pd.NA})
        # Object strings are supported reliably by OneHotEncoder across environments.
        df[column] = df[column].astype(object).where(df[column].notna(), None)
    for column in NUMERIC_COLUMNS + ["CityTier"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    return df[FEATURE_COLUMNS + [TARGET]]


def prepare_data(csv_path=DATA_PATH, output_dir=OUTPUT_DIR, test_size=0.20):
    validate_dataset(csv_path)
    raw = pd.read_csv(csv_path)
    cleaned = clean_data(raw)
    X = cleaned[FEATURE_COLUMNS]
    y = cleaned[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, df in [("Xtrain.csv", X_train), ("Xtest.csv", X_test),
                         ("ytrain.csv", y_train.to_frame()), ("ytest.csv", y_test.to_frame())]:
        df.reset_index(drop=True).to_csv(output_dir / filename, index=False)
    report = {
        "training_rows": len(X_train), "holdout_rows": len(X_test),
        "training_purchase_rate": round(float(y_train.mean()), 6),
        "holdout_purchase_rate": round(float(y_test.mean()), 6),
        "available_at_scoring": FEATURE_COLUMNS,
        "excluded_post_contact_fields": ["TypeofContact", "DurationOfPitch",
                                         "NumberOfFollowups", "ProductPitched", "PitchSatisfactionScore"],
        "gender_typo_normalized": int(raw["Gender"].eq("Fe Male").sum())
    }
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_data()
