"""Lightweight CI checks; train script has already produced the model in Actions."""
import json
from pathlib import Path
import joblib
import pandas as pd
from tourism_project.model_building.config import (DATA_PATH, FEATURE_COLUMNS,
    MODEL_PATH, METADATA_PATH)
from tourism_project.model_building.data_register import validate_dataset
from tourism_project.model_building.prep import clean_data
from tourism_project.model_building.train import top_fraction_metrics


def test_registration_schema():
    report=validate_dataset(DATA_PATH)
    assert report["rows"] == report["unique_customers"]
    assert 0 < report["historical_purchase_rate"] < 1


def test_pre_contact_fields_and_canonical_gender():
    cleaned=clean_data(pd.read_csv(DATA_PATH))
    assert cleaned.drop(columns=["ProdTaken"]).columns.tolist() == FEATURE_COLUMNS
    assert "Fe Male" not in set(cleaned["Gender"])
    assert "DurationOfPitch" not in cleaned


def test_exported_model_scores():
    assert MODEL_PATH.exists() and METADATA_PATH.exists()
    model=joblib.load(MODEL_PATH)
    metadata=json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    X=clean_data(pd.read_csv(DATA_PATH))[FEATURE_COLUMNS].head(4)
    probabilities=model.predict_proba(X)[:,1]
    assert len(probabilities)==4
    assert all(0 <= p <=1 for p in probabilities)
    assert metadata["feature_names"]==FEATURE_COLUMNS


def test_targeting_lift_math():
    m=top_fraction_metrics([1,0,0,0,0], [0.9,0.4,0.3,0.2,0.1], 0.2)
    assert m["top20_precision"] == 1.0
    assert m["top20_lift"] == 5.0
