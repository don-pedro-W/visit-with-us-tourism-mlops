"""Register the repository-held raw CSV with schema, uniqueness and target checks."""
from __future__ import annotations
import argparse
import json
import pandas as pd
from .config import DATA_PATH, OUTPUT_DIR, REQUIRED_COLUMNS, TARGET


def validate_dataset(csv_path=DATA_PATH, report_path=None):
    source = pd.read_csv(csv_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(source.columns))
    if missing_columns:
        raise ValueError(f"Required columns missing from dataset: {missing_columns}")
    if source.empty:
        raise ValueError("Dataset is empty")
    if source["CustomerID"].isna().any() or source["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be non-null and unique")
    if source[TARGET].isna().any() or not set(source[TARGET].unique()).issubset({0, 1}):
        raise ValueError("ProdTaken must contain only non-null binary values 0 and 1")
    if source[TARGET].nunique() != 2:
        raise ValueError("Both outcome classes are required for model evaluation")
    report = {
        "dataset": str(csv_path), "rows": len(source), "columns": len(source.columns),
        "unique_customers": int(source.CustomerID.nunique()),
        "duplicate_rows": int(source.duplicated().sum()),
        "missing_cells": int(source.isna().sum().sum()),
        "missing_by_column": {k:int(v) for k,v in source.isna().sum().items() if v},
        "outcomes": {str(k):int(v) for k,v in source[TARGET].value_counts().sort_index().items()},
        "historical_purchase_rate": round(float(source[TARGET].mean()), 6)
    }
    report_path = OUTPUT_DIR / "registration_report.json" if report_path is None else report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate tourism.csv against expected data dictionary")
    parser.add_argument("--csv", type=str, default=str(DATA_PATH))
    args = parser.parse_args()
    validate_dataset(args.csv)
