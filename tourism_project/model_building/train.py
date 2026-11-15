"""Fit models, tune average precision, log experiments and export the winner.

MLflow uses a local file store in Actions, archived as a workflow artifact. For an
offline notebook runtime without MLflow, a complete JSON experiment ledger is
still written; CI installs MLflow and performs full tracking.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score,
                             balanced_accuracy_score, confusion_matrix, brier_score_loss)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .config import (OUTPUT_DIR, MODEL_PATH, METADATA_PATH, TARGET, SEED,
                     FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS)


def top_fraction_metrics(y_true, probabilities, fraction=0.20):
    """Assess how effectively a fixed-size contact list captures past buyers."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    size = max(1, int(np.ceil(len(y_true) * fraction)))
    chosen = np.argsort(-probabilities, kind="stable")[:size]
    selected_positive = int(y_true[chosen].sum())
    precision = selected_positive / size
    baseline = float(y_true.mean())
    return {
        "top20_precision": float(precision),
        "top20_recall": float(selected_positive / max(1, int(y_true.sum()))),
        "top20_lift": float(precision / baseline) if baseline > 0 else 0.0,
        "top20_contact_count": int(size), "top20_actual_buyers": selected_positive
    }


def build_pipeline(estimator):
    transformer = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
            ("cat", Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), CATEGORICAL_COLUMNS)
        ], remainder="drop", verbose_feature_names_out=False
    )
    return Pipeline(steps=[("preprocessor", transformer), ("classifier", estimator)])


def candidate_grids():
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, class_weight="balanced_subsample", n_jobs=1),
            {"classifier__n_estimators": [150, 250],
             "classifier__max_depth": [8, None],
             "classifier__min_samples_leaf": [2, 5]}
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"classifier__n_estimators": [100, 180],
             "classifier__learning_rate": [0.05, 0.10],
             "classifier__max_depth": [2]}
        )
    }


def train_model(split_dir=OUTPUT_DIR, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    X_train = pd.read_csv(split_dir / "Xtrain.csv")
    X_test = pd.read_csv(split_dir / "Xtest.csv")
    y_train = pd.read_csv(split_dir / "ytrain.csv")[TARGET].astype(int)
    y_test = pd.read_csv(split_dir / "ytest.csv")[TARGET].astype(int)
    # Data contracts: prohibit accidentally training with retrospective inputs.
    if X_train.columns.tolist() != FEATURE_COLUMNS or X_test.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature order or pre-contact schema does not match config.py")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    leaderboard = []
    all_trials = []
    winners = {}

    # Soft availability check is solely for offline execution without network.
    try:
        import mlflow
        mlflow.set_tracking_uri((split_dir / "mlruns").resolve().as_uri())
        mlflow.set_experiment("Tourism-PreContact-Propensity")
        tracking_enabled = True
    except ImportError:
        mlflow = None
        tracking_enabled = False
        print("Local MLflow unavailable; full trial log saved to JSON. GitHub Actions installs MLflow.")

    for name, (estimator, grid_params) in candidate_grids().items():
        search = GridSearchCV(
            build_pipeline(estimator), grid_params, cv=cv, scoring="average_precision",
            n_jobs=1, refit=True, verbose=0, return_train_score=False
        )
        search.fit(X_train, y_train)
        trials = []
        for index, params in enumerate(search.cv_results_["params"]):
            trial = {
                "model": name, "trial": index + 1,
                "params": {key: (None if val is None else val) for key, val in params.items()},
                "mean_cv_average_precision": float(search.cv_results_["mean_test_score"][index]),
                "std_cv_average_precision": float(search.cv_results_["std_test_score"][index])
            }
            trials.append(trial)
            all_trials.append(trial)
            if tracking_enabled:
                with mlflow.start_run(run_name=f"{name}-trial-{index + 1}"):
                    mlflow.log_params({k: "None" if v is None else v for k, v in params.items()})
                    mlflow.log_metric("cv_average_precision", trial["mean_cv_average_precision"])
                    mlflow.log_metric("cv_average_precision_std", trial["std_cv_average_precision"])
        leaderboard.append({"model": name,
                            "cv_average_precision": float(search.best_score_),
                            "best_params": search.best_params_,
                            "trials": len(trials)})
        winners[name] = search.best_estimator_
        print(f"{name}: best CV average precision={search.best_score_:.4f}; "
              f"best params={search.best_params_}")

    # Holdout is deliberately NOT used to select model family or hyperparameters.
    chosen = max(leaderboard, key=lambda row: row["cv_average_precision"])
    final_model = winners[chosen["model"]]
    prob = final_model.predict_proba(X_test)[:, 1]
    label = (prob >= 0.50).astype(int)
    metrics = {
        "average_precision": float(average_precision_score(y_test, prob)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "precision_at_0_5": float(precision_score(y_test, label, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_test, label, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_test, label, zero_division=0)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_test, label)),
        "brier_score": float(brier_score_loss(y_test, prob)),
        **top_fraction_metrics(y_test, prob)
    }
    # Data summary is for audit and does not imply wellness-specific target validation.
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on": "historical generic travel-package purchase outcomes",
        "not_yet_validated_on": "new Wellness Tourism Package",
        "scoring_point": "before sales contact",
        "model_family": chosen["model"],
        "selection_criterion": "3-fold stratified CV average precision on training data",
        "cv_average_precision": chosen["cv_average_precision"],
        "best_params": chosen["best_params"],
        "feature_names": FEATURE_COLUMNS,
        "decision_threshold": 0.5,
        "test_rows": int(len(y_test)),
        "holdout_positive_rate": float(y_test.mean()),
        "test_metrics": metrics,
        "confusion_matrix_0_5": confusion_matrix(y_test, label).tolist(),
        "mlflow_tracking_enabled": tracking_enabled,
        "candidate_models": leaderboard
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path, compress=3)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (split_dir / "experiment_trials.json").write_text(json.dumps(all_trials, indent=2), encoding="utf-8")
    (split_dir / "holdout_predictions.csv").write_text(
        pd.DataFrame({"actual": y_test, "pred_probability": prob,
                      "predicted_at_0_5": label}).to_csv(index=False), encoding="utf-8")
    if tracking_enabled:
        with mlflow.start_run(run_name=f"CHAMPION-{chosen['model']}"):
            mlflow.log_param("model_family", chosen["model"])
            mlflow.log_params({k: "None" if v is None else v for k,v in chosen["best_params"].items()})
            mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v, (float,int))})
            mlflow.log_metric("best_cv_average_precision", chosen["cv_average_precision"])
            mlflow.log_artifact(str(model_path), artifact_path="model")
            mlflow.log_artifact(str(metadata_path), artifact_path="evaluation")
            mlflow.log_artifact(str(split_dir / "experiment_trials.json"), artifact_path="evaluation")
    print(json.dumps({"selected_model": chosen["model"], "cv_ap": chosen["cv_average_precision"],
                      "holdout_metrics": metrics}, indent=2))
    return final_model, metadata

if __name__ == "__main__":
    train_model()
