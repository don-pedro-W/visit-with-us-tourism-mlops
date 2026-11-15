# Visit with Us — pre-contact tourism-purchase propensity MLOps

**Rubric-aligned implementation:** GitHub Actions dataset registration → train/test preprocessing → 3-fold CV hyperparameter tuning / MLflow tracking → held-out evaluation → committed model → Streamlit Community Cloud.

## Quickstart

Requires **Python 3.11**. `tourism.csv` is intentionally stored in `tourism_project/data/`.

```bash
python -m pip install -r tourism_project/requirements.txt
python -m tourism_project.model_building.data_register
python -m tourism_project.model_building.prep
python -m tourism_project.model_building.train
python -m pytest -q tests
```

MLflow runs are stored at `tourism_project/outputs/mlruns/` and are downloadable via Actions' **model-and-evaluation** artifact. No ngrok or public MLflow server is required. In strictly offline environments without MLflow, the training script writes full parameter/metric trial results to `tourism_project/outputs/experiment_trials.json` but does not claim that MLflow ran.

## Deploy

1. Push this directory's contents to a GitHub repository with a `main` branch.
2. Open **Actions → Tourism propensity → Run workflow** (or push code to `main`). Ensure repository Actions settings permit the `GITHUB_TOKEN` to write repository contents; branch protection may require using a PR or approved workflow instead.
3. Confirm that **register-dataset**, **data-prep**, **model-training**, and **publish-model** have succeeded and `tourism_project/deployment/best_model.joblib` + `model_metadata.json` are committed.
4. Sign in at <https://share.streamlit.io>, create an app from the repository's `main` branch, **main file path:** `tourism_project/deployment/app.py`, Python version **3.11**. The `deployment/requirements.txt` is detected adjacent to `app.py`.
5. Capture real GitHub repository tree, Actions successful-run screenshots, Streamlit URL and app screenshot and append those to the notebook's **Output Evaluation** section before exporting HTML.

**Modeling caveat:** `ProdTaken` is a *historical travel-product purchase outcome*, not a wellness-specific label. Exclude identifiers and retrospective pitch/follow-up fields from pre-contact scoring, and conduct prospective wellness-package validation before relying on scores for campaign allocation. This is a risk-ranking prototype, not a validated wellness conversion model. The conversion observations are associational, not causal.

## Workflow graph

```text
Repo tourism.csv  -> Register + validate -> registered-data artifact
                  -> Prepare + stratified split -> prepared-splits artifact
                  -> 3-fold CV RF / GB + MLflow + held-out metrics
                  -> model-and-evaluation artifact
                  -> publish model + metadata on main
                  -> Streamlit Community Cloud (deploy from GitHub)
```
