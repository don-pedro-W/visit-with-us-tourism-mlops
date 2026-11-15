"""Shared, leakage-safe schema for prospective customer scoring."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DATA_PATH = PROJECT_ROOT / "data" / "tourism.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = PROJECT_ROOT / "deployment" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "deployment" / "model_metadata.json"
TARGET = "ProdTaken"
ID_COLUMNS = ["CustomerID", "Unnamed: 0"]
# These fields are generated during/after the sales interaction, not at initial lead scoring.
POST_CONTACT_COLUMNS = ["TypeofContact", "DurationOfPitch", "NumberOfFollowups",
                        "ProductPitched", "PitchSatisfactionScore"]
CATEGORICAL_COLUMNS = ["Occupation", "Gender", "MaritalStatus", "Designation", "CityTier"]
NUMERIC_COLUMNS = ["Age", "NumberOfPersonVisiting", "PreferredPropertyStar",
                   "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
                   "MonthlyIncome"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
REQUIRED_COLUMNS = ["CustomerID", TARGET] + FEATURE_COLUMNS + POST_CONTACT_COLUMNS
SEED = 42
