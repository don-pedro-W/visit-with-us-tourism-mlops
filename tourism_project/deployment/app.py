"""Streamlit Community Cloud app: prospective historical-purchase propensity."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

DEPLOY_DIR = Path(__file__).resolve().parent
MODEL_FILE = DEPLOY_DIR / "best_model.joblib"
METADATA_FILE = DEPLOY_DIR / "model_metadata.json"
st.set_page_config(page_title="Visit with Us | Lead Prioritisation", page_icon="🌿", layout="wide")

@st.cache_resource
def get_model():
    return joblib.load(MODEL_FILE)

st.title("🌿 Visit with Us · Lead Prioritisation")
st.write("Predict **historical travel-package purchase propensity** using information available *before* sales contact.")
st.info("Research prototype: historical package outcomes are not verified Wellness Tourism Package outcomes. Validate on wellness-specific labelled customers before commercial rollout.")
if not MODEL_FILE.exists() or not METADATA_FILE.exists():
    st.error("Trained model not found. Run the GitHub Actions pipeline and commit best_model.joblib before deployment.")
    st.stop()
model = get_model()
metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

with st.form("customer"):
    st.subheader("Prospective customer profile")
    c1,c2,c3 = st.columns(3)
    with c1:
        age=st.number_input("Age", min_value=18, max_value=100, value=37)
        city=st.selectbox("City tier", [1,2,3])
        occupation=st.selectbox("Occupation", ["Salaried","Small Business","Large Business","Free Lancer"])
        gender=st.selectbox("Gender", ["Female","Male"])
        marital=st.selectbox("Marital status", ["Married","Single","Unmarried","Divorced"])
    with c2:
        people=st.number_input("People visiting", min_value=1,max_value=20,value=3)
        stars=st.selectbox("Preferred hotel stars", [3,4,5])
        trips=st.number_input("Trips per year", min_value=0,max_value=30,value=3)
        passport=st.selectbox("Valid passport", [0,1], format_func=lambda x:"Yes" if x else "No")
        car=st.selectbox("Owns a car", [0,1], format_func=lambda x:"Yes" if x else "No")
    with c3:
        children=st.number_input("Children under 5 visiting",min_value=0,max_value=10,value=1)
        designation=st.selectbox("Job designation", ["Executive","Manager","Senior Manager","AVP","VP"])
        income=st.number_input("Gross monthly income (dataset units)",min_value=0,max_value=300000,value=23000,step=500)
        st.caption("Use the same income units as the training dataset; its currency was not specified.")
    submitted=st.form_submit_button("Score customer", type="primary", use_container_width=True)

if submitted:
    row = {
        "Age":float(age), "NumberOfPersonVisiting":int(people),
        "PreferredPropertyStar":int(stars), "NumberOfTrips":int(trips),
        "Passport":int(passport), "OwnCar":int(car),
        "NumberOfChildrenVisiting":int(children), "MonthlyIncome":float(income),
        "Occupation":occupation, "Gender":gender, "MaritalStatus":marital,
        "Designation":designation,"CityTier":int(city)
    }
    df = pd.DataFrame([row], columns=metadata["feature_names"])
    score = float(model.predict_proba(df)[0,1])
    st.metric("Estimated historical-purchase propensity", f"{score:.1%}")
    st.progress(min(1.0, max(0.0, score)))
    if score >= metadata["decision_threshold"]:
        st.success("Above the illustrative 50% score threshold. Consider for priority review.")
    else:
        st.write("Below the illustrative 50% threshold. Consider ranking with the wider lead pool before allocating contact capacity.")
    st.caption("A prediction is not a guaranteed purchase. Scores are not proven calibrated; benchmark campaign lift prospectively.")

with st.expander("Model card and governance", expanded=False):
    st.write("**Model:**",metadata["model_family"])
    st.write("**Training choice:**",metadata["selection_criterion"])
    st.write("**Holdout average precision:**",f"{metadata['test_metrics']['average_precision']:.3f}")
    st.write("**Holdout top-20% lift:**",f"{metadata['test_metrics']['top20_lift']:.2f}×")
    st.write("**Inputs:** 13 pre-contact fields; no pitch or follow-up variables")
    st.write("**Limitations:** target is past product purchase; monitor data drift, disparate outcomes, and retrain only with labelled customers")
