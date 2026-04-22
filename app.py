import boto3, os
import pandas as pd
from pathlib import Path
import plotly.express as px
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
B2_BUCKET = os.getenv("B2_BUCKET", "carbon-dioxide-data")
REGION = os.getenv("B2_REGION", "ca-east-006")

b2 = boto3.client("s3", region_name=REGION)

def load_from_b2(key, local_path):
    local_path = Path(local_path)

    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"📥 Downloading {key} from S3…")
        b2.download_file(B2_BUCKET, key, str(local_path))

    return str(local_path)

TEST_ENGINEERED_PATH = load_from_b2(
    "processed/test_feature_engineered.csv",
    "data/processed/test_feature_engineered.csv"
)
TEST_META_PATH = load_from_b2(
    "processed/test_cleaned.csv",
    "data/processed/test_cleaned.csv"
)

@st.cache_data
def load_data():
    fe = pd.read_csv(TEST_ENGINEERED_PATH)
    meta = pd.read_csv(TEST_META_PATH, parse_dates=["Year"])[["Year", "Country Name"]]

    if len(fe) != len(meta):
        st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()
    
    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["country"] = meta["Country Name"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["Actual Total CO2 Emissions"] = fe["Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)"].dt.year

    return fe, disp

df_fe, df_disp = load_data()

st.title("Carbon Dioxide Emission Prediction")

years = sorted(df_disp["year"].unique())
months = list(range(1, 13))
countries = ["All"] + sorted(df_disp["country"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    country = st.selectbox("Select Countries", years, index=0)