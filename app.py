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
    disp["country"] = meta["Country Name"]
    disp["year"] = meta["year"].dt.year
    disp["Actual Total CO2 Emissions"] = fe["Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)"].dt.year

    return fe, disp

df_fe, df_disp = load_data()

st.title("Carbon Dioxide Emission Prediction")

years = sorted(df_disp["year"].unique())
months = list(range(1, 13))
countries = ["All"] + sorted(df_disp["country"].dropna().unique())

col1, col2 = st.columns(2)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    country = st.selectbox("Select Countries", years, index=0)

if st.button("Show Predictions 🚀"):
    mask = (df_disp["year"] == year)

    if countries != "All":
        mask &= (df_disp["country"] == countries)
    
    idx = df_disp.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(f"📅 Running predictions for **{year}** | Countries: **{countries}**")

        payload = df_fe.loc[idx].to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            view = df_disp.loc[idx, ["year", "country", "Actual Total CO2 Emissions"]].copy()
            view = view.sort_values("year")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            if actuals and len(actuals) == len(view):
                view["Actual Total CO2 Emissions"] = pd.Series(actuals, index=view.index).astype(float)
            
            mae = (view["prediction"] - view["Actual Total CO2 Emissions"]).abs().mean()
            rmse = ((view["prediction"] - view["Actual Total CO2 Emissions"]) ** 2).mean() ** 0.5
            avg_pct_error = ((view["prediction"] - view["Actual Total CO2 Emissions"]).abs() / view["Actual Total CO2 Emissions"]).mean() * 100

            st.subheader("Predictions vs. Actuals")
            st.dataframe(
                view[["year", "countries", "Actual Total CO2 Emissions", "prediction"]].reset_index(drop=True),
                use_container_width=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae:,.0f}")
            with c2:
                st.metric("RMSE", f"{rmse:,.0f}")
            with c3:
                st.metric("Avg %% Error", f"{avg_pct_error:.2f}%")
            
            if country == "All":
                yearly_data = df_disp[df_disp["year"] == year].copy()
                idx_all = yearly_data.index
                payload_all = df_fe.loc[idx_all].to_dict(orient="records")

                resp_all = requests.post(API_URL, json=payload_all, timeout=60)
                resp_all.raise_for_status()
                preds_all = resp_all.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_all, index=yearly_data.index).astype(float)
            else:
                yearly_data = df_disp[df_disp["year"] == year & (df_disp["country"] == countries)].copy()
                idx_countries = yearly_data.index
                payload_countries = df_fe.loc[idx_countries].to_dict(orient="records")

                resp_countries = requests.post(API_URL, json=payload_countries, timeout=60)
                resp_countries.raise_for_status()
                preds_countries = resp_countries.json().get("predictions", [])

                yearly_data["prediction"] = pd.Series(preds_countries, index=yearly_data.index).astype(float)

            fig = px.line(
                yearly_data,
                x="year",
                y=["Actual Total CO2 Emissions", "prediction"],
                markers=True,
                labels={"value": "Total CO2 Emissions"},
                title=f"Yearly Trend — {year}{'' if countries=='All' else f' — {countries}'}"
            )

            highlight_year = year
            fig.add_vrect(
                x0=highlight_year - 0.5,
                x1=highlight_year + 0.5,
                fillcolor="red",
                opacity=0.1,
                layer="below",
                line_width=0,
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API call failed: {e}")
            st.exception(e)

else:
    st.info("Choose filters and click **Show Predictions** to compute.")
