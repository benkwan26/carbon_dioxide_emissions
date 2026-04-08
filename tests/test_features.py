import pandas as pd
import numpy as np
from pathlib import Path
import pytest

from src.feature_pipeline.load import split_data
from src.feature_pipeline.preprocess import drop_data, interpolate_data, fill_with_zeros
from src.feature_pipeline.feature_engineering import add_core_features, add_per_capita, add_log_features, add_pct_change, feature_engineer

# ---------- load.py - unit test ----------

def test_split_data_creates_splits():
    df = pd.DataFrame({
        "Year": [2019, 2020, 2021, 2022],
        "Value": [10, 20, 30, 40],
    })

    train_df, test_df = split_data(df)

    assert train_df["Year"].tolist() == [2019, 2020]
    assert test_df["Year"].tolist() == [2021, 2022]
    assert train_df["Value"].tolist() == [10, 20]
    assert test_df["Value"].tolist() == [30, 40]
    print("✅ Data splitting test passed")

# ---------- preprocess.py - unit test ----------

def test_drop_data_filters_years_countries_and_indicators():
    df = pd.DataFrame({
        "Country Name": ["Afghanistan", "Monaco", "Canada"],
        "Year": [2021, 2021, 1985],
        "Access to electricity (% of population)": [97.7, 100.0, 99.0],
        "Agricultural irrigated land (% of total agricultural land)": [6.3, 1.2, 5.0],
        "Population, total": [40000412.0, 39000.0, 38000000.0],
    })

    out = drop_data(
        df,
        years=[1985],
        countries=["Monaco"],
        indicators=["Agricultural irrigated land (% of total agricultural land)"],
    )

    assert out["Country Name"].tolist() == ["Afghanistan"]
    assert out["Year"].tolist() == [2021]
    assert "Agricultural irrigated land (% of total agricultural land)" not in out.columns
    assert "Access to electricity (% of population)" in out.columns
    assert "Population, total" in out.columns
    print("✅ Data dropping test passed")

def test_interpolate_data_by_country_on_numeric_columns():
    df = pd.DataFrame({
        "Country Name": ["Afghanistan", "Afghanistan", "Afghanistan", "Canada", "Canada"],
        "Year": [2019, 2020, 2021, 2020, 2021],
        "Access to electricity (% of population)": [97.0, None, 99.0, None, 100.0],
        "Population, total": [38000000.0, None, 40000000.0, 37000000.0, None],
    })

    out = interpolate_data(df.copy())

    assert out.loc[1, "Access to electricity (% of population)"] == 98.0
    assert out.loc[3, "Access to electricity (% of population)"] == 100.0
    assert out.loc[4, "Population, total"] == 37000000.0
    print("✅ Data interpolation test passed")

def test_fill_with_zeros_replaces_nan():
    df = pd.DataFrame({
            "Access to electricity (% of population)": [97.7, None],
            "Population, total": [40000412.0, None],
    })

    out = fill_with_zeros(df)

    assert out.loc[0, "Access to electricity (% of population)"] == 97.7
    assert out.loc[1, "Access to electricity (% of population)"] == 0
    assert out.loc[0, "Population, total"] == 40000412.0
    assert out.loc[1, "Population, total"] == 0
    print("✅ Fill with zeros test passed")

# ---------- feature_engineering.py - unit tests ----------

def test_add_core_features_creates_expected_columns_and_values():
    df = pd.DataFrame({
            "Electricity production from coal sources (% of total)": [10.0],
            "Electricity production from oil sources (% of total)": [5.0],
            "Electricity production from natural gas sources (% of total)": [15.0],
            "Electricity production from hydroelectric sources (% of total)": [20.0],
            "Electricity production from nuclear sources (% of total)": [0.0],
            "Electricity production from renewable sources, excluding hydroelectric (% of total)": [10.0],
            "Energy use (kg of oil equivalent per capita)": [2.0],
            "Urban population (% of total population)": [50.0],
    })

    out = add_core_features(df)

    assert out.loc[0, "fossil_share"] == 30.0
    assert out.loc[0, "clean_share"] == 30.0
    assert out.loc[0, "energy_x_fossil"] == 60.0
    assert out.loc[0, "energy_x_clean"] == 60.0
    assert out.loc[0, "urban_energy"] == 100.0
    print("✅ Add core features test passed")

def test_add_per_capita_creates_expected_columns():
    df = pd.DataFrame({
            "Population, total": [100.0],
            "Agricultural land (sq. km)": [200.0],
            "Forest area (sq. km)": [300.0],
            "Annual freshwater withdrawals, total (billion cubic meters)": [10.0],
    })

    out = add_per_capita(df)

    assert out.loc[0, "ag_land_sqkm_per_capita"] == 2.0
    assert out.loc[0, "forest_area_sqkm_per_capita"] == 3.0
    assert out.loc[0, "water_withdrawal_bcm_per_capita"] == 0.1
    print("✅ Add per capita features test passed")

def test_add_log_features_adds_log1p_columns_when_present():
    df = pd.DataFrame({
            "Population, total": [9.0],
            "Urban population": [4.0],
    })

    out = add_log_features(df)

    assert "log1p_Population, total" in out.columns
    assert "log1p_Urban population" in out.columns
    assert out.loc[0, "log1p_Population, total"] == pytest.approx(np.log1p(9.0))
    assert out.loc[0, "log1p_Urban population"] == pytest.approx(np.log1p(4.0))
    print("✅ Add log features test passed")

def test_add_pct_change_groups_by_country():
    df = pd.DataFrame({
            "Country Name": ["A", "A", "B", "B"],
            "Energy use (kg of oil equivalent per capita)": [10.0, 20.0, 5.0, 10.0],
            "Electric power consumption (kWh per capita)": [100.0, 200.0, 50.0, 75.0],
            "Population, total": [1000.0, 1100.0, 500.0, 600.0],
            "Urban population": [400.0, 500.0, 200.0, 240.0],
            "fossil_share": [30.0, 60.0, 10.0, 20.0],
    })

    out = add_pct_change(df)

    assert out.loc[0, "Energy use (kg of oil equivalent per capita)_yoy_pct"] != out.loc[1, "Energy use (kg of oil equivalent per capita)_yoy_pct"]
    assert out.loc[1, "Energy use (kg of oil equivalent per capita)_yoy_pct"] == pytest.approx(1.0)
    assert out.loc[3, "Population, total_yoy_pct"] == pytest.approx(0.2)
    print("✅ Add %% change features test passed")

def test_feature_engineer_replaces_nan_and_inf_with_zero():
    df = pd.DataFrame({
            "Country Name": ["A"],
            "Electricity production from coal sources (% of total)": [0.0],
            "Electricity production from oil sources (% of total)": [0.0],
            "Electricity production from natural gas sources (% of total)": [0.0],
            "Electricity production from hydroelectric sources (% of total)": [0.0],
            "Electricity production from nuclear sources (% of total)": [0.0],
            "Electricity production from renewable sources, excluding hydroelectric (% of total)": [0.0],
            "Energy use (kg of oil equivalent per capita)": [0.0],
            "Urban population (% of total population)": [0.0],
            "Population, total": [0.0],
            "Agricultural land (sq. km)": [0.0],
            "Forest area (sq. km)": [0.0],
            "Annual freshwater withdrawals, total (billion cubic meters)": [0.0],
            "Urban population": [0.0],
            "Electric power consumption (kWh per capita)": [0.0],
            "Electricity production from renewable sources, excluding hydroelectric (kWh)": [0.0],
    })

    out = feature_engineer(df.copy())

    numeric = out.select_dtypes(include="number").to_numpy()
    assert np.isfinite(numeric).all()
    print("✅ Eliminate inf and NaN test passed")