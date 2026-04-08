import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Utility Functions ----------

def load_data(dir: Path) -> pd.DataFrame:
    return pd.read_csv(dir)

def save_data(df: pd.DataFrame, dir: Path) -> None:
    df.to_csv(dir, index=False)

# ---------- Feature Functions ----------

# Shares & Ratios
def add_core_features(df: pd.DataFrame) -> pd.DataFrame:    
    # Fossil vs clean shares
    df["fossil_share"] = (
        df["Electricity production from coal sources (% of total)"].fillna(0) +
        df["Electricity production from oil sources (% of total)"].fillna(0) +
        df["Electricity production from natural gas sources (% of total)"].fillna(0)
    )
    
    df["clean_share"] = (
        df["Electricity production from hydroelectric sources (% of total)"].fillna(0) +
        df["Electricity production from nuclear sources (% of total)"].fillna(0) +
        df["Electricity production from renewable sources, excluding hydroelectric (% of total)"].fillna(0)
    )
    
    # Energy intensity style features
    df["energy_x_fossil"] = df["Energy use (kg of oil equivalent per capita)"] * df["fossil_share"]
    df["energy_x_clean"] = df["Energy use (kg of oil equivalent per capita)"] * df["clean_share"]
    
    # Urbanization intensity
    df["urban_energy"] = df["Urban population (% of total population)"] * df["Energy use (kg of oil equivalent per capita)"]
    
    return df

# Per‑Capita Conversions
def add_per_capita(df: pd.DataFrame) -> pd.DataFrame:
    pop = df["Population, total"]
    
    df["ag_land_sqkm_per_capita"] = df["Agricultural land (sq. km)"] / pop
    df["forest_area_sqkm_per_capita"] = df["Forest area (sq. km)"] / pop
    df["water_withdrawal_bcm_per_capita"] = df["Annual freshwater withdrawals, total (billion cubic meters)"] / pop
    
    return df

# Log Transforms
def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Population, total",
        "Urban population",
        "Energy use (kg of oil equivalent per capita)",
        "Electric power consumption (kWh per capita)",
        "Agricultural land (sq. km)",
        "Forest area (sq. km)",
        "Electricity production from renewable sources, excluding hydroelectric (kWh)"
    ]
    
    for col in cols:
        if col in df.columns:
            df[f"log1p_{col}"] = np.log1p(df[col])
    
    return df

# Growth Rates (YoY %)
def add_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Energy use (kg of oil equivalent per capita)",
        "Electric power consumption (kWh per capita)",
        "Population, total",
        "Urban population",
        "fossil_share"
    ]

    g = df.groupby("Country Name", sort=False)
    
    for col in cols:
        df[f"{col}_yoy_pct"] = g[col].pct_change()

    return df

# ---------- Pipeline ----------

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = add_core_features(df)
    df = add_per_capita(df)
    df = add_log_features(df)
    df = add_pct_change(df)
    df = df.replace([np.inf, -np.inf, np.nan], 0)

    return df

def run_feature_engineering(dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_data(dir / 'train_cleaned.csv')
    test_df = load_data(dir / 'test_cleaned.csv')

    print(f"Train date range: {train_df['Year'].min()} to {train_df['Year'].max()}")
    print(f"Train date range: {test_df['Year'].min()} to {test_df['Year'].max()}")

    train_df = feature_engineer(train_df)
    test_df = feature_engineer(test_df)

    save_data(dir / 'train_feature_engineered.csv')
    save_data(dir / 'test_feature_engineered.csv')

    print(f"✅ Feature engineering completed (saved to {dir}).")
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

    return train_df, test_df

if __name__ == '__main__':
    run_feature_engineering()