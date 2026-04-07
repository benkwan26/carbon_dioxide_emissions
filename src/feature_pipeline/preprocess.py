import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

YEARS_TO_DROP = range(1960, 1990)
COUNTRIES_TO_DROP = [
    "Sint Maarten (Dutch part)",
    "South Sudan",
    "St. Martin (French part)",
    "Monaco",
    "Kosovo",
    "Macao SAR, China",
    "Channel Islands",
    "Montenegro",
    "Curacao",
    "Northern Mariana Islands",
    "Isle of Man",
    "San Marino",
    "Liechtenstein"
]
INDICATORS_TO_DROP = [
    "Disaster risk reduction progress score (1-5 scale; 5=best)", 
    "Droughts, floods, extreme temperatures (% of population, average 1990-2009)", 
    "Marine protected areas (% of territorial waters)", 
    "Terrestrial and marine protected areas (% of total territorial area)", 
    "Terrestrial protected areas (% of total land area)", 
    "Community health workers (per 1,000 people)",
    "Agricultural irrigated land (% of total agricultural land)",
    "CPIA public sector management and institutions cluster average (1=low to 6=high)",
    "Land area where elevation is below 5 meters (% of total land area)",
    "Population living in areas where elevation is below 5 meters (% of total population)",
    "Rural land area where elevation is below 5 meters (% of total land area)",
    "Rural land area where elevation is below 5 meters (sq. km)",
    "Rural population living in areas where elevation is below 5 meters (% of total population)",
    "Urban land area where elevation is below 5 meters (% of total land area)",
    "Urban land area where elevation is below 5 meters (sq. km)",
    "Urban population living in areas where elevation is below 5 meters (% of total population)",
    "Prevalence of underweight, weight for age (% of children under 5)",
    "Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)"
]

def load_data(dir: Path) -> pd.DataFrame:
    return pd.read_csv(dir)

def drop_data(df: pd.DataFrame, years: list[int] = YEARS_TO_DROP, countries: list[str] = COUNTRIES_TO_DROP, indicators: list[str] = INDICATORS_TO_DROP) -> pd.DataFrame:
    df = df[~df['Year'].isin(years)]
    df = df[~df['Country Name'].isin(countries)]
    df = df.drop(columns=indicators)

    return df

def interpolate_data(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns

    df[num_cols] = (
        df.groupby("Country Name", group_keys=False)[num_cols]
        .apply(lambda g: g.interpolate(method="linear", limit_direction="both"))
    )

    return df

def fill_with_zeros(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(0)

def save_data(df: pd.DataFrame, dir: Path) -> None:
    df.to_csv(dir, index=False)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_data(df)
    df = interpolate_data(df)
    df = fill_with_zeros(df)

    return df

def main():
    train_df = load_data(RAW_DIR + 'train.csv')
    test_df = load_data(RAW_DIR + 'test.csv')

    train_df = preprocess(train_df)
    test_df = preprocess(test_df)

    save_data(train_df, PROCESSED_DIR / 'train_cleaned.csv')
    save_data(test_df, PROCESSED_DIR / 'test_cleaned.csv')

    print(f"✅ Data preprocessing completed (saved to {PROCESSED_DIR}).")
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

if __name__ == '__main__':
    main()