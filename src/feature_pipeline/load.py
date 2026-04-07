import pandas as pd
from pathlib import Path

DATA_DIR = Path('data/raw')
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_data(dir: Path) -> pd.DataFrame:
    return pd.read_csv(dir)

def melt_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=["Country Code", "Indicator Code", "2025", "Unnamed: 70"], inplace=True)

    df = df.melt(
        id_vars=["Country Name", "Indicator Name"],
        var_name="Year",
        value_name="Value"
    )

    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    return df

def merge_data(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df_concat = pd.concat([df1, df2])

    return df_concat.pivot_table(
        index=['Country Name', 'Year'],
        columns='Indicator Name',
        values='Value'
    ).reset_index()

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff_year = 2021

    train_df = df[df['Year'] < cutoff_year]
    test_df = df[cutoff_year <= df['Year']]

    return train_df, test_df

def save_data(df: pd.DataFrame, dir: Path) -> None:
    df.to_csv(dir, index=False)

def main():
    carbon_df = load_data(DATA_DIR + 'carbon_dioxide_emissions.csv')
    indicator_df = load_data(DATA_DIR + 'world_development_indicators.csv')

    carbon_df = melt_data(carbon_df)
    indicator_df = melt_data(indicator_df)

    df = merge_data(carbon_df, indicator_df)

    train_df, test_df = split_data(df)

    save_data(train_df, DATA_DIR / 'train.csv')
    save_data(test_df, DATA_DIR / 'test.csv')

    print(f"✅ Data split completed (saved to {DATA_DIR}).")
    print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

if __name__ == '__main__':
    main()