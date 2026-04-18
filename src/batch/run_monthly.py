import pandas as pd
from pathlib import Path
from src.inference_pipeline.inference import predict

DATA_DIR = Path('data/processed')
TEST_PATH = DATA_DIR / 'test_cleaned.csv'
OUTPUT_DIR = Path('data/predictions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_monthly_predictions():
    df = pd.read_csv(TEST_PATH)
    
    grouped = df.groupby('Year')

    all_outputs = []
    for year, group in grouped:
        print(f"📅 Running predictions for {year} ({len(group)} rows)")

        preds_df = predict(group)

        out_path = OUTPUT_DIR / f"preds_{year}.csv"
        preds_df.to_csv(out_path, index=False)
        print(f"✅ Saved predictions to {out_path}")

        all_outputs.append(preds_df)

    return pd.concat(all_outputs, ignore_index=True)

if __name__ == "__main__":
    all_preds = run_monthly_predictions()
    print("🎉 Batch inference complete.")
    print(all_preds.head())
