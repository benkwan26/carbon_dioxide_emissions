from __future__ import annotations
import argparse
from joblib import load
import numpy as np
import pandas as pd
from pathlib import Path

from feature_pipeline.preprocess import preprocess
from feature_pipeline.feature_engineering import feature_engineer

TARGET = "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = PROJECT_ROOT / 'models' / 'xgb_best_model.pkl'
TRAIN_FE_PATH = PROJECT_ROOT / 'data' / 'processed' / 'train_feature_engineered.csv'
DEFAULT_OUTPUT = PROJECT_ROOT / 'predictions.csv'

print("📂 Inference using project root:", PROJECT_ROOT)

if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    _train_cols = _train_cols.drop(columns=[TARGET], errors="ignore")
    TRAIN_FEATURE_COLUMNS = list(
        _train_cols.select_dtypes(include=[np.number, "bool"]).columns
    )
else:
    TRAIN_FEATURE_COLUMNS = None

def predict(input_df: pd.DataFrame, model_path: Path | str = DEFAULT_MODEL) -> pd.DataFrame:
    df = preprocess(input_df)
    df = feature_engineer(df)

    y_true = None
    if TARGET in df.columns:
        y_true = df[TARGET].tolist()
        df = df.drop(columns=TARGET)
    
    if TRAIN_FEATURE_COLUMNS:
        X = df.reindex(columns=TRAIN_FEATURE_COLUMNS, fill_value=0)
    else:
        X = df.select_dtypes(include=[np.number, "bool"]).copy()
    
    model = load(model_path)
    preds = model.predict(X)

    out = df.copy()
    out["Predicted Total CO2 Emissions"] = preds
    if y_true:
        out["Actual Total CO2 Emissions"] = y_true
    
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on new housing data (raw).")
    parser.add_argument("--input", type=str, required=True, help="Path to input RAW CSV file")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Path to save predictions CSV")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to trained model file")

    args = parser.parse_args()
    raw_df = pd.read_csv(args.input)
    preds_df = predict(raw_df, model_path=args.model)
    preds_df.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")
