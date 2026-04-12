import os
import pandas as pd
from pathlib import Path
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference import predict

@pytest.fixture(scope='session')
def sample_df() -> pd.DataFrame:
    sample_path = ROOT / "data/raw/test.csv"
    df = pd.read_csv(sample_path).sample(5, random_state=42).reset_index(drop=True)
    return df

def test_inference_runs_and_returns_predictions(sample_df):
    preds_df = predict(sample_df)

    assert not preds_df.empty
    assert "Predicted Total CO2 Emissions" in preds_df.columns
    assert pd.api.types.is_numeric_dtype(preds_df["Predicted Total CO2 Emissions"])

    print("✅ Inference pipeline test passed. Predictions:")
    print(preds_df[["Predicted Total CO2 Emissions"]].head())