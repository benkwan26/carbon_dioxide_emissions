from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_TEST = Path("data/processed/test_feature_engineered.csv")
DEFAULT_MODEL = Path("models/xgb_model.pkl")

def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state:int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

def test_model(
        model_path: Path | str = DEFAULT_MODEL,
        test_path: Path | str = DEFAULT_TEST,
        sample_frac: Optional[float] = None,
        random_state: int = 42
    ) -> Dict[str, float]:
    test_df = pd.read_csv(test_path)
    test_df = _maybe_sample(test_df, sample_frac, random_state)

    target = "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)"
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    model = load(model_path)
    y_hat = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_hat))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
    r2 = float(r2_score(y_test, y_hat))
    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}

    print("📊 Evaluation:")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    return model, metrics

if __name__ == '__main__':
    test_model()