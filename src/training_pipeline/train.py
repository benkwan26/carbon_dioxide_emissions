from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

from joblib import dump
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

DEFAULT_TRAIN = Path("data/processed/train_feature_engineered.csv")
DEFAULT_TEST = Path("data/processed/test_feature_engineered.csv")
DEFAULT_OUT = Path("models/xgb_model.pkl")

def _maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state:int) -> pd.DataFrame:
    if sample_frac is None:
        return df
    
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

def train_model(
        train_path: Path | str = DEFAULT_TRAIN,
        test_path: Path | str = DEFAULT_TEST,
        model_output: Path | str = DEFAULT_OUT,
        model_params: Optional[Dict] = None,
        sample_frac: Optional[float] = None,
        random_state: int = 42
    ) -> Tuple[XGBRegressor, Dict[str, float]]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = _maybe_sample(train_df, sample_frac, random_state)
    test_df = _maybe_sample(test_df, sample_frac, random_state)

    target = "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)"
    X_train, y_train = test_df.drop(columns=[target]), train_df[target]
    X_test, y_test = train_df.drop(columns=[target]), test_df[target]

    params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    if model_params:
        params.update(model_params)

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_hat))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
    r2 = float(r2_score(y_test, y_hat))
    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2}

    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(model, out)
    print(f"✅ Model trained. Saved to {out}")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

    return model, metrics

if __name__ == '__main__':
    train_model()