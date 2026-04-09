from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple

from joblib import dump
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

import mlflow
import mlflow.xgboost

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

def _load_data(
        train_path: Path | str,
        test_path: Path | str,
        sample_frac: Optional[float],
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = _maybe_sample(train_df, sample_frac, random_state)
    test_df = _maybe_sample(test_df, sample_frac, random_state)

    target = "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)"
    X_train, y_train = test_df.drop(columns=[target]), train_df[target]
    X_test, y_test = train_df.drop(columns=[target]), test_df[target]

    return X_train, y_train, X_test, y_test

def tune_model(
        train_path: Path | str = DEFAULT_TRAIN,
        test_path: Path | str = DEFAULT_TEST,
        model_output: Path | str = DEFAULT_OUT,
        n_trials: int = 15,
        sample_frac: Optional[float] = None,
        tracking_uri: Optional[str] = None,
        experiment_name: str = 'xgboost_optuna_carbon_dioxide',
        random_state: int = 42
    ) -> tuple[Dict, Dict]:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, y_train, X_test, y_test = _load_data(train_path, test_path, sample_frac, random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': random_state,
            'n_jobs': -1,
            'tree_method': 'hist'
        }

        with mlflow.start_run(nested=True):
            model = XGBRegressor(**params)
            model.fit(X_train, y_train)

            y_hat = model.predict(X_test)
            mae = float(mean_absolute_error(y_test, y_hat))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
            r2 = float(r2_score(y_test, y_hat))

            mlflow.log_params(params)
            mlflow.log_metrics({'mae': mae, 'rmse': rmse, 'r2': r2})

        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_trial.params
    print("Best params:", best_params)

    best_model= XGBRegressor(**{**best_params, 'random_state': random_state, 'n_jobs': -1, 'tree_method': 'hist'})
    best_model.fit(X_train, y_train)
    y_hat = best_model.predict(X_test)
    best_metrics = {
        'mae': float(mean_absolute_error(y_test, y_hat)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_hat))),
        'r2': float(r2_score(y_test, y_hat))
    }
    print("📊 Best tuned model metrics:", best_metrics)

    out = Path(model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, out)
    print(f"✅ Model trained. Saved to {out}")

    with mlflow.start_run(run_name='best_xgb_model'):
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)
        mlflow.xgboost.log_model(best_model, 'model')

    return best_params, best_metrics

if __name__ == '__main__':
    tune_model()