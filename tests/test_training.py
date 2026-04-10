from joblib import load
import math
from pathlib import Path

from src.training_pipeline.train import train_model
from src.training_pipeline.test import test_model
from src.training_pipeline.tune import tune_model

TRAIN_PATH = Path("data/processed/train_feature_engineered.csv")
TEST_PATH = Path("data/processed/test_feature_engineered.csv")

def _assert_metrics(m):
    assert set(m.keys()) == {'mae', 'rmse', 'r2'}
    assert all(isinstance(v, float) and math.isfinite(v) for v in m.values())

def test_train_creates_model_and_metrics(tmp_path):
    out_path = tmp_path / 'xgb_model.pkl'

    _, metrics = train_model(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        model_output=out_path,
        model_params={'n_estimators': 20, 'max_depth': 4, 'learning_rate': 0.1},
        sample_frac=0.02
    )

    assert out_path.exists()
    _assert_metrics(metrics)
    model = load(out_path)
    assert model is not None
    print("✅ train_model test passed")

def test_test_model_works_with_saved_model(tmp_path):
    model_path = tmp_path / 'xgb_model.pkl'

    train_model(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        model_output=model_path,
        model_params={'n_estimators': 20},
        sample_frac=0.02
    )

    metrics = test_model(model_path=model_path, test_path=TEST_PATH, sample_frac=0.02)
    _assert_metrics(metrics)
    print("✅ test_model test passed")

def test_tune_saves_best_model(tmp_path):
    model_out = tmp_path / 'xgb_model.pkl'
    tracking_dir = tmp_path / 'mlruns'

    best_params, best_metrics = tune_model(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        model_output=model_out,
        n_trials=2,
        sample_frac=0.02,
        tracking_uri=str(tracking_dir),
        experiment_name='test_xgb_optuna'
    )

    assert model_out.exists()
    assert isinstance(best_params, dict) and best_params
    _assert_metrics(best_metrics)
    print("✅ tune_model test passed")
