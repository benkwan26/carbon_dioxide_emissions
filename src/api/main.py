import boto3, os
from fastapi import FastAPI
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from src.inference_pipeline.inference import predict

B2_BUCKET = os.getenv("B2_BUCKET", "carbon-dioxide-data")
REGION = os.getenv("B2_REGION", "ca-east-006")
b2 = boto3.client("s3", region_name=REGION)

def load_from_b2(key, local_path):
    local_path = Path(local_path)

    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"📥 Downloading {key} from S3…")
        b2.download_file(B2_BUCKET, key, str(local_path))

    return str(local_path)

MODEL_PATH = Path(load_from_b2("models/xgb_best_model.pkl", "models/xgb_best_model.pkl"))
TRAIN_FE_PATH = Path(load_from_b2("processed/train_feature_engineered.csv", "data/processed/train_feature_engineered.csv"))

if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = list(
        _train_cols.select_dtypes(include=['number']).columns
    )
else:
    TRAIN_FEATURE_COLUMNS = None

app = FastAPI(title="Carbon Dioxide Emission Regression")

# / → simple landing endpoint to confirm API is alive.
@app.get('/')
def root():
    return {"message": "Carbon Dioxide Emission Regression API is running 🚀"}

# /health → checks if model exists, returns status info (like expected feature count).
@app.get('/health')
def health():
    status: Dict[str, Any] = {'model_path': str(MODEL_PATH)}

    if not MODEL_PATH.exists():
        status['status'] = 'unhealthy'
        status['error'] = "Model not found"
    else:
        status['status'] = 'healthy'

        if TRAIN_FEATURE_COLUMNS:
            status['n_features_expected'] = len(TRAIN_FEATURE_COLUMNS)
    
    return status

# Prediction Endpoint: This is the core ML serving endpoint.
@app.post('/predict')
def predict_batch(data: List[Dict]):
    if not MODEL_PATH.exists():
        return {'error': f"Model not found at {str(MODEL_PATH)}"}
    
    df = pd.DataFrame(data)
    if df.empty:
        return {'error': "No data provided"}
    
    preds_df = predict(df, model_path=MODEL_PATH)
    resp = {'predictions': preds_df['predicted_price'].astype(float).tolist()}
    if 'actual_price' in preds_df.columns:
        resp['actuals'] = preds_df['actual_price'].astype(float).tolist()

    return resp

from src.batch.run_monthly import run_monthly_predictions

# Trigger a monthly batch job via API.
@app.post("/run-batch")
def run_batch():
    preds = run_monthly_predictions()

    return {
        'status': 'success',
        'rows_predicted': int(len(preds)),
        'output_dir': 'data/predictions/'
    }

@app.get('/latest-predictions')
def latest_predictions(limit: int = 5):
    pred_dir = Path('data/predictions')
    files = sorted(pred_dir.glob('preds_*.csv'))
    if not files:
        return {'error': "No predictions found"}
    
    latest_file = files[-1]
    df = pd.read_csv(latest_file)

    return {
        'file': latest_file.name,
        'rows': int(len(df)),
        'preview': df.head(limit).to_dict(orient='records')
    }
