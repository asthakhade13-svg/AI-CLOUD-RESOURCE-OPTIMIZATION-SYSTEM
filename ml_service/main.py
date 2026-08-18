from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Import core business logic from src
from src.pipeline import BASE_FEATURES, preprocess_single_record
from src.forecasting import forecast_next_workloads
from src.capacity import calculate_required_servers, estimate_prediction_uncertainty
from src.anomaly import detect_anomaly_record
from src.explainability import explain_prediction_shap
import shap

app = FastAPI(
    title="ML Model Service",
    description="Microservice exposing capacity predictions, forecasts, anomalies, and SHAP explainability.",
    version="1.0.0"
)

MODEL_PATH = "artifacts/cloud_resource_optimization_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"
CLEANED_DATA_PATH = "data/cleaned_workload.csv"
FEATURES_LIST_PATH = "artifacts/features_list.pkl"

model = None
scaler = None
anomaly_detector = None
anomaly_scaler = None
anomaly_features = None
shap_explainer = None
features_list = None

# Thread-safe sliding buffer
history_buffer = None
buffer_lock = threading.Lock()

@app.on_event("startup")
def load_assets():
    global model, scaler, anomaly_detector, anomaly_scaler, anomaly_features, shap_explainer, features_list, history_buffer
    
    # 1. Load capacity model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        
    # 2. Load scalers
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    if os.path.exists(FEATURES_LIST_PATH):
        features_list = joblib.load(FEATURES_LIST_PATH)
        
    # 3. Seed history buffer
    if os.path.exists(CLEANED_DATA_PATH):
        df_clean = pd.read_csv(CLEANED_DATA_PATH)
        raw_columns = ["timestamp"] + BASE_FEATURES
        available_cols = [c for c in raw_columns if c in df_clean.columns]
        history_buffer = df_clean[available_cols].tail(30).reset_index(drop=True)
        
    # 4. Load anomaly models
    anomaly_model_path = "artifacts/anomaly_detector.pkl"
    anomaly_scaler_path = "artifacts/anomaly_scaler.pkl"
    anomaly_features_path = "artifacts/anomaly_features_list.pkl"
    
    if os.path.exists(anomaly_model_path):
        anomaly_detector = joblib.load(anomaly_model_path)
    if os.path.exists(anomaly_scaler_path):
        anomaly_scaler = joblib.load(anomaly_scaler_path)
    if os.path.exists(anomaly_features_path):
        anomaly_features = joblib.load(anomaly_features_path)
        
    # 5. Load SHAP
    if model is not None:
        shap_explainer = shap.TreeExplainer(model)

class PredictRawInput(BaseModel):
    cpu_usage: float
    memory_usage: float
    network_in: float
    network_out: float
    network_traffic: Optional[float] = None
    disk_read: float
    disk_write: float
    active_users: int
    request_rate: float
    response_time: float
    error_rate: float
    current_servers: int
    server_cost: float
    safety_margin: float = 0.10
    min_servers: int = 1
    max_servers: int = 20

@app.get("/health")
def health():
    return {"status": "healthy", "assets_loaded": model is not None}

@app.post("/predict_raw")
def predict_raw(payload: PredictRawInput):
    global model, scaler, history_buffer, anomaly_detector, anomaly_scaler, anomaly_features, shap_explainer, features_list
    
    if model is None or scaler is None or history_buffer is None:
        raise HTTPException(status_code=503, detail="ML Service assets not fully loaded.")
        
    input_dict = payload.dict()
    if input_dict.get("network_traffic") is None:
        input_dict["network_traffic"] = input_dict["network_in"] + input_dict["network_out"]
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_dict["timestamp"] = now_str
    
    try:
        new_row_df = pd.DataFrame([input_dict])
        
        # 1. Update history and calculate workloads forecast
        with buffer_lock:
            combined_df = pd.concat([history_buffer, new_row_df], ignore_index=True)
            context_df = combined_df.tail(30).reset_index(drop=True)
            
            # Use core src forecast function
            forecasts = forecast_next_workloads(context_df)
            
            # Update cache
            raw_cols = ["timestamp"] + BASE_FEATURES
            history_buffer = context_df[raw_cols].tail(30).reset_index(drop=True)
            
        # 2. Predictive Server Sizing
        proj_15 = input_dict.copy()
        fc_15 = forecasts["15min"]
        proj_15["cpu_usage"] = fc_15["cpu_usage"]
        proj_15["memory_usage"] = fc_15["memory_usage"]
        proj_15["network_traffic"] = fc_15["network_traffic"]
        proj_15["active_users"] = int(np.round(fc_15["active_users"]))
        proj_15["request_rate"] = fc_15["request_rate"]
        proj_15["response_time"] = fc_15["response_time"]
        
        future_dt = datetime.strptime(now_str, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=15)
        proj_15["timestamp"] = future_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        proj_df = pd.DataFrame([proj_15])
        projected_context = pd.concat([context_df, proj_df], ignore_index=True).tail(31).reset_index(drop=True)
        
        scaled_projected_input = preprocess_single_record(projected_context, scaler)
        raw_pred = model.predict(scaled_projected_input)[0]
        
        capacity = calculate_required_servers(
            prediction=raw_pred,
            current_servers=payload.current_servers,
            min_servers=payload.min_servers,
            max_servers=payload.max_servers,
            safety_margin=payload.safety_margin
        )
        uncertainty = estimate_prediction_uncertainty(model, scaled_projected_input)
        
        # 3. Anomaly check
        new_row_with_time = new_row_df.copy()
        latest_ts = pd.to_datetime(now_str)
        new_row_with_time["hour"] = latest_ts.hour
        new_row_with_time["day_of_week"] = latest_ts.dayofweek
        new_row_with_time["sin_hour"] = np.sin(2 * np.pi * latest_ts.hour / 24.0)
        new_row_with_time["cos_hour"] = np.cos(2 * np.pi * latest_ts.hour / 24.0)
        new_row_with_time["sin_day_of_week"] = np.sin(2 * np.pi * latest_ts.dayofweek / 7.0)
        new_row_with_time["cos_day_of_week"] = np.cos(2 * np.pi * latest_ts.dayofweek / 7.0)
        
        anomaly_res = detect_anomaly_record(new_row_with_time, history_buffer)
        
        # 4. SHAP Local explanations
        xai_res = explain_prediction_shap(shap_explainer, scaled_projected_input, features_list, payload.current_servers)
        
        return {
            "predicted_servers": capacity["predicted_servers"],
            "recommended_servers": capacity["recommended_servers"],
            "uncertainty_std": uncertainty["uncertainty_std"],
            "is_anomaly": anomaly_res["is_anomaly"],
            "anomaly_score": anomaly_res["anomaly_score"],
            "severity": anomaly_res["severity"],
            "affected_metrics": anomaly_res["affected_metrics"],
            "recommendation": anomaly_res["recommendation"],
            "shap_explanation": xai_res["shap_explanation"],
            "shap_contributions": xai_res["category_contributions"],
            "forecasts": forecasts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
