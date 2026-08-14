from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime, timedelta
from src.pipeline import BASE_FEATURES, preprocess_single_record, DatasetValidationError
from src.forecasting import forecast_next_workloads, FORECAST_METRICS

app = FastAPI(
    title="AI Cloud Resource Optimization API",
    description="A FastAPI backend leveraging two-stage predictive autoscaling: time-series workload forecasting + capacity modeling.",
    version="4.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to serialized objects
MODEL_PATH = "artifacts/cloud_resource_optimization_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"
CLEANED_DATA_PATH = "data/cleaned_workload.csv"

# Global variables to store loaded assets
model = None
scaler = None

# Thread-safe sliding window buffer to cache the historical workload state
history_buffer = None
buffer_lock = threading.Lock()

@app.on_event("startup")
def load_assets_and_seed_buffer():
    global model, scaler, history_buffer
    
    # 1. Load Capacity Predictor Model (Stage 2)
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Capacity Predictor loaded successfully.")
        except Exception as e:
            print(f"Error loading capacity model: {e}")
    else:
        print(f"Warning: Capacity model file '{MODEL_PATH}' not found.")
        
    # 2. Load Scaler
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("Scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    else:
        print(f"Warning: Scaler file '{SCALER_PATH}' not found.")
        
    # 3. Seed the Sliding Window History Buffer (requires 30 past observations to compute rolling/lag features)
    with buffer_lock:
        if os.path.exists(CLEANED_DATA_PATH):
            try:
                # Load columns representing raw telemetry metrics
                raw_columns = ["timestamp"] + BASE_FEATURES
                df_clean = pd.read_csv(CLEANED_DATA_PATH)
                available_cols = [c for c in raw_columns if c in df_clean.columns]
                # Seed with the last 30 observations (representing 150 minutes of history at 5-minute intervals)
                history_buffer = df_clean[available_cols].tail(30).reset_index(drop=True)
                print(f"History buffer successfully seeded with {len(history_buffer)} records from cleaned workload history.")
            except Exception as e:
                print(f"Error seeding history buffer: {e}")
                history_buffer = pd.DataFrame(columns=["timestamp"] + BASE_FEATURES)
        else:
            print(f"Warning: Cleaned workload CSV '{CLEANED_DATA_PATH}' not found. History buffer initialized empty.")
            history_buffer = pd.DataFrame(columns=["timestamp"] + BASE_FEATURES)

class PredictionInput(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage (0-100)", json_schema_extra={"example": 68.0})
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage (0-100)", json_schema_extra={"example": 72.0})
    network_in: float = Field(..., ge=0, description="Network In throughput (Mbps)", json_schema_extra={"example": 100.0})
    network_out: float = Field(..., ge=0, description="Network Out throughput (Mbps)", json_schema_extra={"example": 250.0})
    network_traffic: float = Field(None, ge=0, description="Total traffic (Mbps). If omitted, calculated as network_in + network_out.", json_schema_extra={"example": 350.0})
    disk_read: float = Field(..., ge=0, description="Disk Read IOPS", json_schema_extra={"example": 80.0})
    disk_write: float = Field(..., ge=0, description="Disk Write IOPS", json_schema_extra={"example": 40.0})
    active_users: int = Field(..., ge=0, description="Number of active user sessions", json_schema_extra={"example": 250})
    request_rate: float = Field(..., ge=0, description="Requests per second", json_schema_extra={"example": 625.0})
    response_time: float = Field(..., ge=0, description="Average response latency (ms)", json_schema_extra={"example": 185.0})
    error_rate: float = Field(..., ge=0, le=100, description="Error rate percentage (0-100)", json_schema_extra={"example": 0.05})
    current_servers: int = Field(..., ge=1, description="Current number of active servers", json_schema_extra={"example": 5})
    server_cost: float = Field(..., ge=0, description="Current server hosting cost ($/hour)", json_schema_extra={"example": 0.60})

class PredictionOutput(BaseModel):
    current_cpu: float = Field(..., description="The current CPU usage.")
    predicted_cpu_5min: float = Field(..., description="Forecasted CPU usage in 5 minutes.")
    predicted_cpu_10min: float = Field(..., description="Forecasted CPU usage in 10 minutes.")
    predicted_cpu_15min: float = Field(..., description="Forecasted CPU usage in 15 minutes.")
    
    current_servers: int = Field(..., description="The input number of current servers.")
    predicted_required_servers: int = Field(..., description="Proactive server capacity required in 15 minutes.")
    
    scaling_action: str = Field(..., description="Proactive scaling action: SCALE UP, SCALE DOWN, or NO ACTION NEEDED.")
    reasoning: str = Field(..., description="Detailed prediction summary and recommendation.")
    
    forecasts: dict = Field(..., description="Full multi-horizon forecasts mapping metrics to time intervals.")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI Cloud Resource Optimization API v4.0 (Predictive Autoscaling)",
        "docs_url": "/docs",
        "health_check_url": "/health",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "history_buffer_size": len(history_buffer) if history_buffer is not None else 0
    }

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Capacity model is not loaded. Please train models.")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler is not loaded. Please run pipeline.")
    return {
        "status": "healthy",
        "model_file": MODEL_PATH,
        "model_type": type(model).__name__,
        "scaler_file": SCALER_PATH,
        "history_buffer_size": len(history_buffer) if history_buffer is not None else 0
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    global model, scaler, history_buffer
    
    # 1. Asset check
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="ML model or preprocessor assets not loaded. Run pipeline and training scripts first."
        )
        
    # 2. Extract input record and compute traffic
    input_dict = payload.dict()
    if input_dict.get("network_traffic") is None:
        input_dict["network_traffic"] = input_dict["network_in"] + input_dict["network_out"]
        
    # Standardize timestamp for calculations
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_dict["timestamp"] = now_str
    
    try:
        new_row_df = pd.DataFrame([input_dict])
        
        # 3. Update history buffer and calculate time-series forecast (Stage 1)
        with buffer_lock:
            # Append new record to history
            combined_df = pd.concat([history_buffer, new_row_df], ignore_index=True)
            context_df = combined_df.tail(30).reset_index(drop=True)
            
            # Forecast next workloads (outputs dictionary: 5min, 10min, 15min)
            forecasts = forecast_next_workloads(context_df)
            
            # Slide history buffer forward
            raw_cols = ["timestamp"] + BASE_FEATURES
            history_buffer = context_df[raw_cols].tail(30).reset_index(drop=True)
            
        # 4. Proactive Server Capacity Projection (Stage 2)
        # We construct a projected 15-minute ahead telemetry row
        # We use the 15-minute forecasted values for: CPU, Memory, Traffic, Users, Request Rate, Latency
        # For the rest (ingress/egress traffic split, disk IOPS, errors, cost), we keep them constant
        proj_15 = input_dict.copy()
        
        # Override with Stage 1 forecast results
        fc_15 = forecasts["15min"]
        proj_15["cpu_usage"] = fc_15["cpu_usage"]
        proj_15["memory_usage"] = fc_15["memory_usage"]
        proj_15["network_traffic"] = fc_15["network_traffic"]
        proj_15["active_users"] = int(np.round(fc_15["active_users"]))
        proj_15["request_rate"] = fc_15["request_rate"]
        proj_15["response_time"] = fc_15["response_time"]
        
        # Adjust timestamp forward by 15 minutes
        future_dt = datetime.strptime(now_str, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=15)
        proj_15["timestamp"] = future_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # Append projected future row to context
        proj_df = pd.DataFrame([proj_15])
        projected_context = pd.concat([context_df, proj_df], ignore_index=True).tail(31).reset_index(drop=True)
        
        # Run features transformation on projected context
        scaled_projected_input = preprocess_single_record(projected_context, scaler)
        
        # Generate Proactive Server Capacity Prediction
        raw_pred = model.predict(scaled_projected_input)[0]
        predicted_required_servers = int(np.round(raw_pred))
        predicted_required_servers = max(1, predicted_required_servers)
        
        # 5. Proactive Scaling Recommendation
        curr_servers = payload.current_servers
        if predicted_required_servers > curr_servers:
            action = "SCALE UP"
            reasoning = (
                f"Workload forecasting detects incoming spike. "
                f"Predicted CPU: {fc_15['cpu_usage']:.1f}% in 15 mins. "
                f"Proactive Recommendation: SCALE UP BEFORE WORKLOAD SPIKE."
            )
        elif predicted_required_servers < curr_servers:
            action = "SCALE DOWN"
            reasoning = (
                f"Workload forecasting detects load reduction. "
                f"Predicted CPU: {fc_15['cpu_usage']:.1f}% in 15 mins. "
                f"Proactive Recommendation: SCALE DOWN to save hosting costs."
            )
        else:
            action = "NO ACTION NEEDED"
            reasoning = "System resources are projected to remain fully optimized over the 15-minute horizon."
            
        return PredictionOutput(
            current_cpu=payload.cpu_usage,
            predicted_cpu_5min=round(forecasts["5min"]["cpu_usage"], 2),
            predicted_cpu_10min=round(forecasts["10min"]["cpu_usage"], 2),
            predicted_cpu_15min=round(forecasts["15min"]["cpu_usage"], 2),
            current_servers=curr_servers,
            predicted_required_servers=predicted_required_servers,
            scaling_action=action,
            reasoning=reasoning,
            forecasts=forecasts
        )
    except DatasetValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
def get_features():
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        features_list_path = "artifacts/features_list.pkl"
        if os.path.exists(features_list_path):
            features = joblib.load(features_list_path)
        else:
            features = BASE_FEATURES
            
        importance = model.feature_importances_
        importance_dict = {feat: float(imp) for feat, imp in zip(features, importance)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "features": features,
            "importances": sorted_importance
        }
    except AttributeError:
        raise HTTPException(status_code=400, detail="Feature importances not available for this model type.")
