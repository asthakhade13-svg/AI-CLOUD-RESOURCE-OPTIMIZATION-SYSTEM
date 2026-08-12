from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime
from src.pipeline import BASE_FEATURES, preprocess_single_record, DatasetValidationError

app = FastAPI(
    title="AI Cloud Resource Optimization API",
    description="A FastAPI backend leveraging advanced feature engineering (lags, rolling stats, cyclical time) for autoscaling predictions.",
    version="3.0.0"
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
    
    # 1. Load ML Model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully on startup.")
        except Exception as e:
            print(f"Error loading model on startup: {e}")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found. Please train the model first.")
        
    # 2. Load Scaler
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("Scaler loaded successfully on startup.")
        except Exception as e:
            print(f"Error loading scaler on startup: {e}")
    else:
        print(f"Warning: Scaler file '{SCALER_PATH}' not found. Please run the preprocessing pipeline first.")
        
    # 3. Seed the Sliding Window History Buffer (requires 30 past observations to compute rolling/lag features)
    with buffer_lock:
        if os.path.exists(CLEANED_DATA_PATH):
            try:
                # Load columns representing raw telemetry metrics
                raw_columns = ["timestamp"] + BASE_FEATURES
                df_clean = pd.read_csv(CLEANED_DATA_PATH)
                
                # If required_servers target column is in the CSV but not needed for predict features,
                # we drop it and filter to the expected columns
                available_cols = [c for c in raw_columns if c in df_clean.columns]
                history_buffer = df_clean[available_cols].tail(30).reset_index(drop=True)
                print(f"History buffer successfully seeded with {len(history_buffer)} records from cleaned workload history.")
            except Exception as e:
                print(f"Error seeding history buffer: {e}")
                # Initialize empty structure
                history_buffer = pd.DataFrame(columns=["timestamp"] + BASE_FEATURES)
        else:
            print(f"Warning: Cleaned workload CSV '{CLEANED_DATA_PATH}' not found. History buffer initialized empty.")
            history_buffer = pd.DataFrame(columns=["timestamp"] + BASE_FEATURES)

class PredictionInput(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage (0-100)", json_schema_extra={"example": 75.0})
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage (0-100)", json_schema_extra={"example": 80.0})
    network_in: float = Field(..., ge=0, description="Network In throughput (Mbps)", json_schema_extra={"example": 100.0})
    network_out: float = Field(..., ge=0, description="Network Out throughput (Mbps)", json_schema_extra={"example": 250.0})
    network_traffic: float = Field(None, ge=0, description="Total traffic (Mbps). If omitted, calculated as network_in + network_out.", json_schema_extra={"example": 350.0})
    disk_read: float = Field(..., ge=0, description="Disk Read IOPS", json_schema_extra={"example": 80.0})
    disk_write: float = Field(..., ge=0, description="Disk Write IOPS", json_schema_extra={"example": 40.0})
    active_users: int = Field(..., ge=0, description="Number of active user sessions", json_schema_extra={"example": 250})
    request_rate: float = Field(..., ge=0, description="Requests per second", json_schema_extra={"example": 625.0})
    response_time: float = Field(..., ge=0, description="Average response latency (ms)", json_schema_extra={"example": 185.0})
    error_rate: float = Field(..., ge=0, le=100, description="Error rate percentage (0-100)", json_schema_extra={"example": 0.05})
    current_servers: int = Field(..., ge=1, description="Current number of active servers", json_schema_extra={"example": 4})
    server_cost: float = Field(..., ge=0, description="Current server hosting cost ($/hour)", json_schema_extra={"example": 0.487})

class PredictionOutput(BaseModel):
    predicted_required_servers: int = Field(..., description="The recommended number of servers required.")
    raw_prediction: float = Field(..., description="The raw continuous output from the regression model.")
    current_servers: int = Field(..., description="The input number of current servers.")
    scaling_action: str = Field(..., description="Recommended scaling action: SCALE UP, SCALE DOWN, or NO ACTION NEEDED.")
    reasoning: str = Field(..., description="Text rationale explaining the recommendation.")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI Cloud Resource Optimization API v3.0 (Stateful Feature Engineering)",
        "docs_url": "/docs",
        "health_check_url": "/health",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "history_buffer_size": len(history_buffer) if history_buffer is not None else 0
    }

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train the model.")
    if scaler is None:
        raise HTTPException(status_code=503, detail="Scaler is not loaded. Please run pipeline first.")
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
    
    # 1. Check assets
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="ML model or preprocessor assets not loaded. Run pipeline and training scripts first."
        )
        
    # 2. Extract inputs and compute network_traffic if missing
    input_dict = payload.dict()
    if input_dict.get("network_traffic") is None:
        input_dict["network_traffic"] = input_dict["network_in"] + input_dict["network_out"]
        
    # Attach current timestamp for cyclical extraction
    input_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Create single row DataFrame
        new_row_df = pd.DataFrame([input_dict])
        
        # 3. Thread-safe buffer update and stateful feature engineering
        with buffer_lock:
            # Append new record to historical dataframe
            combined_df = pd.concat([history_buffer, new_row_df], ignore_index=True)
            # Keep sliding window context: last 30 hours of history + 1 new row = 31 records maximum
            context_df = combined_df.tail(31).reset_index(drop=True)
            
            # Preprocess the entire context window, which computes lags/moving stats 
            # and returns the scaled feature vector of the latest row
            scaled_input = preprocess_single_record(context_df, scaler)
            
            # Update history buffer (slide forward, keeping only base features and timestamp)
            raw_cols = ["timestamp"] + BASE_FEATURES
            history_buffer = context_df[raw_cols].tail(30).reset_index(drop=True)
            
        # 4. Generate prediction using the best model trained on engineered features
        raw_pred = model.predict(scaled_input)[0]
        required_servers = int(np.round(raw_pred))
        required_servers = max(1, required_servers)  # Ensure at least 1 server runs
        
        # 5. Determine Scaling Action
        if required_servers > payload.current_servers:
            action = "SCALE UP"
            diff = required_servers - payload.current_servers
            reasoning = f"Current workload trends suggest scaling up to {required_servers} servers (Add {diff} server(s))."
        elif required_servers < payload.current_servers:
            action = "SCALE DOWN"
            diff = payload.current_servers - required_servers
            reasoning = f"Workload demand has stabilized. Scaling down to {required_servers} servers (Remove {diff} server(s)) to minimize costs."
        else:
            action = "NO ACTION NEEDED"
            reasoning = "System resources are perfectly balanced and optimized for the current demand trends."
            
        return PredictionOutput(
            predicted_required_servers=required_servers,
            raw_prediction=float(raw_pred),
            current_servers=payload.current_servers,
            scaling_action=action,
            reasoning=reasoning
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
        # Load feature names dynamically from saved list
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
