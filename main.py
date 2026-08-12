from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
from src.pipeline import FEATURES, preprocess_single_record, DatasetValidationError

app = FastAPI(
    title="AI Cloud Resource Optimization API",
    description="A FastAPI backend to predict required servers and recommend auto-scaling actions based on 13 system metrics.",
    version="2.0.0"
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

# Global variables to store loaded assets
model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully on startup.")
        except Exception as e:
            print(f"Error loading model on startup: {e}")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found. Please train the model first.")
        
    # Load Scaler
    if os.path.exists(SCALER_PATH):
        try:
            scaler = joblib.load(SCALER_PATH)
            print("Scaler loaded successfully on startup.")
        except Exception as e:
            print(f"Error loading scaler on startup: {e}")
    else:
        print(f"Warning: Scaler file '{SCALER_PATH}' not found. Please run the preprocessing pipeline first.")

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
        "message": "Welcome to the AI Cloud Resource Optimization API v2.0",
        "docs_url": "/docs",
        "health_check_url": "/health",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
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
        "scaler_file": SCALER_PATH
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    global model, scaler
    
    # 1. Lazy-load model and scaler if not loaded during startup
    if model is None or scaler is None:
        load_assets()
        if model is None or scaler is None:
            raise HTTPException(
                status_code=503,
                detail="ML model or preprocessor assets not loaded. Run training script first."
            )
            
    # 2. Extract inputs and compute network_traffic if missing
    input_dict = payload.dict()
    if input_dict.get("network_traffic") is None:
        input_dict["network_traffic"] = input_dict["network_in"] + input_dict["network_out"]
        
    try:
        # 3. Clean and scale using the pipeline helper
        scaled_input = preprocess_single_record(input_dict, scaler)
        
        # 4. Generate prediction
        raw_pred = model.predict(scaled_input)[0]
        required_servers = int(np.round(raw_pred))
        required_servers = max(1, required_servers)  # Ensure at least 1 server runs
        
        # 5. Determine Scaling Action
        if required_servers > payload.current_servers:
            action = "SCALE UP"
            diff = required_servers - payload.current_servers
            reasoning = f"Current load requires {required_servers} servers. Scale up by adding {diff} server(s)."
        elif required_servers < payload.current_servers:
            action = "SCALE DOWN"
            diff = payload.current_servers - required_servers
            reasoning = f"Current load only requires {required_servers} servers. Scale down by removing {diff} server(s) to optimize costs."
        else:
            action = "NO ACTION NEEDED"
            reasoning = "Current server count is perfectly optimized for the system load."
            
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
        importance = model.feature_importances_
        importance_dict = {feat: float(imp) for feat, imp in zip(FEATURES, importance)}
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "features": FEATURES,
            "importances": sorted_importance
        }
    except AttributeError:
        raise HTTPException(status_code=400, detail="Feature importances not available for this model type.")
