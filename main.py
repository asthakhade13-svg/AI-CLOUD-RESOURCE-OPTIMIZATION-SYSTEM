from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(
    title="AI Cloud Resource Optimization API",
    description="A FastAPI backend to predict required servers and recommend auto-scaling actions based on system metrics.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to the serialized model
MODEL_PATH = "cloud_resource_optimization_model.pkl"

# Global variable to store loaded model
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded successfully on startup.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model file '{MODEL_PATH}' not found. Please train the model first.")

class PredictionInput(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage (0-100)", example=75.0)
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage (0-100)", example=80.0)
    network_traffic: float = Field(..., ge=0, description="Network traffic in Mbps", example=350.0)
    active_users: int = Field(..., ge=0, description="Number of active users", example=250)
    current_servers: int = Field(..., ge=1, description="Current number of running servers", example=4)

class PredictionOutput(BaseModel):
    predicted_required_servers: int = Field(..., description="The recommended number of servers required.")
    raw_prediction: float = Field(..., description="The raw continuous output from the Random Forest model.")
    current_servers: int = Field(..., description="The input number of current servers.")
    scaling_action: str = Field(..., description="Recommended scaling action: SCALE UP, SCALE DOWN, or NO ACTION NEEDED.")
    reasoning: str = Field(..., description="Text rationale explaining the recommendation.")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI Cloud Resource Optimization API",
        "docs_url": "/docs",
        "health_check_url": "/health",
        "model_loaded": model is not None
    }

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please train the model.")
    return {
        "status": "healthy",
        "model_file": MODEL_PATH,
        "model_type": type(model).__name__
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    global model
    if model is None:
        # Try loading on demand if it wasn't loaded during startup
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            raise HTTPException(
                status_code=503,
                detail="Model file not found. Ensure cloud_resource_optimization_model.pkl exists in the root directory."
            )
            
    # Prepare input DataFrame (ensuring feature names match training features)
    input_df = pd.DataFrame({
        "cpu_usage": [payload.cpu_usage],
        "memory_usage": [payload.memory_usage],
        "network_traffic": [payload.network_traffic],
        "active_users": [payload.active_users],
        "current_servers": [payload.current_servers]
    })
    
    try:
        # Generate prediction
        raw_pred = model.predict(input_df)[0]
        required_servers = int(np.round(raw_pred))
        required_servers = max(1, required_servers)  # Ensure at least 1 server runs
        
        # Determine Scaling Action
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
def get_features():
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Extract features and their importance
        importance = model.feature_importances_
        features = ["cpu_usage", "memory_usage", "network_traffic", "active_users", "current_servers"]
        
        importance_dict = {feat: float(imp) for feat, imp in zip(features, importance)}
        # Sort by importance descending
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "features": features,
            "importances": sorted_importance
        }
    except AttributeError:
        raise HTTPException(status_code=400, detail="Feature importances not available for this model type.")
