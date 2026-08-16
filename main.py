from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime, timedelta
from typing import List
import shap
from src.pipeline import BASE_FEATURES, preprocess_single_record, DatasetValidationError
from src.forecasting import forecast_next_workloads, FORECAST_METRICS
from src.capacity import calculate_required_servers, estimate_prediction_uncertainty
from src.controller import AutoscalingController
from src.optimizer import optimize_capacity_cost
from src.sla import evaluate_sla
from src.anomaly import detect_anomaly_record
from src.explainability import explain_prediction_shap

app = FastAPI(
    title="AI Cloud Resource Optimization API",
    description="A FastAPI backend leveraging two-stage predictive autoscaling, risk-managed capacity planning, cost optimization, performance SLA evaluation, anomaly detection, and SHAP explainability.",
    version="10.0.0"
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
controller = None
shap_explainer = None
features_list = None

# Thread-safe sliding window buffer to cache the historical workload state
history_buffer = None
buffer_lock = threading.Lock()

@app.on_event("startup")
def load_assets_and_seed_buffer():
    global model, scaler, history_buffer, controller, shap_explainer, features_list
    
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
        
    # 3. Seed the Sliding Window History Buffer
    with buffer_lock:
        if os.path.exists(CLEANED_DATA_PATH):
            try:
                raw_columns = ["timestamp"] + BASE_FEATURES
                df_clean = pd.read_csv(CLEANED_DATA_PATH)
                available_cols = [c for c in raw_columns if c in df_clean.columns]
                history_buffer = df_clean[available_cols].tail(30).reset_index(drop=True)
                print(f"History buffer seeded with {len(history_buffer)} records.")
            except Exception as e:
                print(f"Error seeding history buffer: {e}")
                history_buffer = pd.DataFrame(columns=["timestamp"] + BASE_FEATURES)
        else:
            print(f"Warning: Cleaned workload CSV '{CLEANED_DATA_PATH}' not found. History buffer initialized empty.")
            history_buffer = pd.DataFrame(columns=["timestamp"] + BASE_FEATURES)

    # 4. Initialize Stateful Autoscaling Controller
    controller = AutoscalingController(current_servers=5)
    print("Stateful Autoscaling Controller initialized.")
    
    # 5. Initialize SHAP TreeExplainer
    if model is not None:
        try:
            shap_explainer = shap.TreeExplainer(model)
            print("SHAP TreeExplainer successfully initialized.")
        except Exception as e:
            print(f"Error initializing SHAP TreeExplainer: {e}")
            
    # 6. Load feature names
    features_list_path = "artifacts/features_list.pkl"
    if os.path.exists(features_list_path):
        features_list = joblib.load(features_list_path)

class PredictionInput(BaseModel):
    # Telemetry
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
    server_cost: float = Field(..., ge=0, description="Current server hosting cost per hour ($/hour)", json_schema_extra={"example": 0.60})
    
    # Capacity Limits and Safety Parameters
    min_servers: int = Field(1, ge=1, description="Minimum server limit", json_schema_extra={"example": 1})
    max_servers: int = Field(20, ge=1, description="Maximum server limit", json_schema_extra={"example": 20})
    safety_margin: float = Field(0.10, ge=0.0, le=1.0, description="Autoscaling safety margin multiplier (e.g. 0.10 = 10%)", json_schema_extra={"example": 0.10})
    
    # Anti-Thrashing Controller Configurations
    scale_up_cpu_threshold: float = Field(80.0, ge=0.0, le=100.0, description="CPU usage threshold to allow scale-up")
    scale_down_cpu_threshold: float = Field(35.0, ge=0.0, le=100.0, description="CPU usage threshold to allow scale-down")
    cooldown_periods: int = Field(3, ge=0, description="Cooldown periods (ticks) blocking scaling actions")
    scale_up_confirmations: int = Field(3, ge=1, description="Consecutive ticks required to confirm scale-up")
    scale_down_confirmations: int = Field(6, ge=1, description="Consecutive ticks required to confirm scale-down")
    max_scale_up_step: int = Field(2, ge=1, description="Maximum servers added in a single scaling step")
    max_scale_down_step: int = Field(1, ge=1, description="Maximum servers removed in a single scaling step")
    
    # Cost Optimizer Weights
    sla_penalty_weight: float = Field(5.0, ge=0.0, description="SLA violation penalty weight multiplier")
    overprovisioning_weight: float = Field(0.5, ge=0.0, description="Overprovisioning budget wastage weight multiplier")
    
    # SLA Targets
    target_response_time: float = Field(200.0, ge=0.0, description="Target SLA response time (ms)")
    maximum_error_rate: float = Field(1.0, ge=0.0, le=100.0, description="Maximum SLA error rate (%)")
    minimum_availability: float = Field(99.0, ge=0.0, le=100.0, description="Minimum SLA availability (%)")

class PredictionOutput(BaseModel):
    recommended_servers: int = Field(..., description="The final recommended server capacity after all optimizations & safety locks.")
    current_servers: int = Field(..., description="The server count before evaluation.")
    predicted_servers: float = Field(..., description="Raw continuous statistical prediction from ML model.")
    
    # Cost metrics
    hourly_cost: float = Field(..., description="Hourly hosting cost for the recommended configuration.")
    estimated_daily_cost: float = Field(..., description="Daily hosting cost projection.")
    estimated_monthly_cost: float = Field(..., description="Monthly hosting cost projection.")
    estimated_savings: float = Field(..., description="Daily estimated cost savings compared to the current server count.")
    
    # SLA metrics
    sla_status: str = Field(..., description="SLA compliance status: VIOLATED, AT_RISK, or HEALTHY.")
    risk_score: float = Field(..., description="Standardized SLA breach risk score [0.0, 1.0].")
    optimization_reason: str = Field(..., description="Detailed description of the cost-performance trade-offs.")
    
    # Anomaly metrics
    is_anomaly: bool = Field(..., description="Flag indicating if the current workload pattern is anomalous.")
    anomaly_score: float = Field(..., description="Normalized anomaly scoring metric [0.0, 1.0].")
    severity: str = Field(..., description="Severity of detected anomaly (LOW/MEDIUM/HIGH/CRITICAL).")
    anomaly_severity: str = Field(..., description="Alias for severity to maintain backwards compatibility.")
    affected_metrics: List[str] = Field(..., description="List of metrics showing significant deviations.")
    recommendation: str = Field(..., description="Scaling action advice based on anomaly detection output.")
    
    # SHAP Explainability outputs
    shap_explanation: str = Field(..., description="Human-readable SHAP explanation outlining feature contributions.")
    shap_contributions: dict = Field(..., description="Shapley feature value contributions grouped by core categories.")
    
    # Controller outputs
    action: str = Field(..., description="The controller action: SCALE_UP, SCALE_DOWN, or NO_ACTION.")
    scaling_action: str = Field(..., description="Alias for action to maintain backwards compatibility.")
    reason: str = Field(..., description="Controller reason detailing step limits or cooldowns.")
    reasoning: str = Field(..., description="Alias for reason to maintain backwards compatibility.")
    cooldown_active: bool = Field(..., description="Flag indicating if the controller is in cooldown state.")
    
    # Telemetry and metadata for diagnostic visibility
    current_cpu: float = Field(..., description="The current CPU usage.")
    predicted_cpu_5min: float = Field(..., description="Forecasted CPU usage in 5 minutes.")
    predicted_cpu_10min: float = Field(..., description="Forecasted CPU usage in 10 minutes.")
    predicted_cpu_15min: float = Field(..., description="Forecasted CPU usage in 15 minutes.")
    
    safety_margin: float = Field(..., description="The safety margin percentage multiplier applied.")
    safety_buffer: float = Field(..., description="The fractional server count added as a safety buffer.")
    
    prediction_uncertainty_std: float = Field(..., description="Standard deviation of individual estimator predictions (RF uncertainty).")
    confidence_interval_lower: float = Field(..., description="95% Confidence Interval lower bound.")
    confidence_interval_upper: float = Field(..., description="95% Confidence Interval upper bound.")
    
    forecasts: dict = Field(..., description="Full multi-horizon forecasts mapping metrics to time intervals.")

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the AI Cloud Resource Optimization API v10.0 (Explainable AI - SHAP)",
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
    global model, scaler, history_buffer, controller, shap_explainer, features_list
    
    # 1. Asset check
    if model is None or scaler is None or controller is None or shap_explainer is None or features_list is None:
        raise HTTPException(
            status_code=503,
            detail="Autoscaling system dependencies not fully loaded."
        )
        
    # 2. Extract input record and compute traffic
    input_dict = payload.dict()
    # Filter keys to exclude Pydantic-only parameters when updating the history DataFrame
    exclude_keys = [
        "min_servers", "max_servers", "safety_margin", 
        "scale_up_cpu_threshold", "scale_down_cpu_threshold", "cooldown_periods",
        "scale_up_confirmations", "scale_down_confirmations", "max_scale_up_step", "max_scale_down_step",
        "sla_penalty_weight", "overprovisioning_weight",
        "target_response_time", "maximum_error_rate", "minimum_availability"
    ]
    base_input_dict = {k: v for k, v in input_dict.items() if k not in exclude_keys}
    
    if base_input_dict.get("network_traffic") is None:
        base_input_dict["network_traffic"] = base_input_dict["network_in"] + base_input_dict["network_out"]
        
    # Standardize timestamp
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base_input_dict["timestamp"] = now_str
    
    try:
        new_row_df = pd.DataFrame([base_input_dict])
        
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
        # Construct projected 15-minute ahead telemetry row
        proj_15 = base_input_dict.copy()
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
        
        # Generate raw ML prediction
        raw_pred = model.predict(scaled_projected_input)[0]
        
        # 5. Safe Capacity Calculation & Uncertainty estimation
        capacity = calculate_required_servers(
            prediction=raw_pred,
            current_servers=payload.current_servers,
            min_servers=payload.min_servers,
            max_servers=payload.max_servers,
            safety_margin=payload.safety_margin
        )
        
        uncertainty = estimate_prediction_uncertainty(model, scaled_projected_input)
        
        # 6. Cost-Aware Optimization Engine
        opt_res = optimize_capacity_cost(
            predicted_required_servers=capacity["recommended_servers"],
            current_servers=payload.current_servers,
            server_cost_per_hour=payload.server_cost,
            min_servers=payload.min_servers,
            max_servers=payload.max_servers,
            sla_penalty_weight=payload.sla_penalty_weight,
            overprovisioning_weight=payload.overprovisioning_weight
        )
        
        # 7. Performance & SLA Evaluation Module
        sla_res = evaluate_sla(
            response_time=payload.response_time,
            error_rate=payload.error_rate,
            cpu_usage=payload.cpu_usage,
            memory_usage=payload.memory_usage,
            target_response_time=payload.target_response_time,
            maximum_error_rate=payload.maximum_error_rate,
            minimum_availability=payload.minimum_availability
        )
        
        # 8. Run AI Anomaly Detection (Isolation Forest)
        latest_ts = pd.to_datetime(new_row_df.iloc[0]["timestamp"])
        new_row_with_time = new_row_df.copy()
        new_row_with_time["hour"] = latest_ts.hour
        new_row_with_time["day_of_week"] = latest_ts.dayofweek
        new_row_with_time["sin_hour"] = np.sin(2 * np.pi * latest_ts.hour / 24.0)
        new_row_with_time["cos_hour"] = np.cos(2 * np.pi * latest_ts.hour / 24.0)
        new_row_with_time["sin_day_of_week"] = np.sin(2 * np.pi * latest_ts.dayofweek / 7.0)
        new_row_with_time["cos_day_of_week"] = np.cos(2 * np.pi * latest_ts.dayofweek / 7.0)
        
        anomaly_res = detect_anomaly_record(new_row_with_time, history_buffer)
        
        # 9. Apply Stateful Anti-Thrashing Autoscaling Controller
        controller.min_servers = max(1, payload.min_servers)
        controller.max_servers = max(controller.min_servers, payload.max_servers)
        controller.scale_up_cpu_threshold = payload.scale_up_cpu_threshold
        controller.scale_down_cpu_threshold = payload.scale_down_cpu_threshold
        controller.cooldown_periods = payload.cooldown_periods
        controller.scale_up_confirmations = payload.scale_up_confirmations
        controller.scale_down_confirmations = payload.scale_down_confirmations
        controller.max_scale_up_step = payload.max_scale_up_step
        controller.max_scale_down_step = payload.max_scale_down_step
        
        controller.current_server_count = payload.current_servers
        
        decision = controller.make_scaling_decision(
            cpu_usage=payload.cpu_usage,
            predicted_servers=capacity["predicted_servers"],
            recommended_servers=opt_res["recommended_servers"],
            sla_status=sla_res["status"],
            anomaly_severity=anomaly_res["severity"]
        )
        
        # 10. Run SHAP Explainability Engine
        xai_res = explain_prediction_shap(
            explainer=shap_explainer,
            X_scaled_record=scaled_projected_input,
            feature_names=features_list,
            recommended_servers=decision["recommended_servers"]
        )
        
        # Override optimization reason if SLA issues or anomalies were dominant
        reasoning = opt_res["optimization_reason"]
        if anomaly_res["is_anomaly"]:
            reasoning = f"Anomaly Alert: {anomaly_res['severity']} ({anomaly_res['reason']}). " + reasoning
        elif sla_res["status"] != "HEALTHY":
            reasoning = f"SLA State: {sla_res['status']} ({sla_res['reason']}). " + reasoning
            
        return PredictionOutput(
            recommended_servers=decision["recommended_servers"],
            current_servers=payload.current_servers,
            predicted_servers=capacity["predicted_servers"],
            hourly_cost=opt_res["hourly_cost"],
            estimated_daily_cost=opt_res["estimated_daily_cost"],
            estimated_monthly_cost=opt_res["estimated_monthly_cost"],
            estimated_savings=opt_res["estimated_savings_daily"],
            sla_status=sla_res["status"],
            risk_score=sla_res["risk_score"],
            optimization_reason=reasoning,
            is_anomaly=anomaly_res["is_anomaly"],
            anomaly_score=anomaly_res["anomaly_score"],
            severity=anomaly_res["severity"],
            anomaly_severity=anomaly_res["severity"],
            affected_metrics=anomaly_res["affected_metrics"],
            recommendation=anomaly_res["recommendation"],
            shap_explanation=xai_res["shap_explanation"],
            shap_contributions=xai_res["category_contributions"],
            action=decision["action"],
            scaling_action=decision["action"],
            reason=decision["reason"],
            reasoning=decision["reason"],
            cooldown_active=decision["cooldown_active"],
            current_cpu=payload.cpu_usage,
            predicted_cpu_5min=round(forecasts["5min"]["cpu_usage"], 2),
            predicted_cpu_10min=round(forecasts["10min"]["cpu_usage"], 2),
            predicted_cpu_15min=round(forecasts["15min"]["cpu_usage"], 2),
            safety_margin=capacity["safety_margin"],
            safety_buffer=capacity["safety_buffer"],
            prediction_uncertainty_std=uncertainty["uncertainty_std"],
            confidence_interval_lower=uncertainty["confidence_interval_lower"],
            confidence_interval_upper=uncertainty["confidence_interval_upper"],
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
