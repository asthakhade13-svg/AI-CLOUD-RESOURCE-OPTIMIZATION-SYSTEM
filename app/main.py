from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.utils.logging import logger
from app.models.manager import model_manager
from app.schemas import (
    TelemetryPayload, PredictRequest, PredictResponse,
    AutoscaleRequest, AutoscaleOutput,
    OptimizeRequest, OptimizeOutput,
    AnomalyOutput, ForecastOutput
)
from app.services.forecasting import forecast_workloads
from app.services.capacity import evaluate_capacity, estimate_uncertainty
from app.services.optimizer import optimize_cost
from app.services.controller import get_autoscaler
from app.services.anomaly import detect_anomaly
from app.services.explainability import explain_prediction
from app.utils.prometheus import update_prometheus_metrics

from src.pipeline import BASE_FEATURES, preprocess_single_record, DatasetValidationError
from src.sla import evaluate_sla

app = FastAPI(
    title=settings.APP_TITLE,
    description="Production-grade REST API backend for Cloud Resource Optimization with forecasting, predictive capacity sizing, cost optimizations, SLA checks, anomaly overrides, and SHAP explainability.",
    version=settings.APP_VERSION
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global metrics register
API_METRICS: Dict[str, Any] = {
    "total_predict_requests": 0,
    "total_forecast_requests": 0,
    "total_autoscale_requests": 0,
    "total_anomaly_requests": 0,
    "total_optimize_requests": 0,
    "start_time": datetime.now().isoformat(),
    "errors_logged": 0
}

# In-memory sliding window history buffer for time-series forecasting (lags generation)
history_buffer = None
buffer_lock = threading.Lock()

# Custom error handler for validation error
@app.exception_handler(DatasetValidationError)
async def validation_exception_handler(request: Request, exc: DatasetValidationError):
    API_METRICS["errors_logged"] += 1
    logger.error(f"Validation error occurred: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": f"Dataset Validation Error: {exc.message}"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    API_METRICS["errors_logged"] += 1
    logger.error(f"Global server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal Server Error: {str(exc)}"}
    )

@app.on_event("startup")
def startup_event():
    global history_buffer
    logger.info("Starting up FastAPI application...")
    
    # 1. Load ML assets once at startup
    model_manager.load_all_assets()
    
    # 2. Seed history buffer from cleaned workload CSV
    with buffer_lock:
        if os.path.exists(settings.CLEANED_DATA_PATH):
            try:
                df_clean = pd.read_csv(settings.CLEANED_DATA_PATH)
                raw_columns = ["timestamp"] + BASE_FEATURES
                available_cols = [c for c in raw_columns if c in df_clean.columns]
                history_buffer = df_clean[available_cols].tail(30).reset_index(drop=True)
                logger.info(f"Seeded history buffer with {len(history_buffer)} historical workloads.")
            except Exception as e:
                logger.error(f"Error seeding history buffer: {e}")
                history_buffer = pd.DataFrame(columns=["timestamp"] + BASE_FEATURES)
        else:
            logger.warning(f"Cleaned dataset CSV '{settings.CLEANED_DATA_PATH}' not found. Seeding empty history.")
            history_buffer = pd.DataFrame(columns=["timestamp"] + BASE_FEATURES)

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Returns application health and model loading metadata."""
    if not model_manager.assets_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning models are not loaded. Run pipeline/training first."
        )
    return {
        "status": "healthy",
        "app_title": settings.APP_TITLE,
        "app_version": settings.APP_VERSION,
        "assets_loaded": model_manager.assets_loaded,
        "history_buffer_size": len(history_buffer) if history_buffer is not None else 0
    }

@app.get("/metrics")
def get_metrics():
    """Exposes backend execution telemetry and cloud metrics in Prometheus format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict_capacity(payload: PredictRequest):
    """
    Executes full predictive scaling decision. Loads Stage 1 models to forecast workloads,
    applies Stage 2 Random Forest capacity planner, cost-minimization optimization,
    SLA checks, anomaly alerts, and SHAP explainability.
    """
    global history_buffer
    start_time = time.perf_counter()
    API_METRICS["total_predict_requests"] += 1
    
    if not model_manager.assets_loaded:
        raise HTTPException(status_code=503, detail="ML Models are not loaded.")
        
    # Map input request to dictionary format
    input_dict = payload.dict()
    
    # Construct complete telemetry dictionary for features list
    telemetry = {
        "cpu_usage": payload.cpu_usage,
        "memory_usage": payload.memory_usage,
        "network_in": payload.network_in,
        "network_out": payload.network_out,
        "network_traffic": payload.network_traffic,
        "disk_read": payload.disk_read,
        "disk_write": payload.disk_write,
        "active_users": payload.active_users,
        "request_rate": payload.request_rate,
        "response_time": payload.response_time,
        "error_rate": payload.error_rate,
        "current_servers": payload.current_servers,
        "server_cost": payload.server_cost
    }
    
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    telemetry["timestamp"] = now_str
    
    try:
        new_row_df = pd.DataFrame([telemetry])
        
        # 1. Update history and calculate workloads forecast
        with buffer_lock:
            combined_df = pd.concat([history_buffer, new_row_df], ignore_index=True)
            context_df = combined_df.tail(30).reset_index(drop=True)
            
            # Forecast next workloads (outputs forecasts dict)
            forecasts = forecast_workloads(context_df)
            
            # Update cache
            raw_cols = ["timestamp"] + BASE_FEATURES
            history_buffer = context_df[raw_cols].tail(30).reset_index(drop=True)
            
        # 2. Predictive Server capacity Sizing (15-min projection)
        proj_15 = telemetry.copy()
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
        
        scaled_projected_input = preprocess_single_record(projected_context, model_manager.capacity_scaler)
        raw_pred = model_manager.capacity_model.predict(scaled_projected_input)[0]
        
        # 3. Apply safe capacity boundaries and uncertainty estimations
        capacity = evaluate_capacity(
            prediction=raw_pred,
            current_servers=payload.current_servers,
            min_servers=payload.min_servers,
            max_servers=payload.max_servers,
            safety_margin=payload.safety_margin
        )
        uncertainty = estimate_uncertainty(scaled_projected_input)
        
        # 4. Cost Optimization tradeoffs
        opt_res = optimize_cost(
            predicted_required_servers=capacity["recommended_servers"],
            current_servers=payload.current_servers,
            server_cost_per_hour=payload.server_cost,
            min_servers=payload.min_servers,
            max_servers=payload.max_servers,
            sla_penalty_weight=payload.sla_penalty_weight,
            overprovisioning_weight=payload.overprovisioning_weight
        )
        
        # 5. SLA evaluation checks
        sla_res = evaluate_sla(
            response_time=payload.response_time,
            error_rate=payload.error_rate,
            cpu_usage=payload.cpu_usage,
            memory_usage=payload.memory_usage,
            target_response_time=payload.target_response_time,
            maximum_error_rate=payload.maximum_error_rate,
            minimum_availability=payload.minimum_availability
        )
        
        # 6. Anomaly Checks
        new_row_with_time = new_row_df.copy()
        new_row_with_time["hour"] = latest_ts = pd.to_datetime(now_str).hour
        new_row_with_time["day_of_week"] = pd.to_datetime(now_str).dayofweek
        new_row_with_time["sin_hour"] = np.sin(2 * np.pi * latest_ts / 24.0)
        new_row_with_time["cos_hour"] = np.cos(2 * np.pi * latest_ts / 24.0)
        new_row_with_time["sin_day_of_week"] = np.sin(2 * np.pi * new_row_with_time["day_of_week"] / 7.0)
        new_row_with_time["cos_day_of_week"] = np.cos(2 * np.pi * new_row_with_time["day_of_week"] / 7.0)
        
        anomaly_res = detect_anomaly(new_row_with_time, history_buffer)
        
        # 7. Autoscaler Decisions
        controller = get_autoscaler(current_servers=payload.current_servers)
        controller.min_servers = payload.min_servers
        controller.max_servers = payload.max_servers
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
        
        # 8. SHAP Local Explanations
        xai_res = explain_prediction(scaled_projected_input, decision["recommended_servers"])
        
        # 9. Update Prometheus metrics
        latency = time.perf_counter() - start_time
        prediction_metrics = {
            "predicted_servers": capacity["predicted_servers"],
            "recommended_servers": decision["recommended_servers"],
            "action": decision["action"]
        }
        update_prometheus_metrics(telemetry, prediction_metrics, latency)
        
        return PredictResponse(
            predicted_servers=int(np.round(capacity["predicted_servers"])),
            recommended_servers=decision["recommended_servers"],
            action=decision["action"],
            current_servers=payload.current_servers,
            scaling_action=decision["action"],
            reason=decision["reason"],
            reasoning=decision["reason"],
            cooldown_active=decision["cooldown_active"],
            sla_status=sla_res["status"],
            risk_score=sla_res["risk_score"],
            is_anomaly=anomaly_res["is_anomaly"],
            anomaly_score=anomaly_res["anomaly_score"],
            severity=anomaly_res["severity"],
            affected_metrics=anomaly_res["affected_metrics"],
            recommendation=anomaly_res["recommendation"],
            shap_explanation=xai_res["shap_explanation"],
            shap_contributions=xai_res["category_contributions"],
            hourly_cost=opt_res["hourly_cost"],
            estimated_daily_cost=opt_res["estimated_daily_cost"],
            estimated_monthly_cost=opt_res["estimated_monthly_cost"],
            estimated_savings=opt_res["estimated_savings_daily"],
            forecasts=forecasts
        )
    except Exception as e:
        logger.error(f"Error executing prediction endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastOutput)
def predict_forecasts(payload: TelemetryPayload):
    """Generates workload forecasts for the next 5, 10, and 15 minutes."""
    API_METRICS["total_forecast_requests"] += 1
    
    input_dict = payload.dict()
    if input_dict.get("network_traffic") is None:
        input_dict["network_traffic"] = input_dict["network_in"] + input_dict["network_out"]
        
    input_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        new_row = pd.DataFrame([input_dict])
        with buffer_lock:
            combined = pd.concat([history_buffer, new_row], ignore_index=True)
            context = combined.tail(30).reset_index(drop=True)
            forecasts = forecast_workloads(context)
            
        return ForecastOutput(forecasts=forecasts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting calculation failed: {str(e)}")

@app.post("/autoscale", response_model=AutoscaleOutput)
def evaluate_autoscaler(payload: AutoscaleRequest):
    """Evaluates stateful autoscaling transitions (anti-thrashing controller checks)."""
    API_METRICS["total_autoscale_requests"] += 1
    
    try:
        controller = get_autoscaler(current_servers=payload.current_servers)
        
        # Configure dynamically
        controller.min_servers = payload.min_servers
        controller.max_servers = payload.max_servers
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
            predicted_servers=payload.predicted_servers,
            recommended_servers=payload.recommended_servers,
            sla_status=payload.sla_status,
            anomaly_severity=payload.anomaly_severity
        )
        
        return AutoscaleOutput(
            current_servers=payload.current_servers,
            predicted_servers=payload.predicted_servers,
            recommended_servers=decision["recommended_servers"],
            action=decision["action"],
            reason=decision["reason"],
            cooldown_active=decision["cooldown_active"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/anomaly", response_model=AnomalyOutput)
def evaluate_anomaly(payload: TelemetryPayload):
    """Evaluates telemetry workload state for active anomalies."""
    API_METRICS["total_anomaly_requests"] += 1
    
    input_dict = payload.dict()
    if input_dict.get("network_traffic") is None:
        input_dict["network_traffic"] = input_dict["network_in"] + input_dict["network_out"]
        
    now = datetime.now()
    input_dict["timestamp"] = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add time features
    new_row = pd.DataFrame([input_dict])
    new_row["hour"] = now.hour
    new_row["day_of_week"] = now.weekday()
    new_row["sin_hour"] = np.sin(2 * np.pi * now.hour / 24.0)
    new_row["cos_hour"] = np.cos(2 * np.pi * now.hour / 24.0)
    new_row["sin_day_of_week"] = np.sin(2 * np.pi * now.weekday() / 7.0)
    new_row["cos_day_of_week"] = np.cos(2 * np.pi * now.weekday() / 7.0)
    
    try:
        res = detect_anomaly(new_row, history_buffer)
        return AnomalyOutput(
            is_anomaly=res["is_anomaly"],
            anomaly_score=res["anomaly_score"],
            severity=res["severity"],
            affected_metrics=res["affected_metrics"],
            recommendation=res["recommendation"],
            reason=res["reason"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize", response_model=OptimizeOutput)
def evaluate_optimization(payload: OptimizeRequest):
    """Evaluates capacity sizing optimizations based on objective cost function."""
    API_METRICS["total_optimize_requests"] += 1
    
    try:
        res = optimize_cost(
            predicted_required_servers=payload.predicted_required_servers,
            current_servers=payload.current_servers,
            server_cost_per_hour=payload.server_cost_per_hour,
            min_servers=payload.min_servers,
            max_servers=payload.max_servers,
            sla_penalty_weight=payload.sla_penalty_weight,
            overprovisioning_weight=payload.overprovisioning_weight
        )
        return OptimizeOutput(
            recommended_servers=res["recommended_servers"],
            hourly_cost=res["hourly_cost"],
            estimated_daily_cost=res["estimated_daily_cost"],
            estimated_monthly_cost=res["estimated_monthly_cost"],
            estimated_savings=res["estimated_savings_daily"],
            sla_status="HEALTHY" if res["recommended_servers"] >= payload.predicted_required_servers else "AT_RISK",
            optimization_reason=res["optimization_reason"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
