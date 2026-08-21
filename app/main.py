from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import numpy as np
import os
import time
import threading
import psycopg2
import requests
from datetime import datetime
from typing import Dict, Any
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.utils.logging import logger
from app.schemas import (
    TelemetryPayload, PredictRequest, PredictResponse,
    AutoscaleRequest, AutoscaleOutput,
    OptimizeRequest, OptimizeOutput,
    AnomalyOutput, ForecastOutput,
    RLPredictRequest, RLPredictResponse,
    RLEvaluateRequest, RLEvaluateResponse, RLStatusResponse,
    SimulationScenarioRequest, SimulationScenarioResponse,
    SimulationCompareRequest, SimulationCompareResponse,
    MultiObjectiveOptimizeRequest, MultiObjectiveOptimizeResponse
)


from app.services.optimizer import optimize_cost
from app.services.controller import get_autoscaler
from app.utils.prometheus import update_prometheus_metrics
from src.sla import evaluate_sla

# Import Safety Layer for RL Autonomous decisions
from rl.safety import SafetyValidator
from rl.actions import action_to_step, idx_to_action

app = FastAPI(
    title="AI Cloud Resource Optimization API Gateway",
    description="Orchestrator backend connecting Nginx frontend, ML Model service, and PostgreSQL database.",
    version="11.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gateway Metrics Registry
API_METRICS: Dict[str, Any] = {
    "total_predict_requests": 0,
    "total_forecast_requests": 0,
    "total_autoscale_requests": 0,
    "total_anomaly_requests": 0,
    "total_optimize_requests": 0,
    "start_time": datetime.now().isoformat(),
    "errors_logged": 0
}

# Env configs
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "resource_optimization")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8050")

# Reinforcement Learning Mode (SIMULATION, APPROVAL, AUTONOMOUS)
RL_AUTOSCALING_MODE = os.getenv("RL_AUTOSCALING_MODE", "SIMULATION")
gateway_safety_validator = SafetyValidator()


def get_db_connection():
    """Tries connecting to PostgreSQL with retry counts to handle boot sequence delays."""
    max_retries = 1 if DB_HOST in ["127.0.0.1", "localhost"] else 5
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                connect_timeout=1
            )
            return conn
        except Exception:
            if max_retries > 1:
                time.sleep(1)
    return None

@app.on_event("startup")
def startup_event():
    logger.info("Initializing API Gateway service...")
    # Verify DB connectivity on startup
    conn = get_db_connection()
    if conn:
        logger.info("Database connectivity successfully verified.")
        conn.close()
    else:
        logger.warning("Database unreachable on startup. Continuing in degraded mode.")

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serves the interactive cloud orchestrator web console dashboard."""
    paths_to_check = ["index.html", "../index.html", "app/index.html"]
    for path in paths_to_check:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Cloud Resource Optimization Console</h1><p>Dashboard HTML asset not found.</p>")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Returns application health gateway status."""
    return {
        "status": "healthy",
        "app_title": settings.APP_TITLE,
        "app_version": settings.APP_VERSION,
        "db_connected": get_db_connection() is not None
    }

@app.get("/metrics")
def get_metrics():
    """Exposes backend execution telemetry and cloud metrics in Prometheus format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictResponse)
def predict_capacity(payload: PredictRequest):
    """
    Gateway predict route. Proxies inputs to ML Model service for raw prediction slices,
    then executes cost optimization, SLA checks, anomaly controller triggers,
    and logs telemetry to PostgreSQL.
    """
    global API_METRICS
    start_time = time.perf_counter()
    API_METRICS["total_predict_requests"] += 1
    
    # 1. Forward to ML service for raw forecast, capacity, anomalies, and SHAP
    raw_payload = {
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
        "server_cost": payload.server_cost,
        "safety_margin": payload.safety_margin,
        "min_servers": payload.min_servers,
        "max_servers": payload.max_servers
    }
    
    try:
        ml_res = requests.post(f"{ML_SERVICE_URL}/predict_raw", json=raw_payload, timeout=5)
        if ml_res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ML Service returned error: {ml_res.text}")
        ml_data = ml_res.json()
    except Exception as e:
        logger.error(f"Failed to communicate with ML Model Service: {e}")
        raise HTTPException(status_code=502, detail=f"ML Model service unreachable: {str(e)}")
        
    try:
        # 2. Local Cost Optimization trade-offs
        opt_res = optimize_cost(
            predicted_required_servers=ml_data["recommended_servers"],
            current_servers=payload.current_servers,
            server_cost_per_hour=payload.server_cost,
            min_servers=payload.min_servers,
            max_servers=payload.max_servers,
            sla_penalty_weight=payload.sla_penalty_weight,
            overprovisioning_weight=payload.overprovisioning_weight
        )
        
        # 3. Local SLA evaluation checks
        sla_res = evaluate_sla(
            response_time=payload.response_time,
            error_rate=payload.error_rate,
            cpu_usage=payload.cpu_usage,
            memory_usage=payload.memory_usage,
            target_response_time=payload.target_response_time,
            maximum_error_rate=payload.maximum_error_rate,
            minimum_availability=payload.minimum_availability
        )
        
        # 4. Local Stateful Anti-Thrashing Autoscaler decisions
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
            predicted_servers=ml_data["predicted_servers"],
            recommended_servers=opt_res["recommended_servers"],
            sla_status=sla_res["status"],
            anomaly_severity=ml_data["severity"]
        )
        
        # 5. Write to PostgreSQL database
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO telemetry_logs 
                    (cpu_usage, memory_usage, network_traffic, active_users, current_servers, recommended_servers, action, sla_status, anomaly_severity, optimization_reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        payload.cpu_usage,
                        payload.memory_usage,
                        payload.network_traffic if payload.network_traffic is not None else (payload.network_in + payload.network_out),
                        payload.active_users,
                        payload.current_servers,
                        decision["recommended_servers"],
                        decision["action"],
                        sla_res["status"],
                        ml_data["severity"],
                        opt_res["optimization_reason"]
                    )
                )
                conn.commit()
                cursor.close()
                conn.close()
            except Exception as dbe:
                logger.error(f"Error logging telemetry to PostgreSQL: {dbe}")
                if conn:
                    conn.close()
                    
        # 6. Update Prometheus Metrics
        latency = time.perf_counter() - start_time
        telemetry_dict = {
            "cpu_usage": payload.cpu_usage,
            "memory_usage": payload.memory_usage,
            "network_traffic": payload.network_traffic if payload.network_traffic is not None else (payload.network_in + payload.network_out),
            "request_rate": payload.request_rate,
            "response_time": payload.response_time,
            "error_rate": payload.error_rate,
            "active_users": payload.active_users,
            "current_servers": payload.current_servers
        }
        pred_dict = {
            "predicted_servers": ml_data["predicted_servers"],
            "recommended_servers": decision["recommended_servers"],
            "action": decision["action"]
        }
        update_prometheus_metrics(telemetry_dict, pred_dict, latency)
        
        return PredictResponse(
            predicted_servers=int(np.round(ml_data["predicted_servers"])),
            recommended_servers=decision["recommended_servers"],
            action=decision["action"],
            current_servers=payload.current_servers,
            scaling_action=decision["action"],
            reason=decision["reason"],
            reasoning=decision["reason"],
            cooldown_active=decision["cooldown_active"],
            sla_status=sla_res["status"],
            risk_score=sla_res["risk_score"],
            is_anomaly=ml_data["is_anomaly"],
            anomaly_score=ml_data["anomaly_score"],
            severity=ml_data["severity"],
            affected_metrics=ml_data["affected_metrics"],
            recommendation=ml_data["recommendation"],
            shap_explanation=ml_data["shap_explanation"],
            shap_contributions=ml_data["shap_contributions"],
            hourly_cost=opt_res["hourly_cost"],
            estimated_daily_cost=opt_res["estimated_daily_cost"],
            estimated_monthly_cost=opt_res["estimated_monthly_cost"],
            estimated_savings=opt_res["estimated_savings_daily"],
            forecasts=ml_data["forecasts"]
        )
    except Exception as e:
        logger.error(f"Error executing prediction endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastOutput)
def predict_forecasts(payload: TelemetryPayload):
    """Generates workload forecasts (delegates to ML service)."""
    API_METRICS["total_forecast_requests"] += 1
    
    input_dict = payload.dict()
    if input_dict.get("network_traffic") is None:
        input_dict["network_traffic"] = input_dict["network_in"] + input_dict["network_out"]
        
    try:
        # Wrap PredictRequest mapping
        ml_payload = {**input_dict, "current_servers": payload.current_servers}
        res = requests.post(f"{ML_SERVICE_URL}/predict_raw", json=ml_payload, timeout=5)
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail="ML Service error.")
        return ForecastOutput(forecasts=res.json()["forecasts"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting proxy failed: {str(e)}")

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
    """Evaluates telemetry workload state for active anomalies (proxies to ML Service)."""
    API_METRICS["total_anomaly_requests"] += 1
    
    input_dict = payload.dict()
    if input_dict.get("network_traffic") is None:
        input_dict["network_traffic"] = input_dict["network_in"] + input_dict["network_out"]
        
    try:
        ml_payload = {**input_dict, "current_servers": payload.current_servers}
        res = requests.post(f"{ML_SERVICE_URL}/predict_raw", json=ml_payload, timeout=5)
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail="ML Service anomaly check failed.")
        data = res.json()
        return AnomalyOutput(
            is_anomaly=data["is_anomaly"],
            anomaly_score=data["anomaly_score"],
            severity=data["severity"],
            affected_metrics=data["affected_metrics"],
            recommendation=data["recommendation"],
            reason=data["recommendation"]
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


# =====================================================================
# REINFORCEMENT LEARNING ROUTERS
# =====================================================================

@app.post("/rl/predict-action", response_model=RLPredictResponse)
def rl_predict_action(payload: RLPredictRequest):
    """
    Gateway RL Predict Route. Proxies observation payload to ML Model service,
    enforces the active operation mode (SIMULATION, APPROVAL, AUTONOMOUS) and safety validation.
    """
    try:
        # 1. Forward raw observation request to ML Service
        ml_res = requests.post(f"{ML_SERVICE_URL}/rl/predict_raw", json=payload.dict(), timeout=5)
        if ml_res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ML Service RL error: {ml_res.text}")
        rl_data = ml_res.json()
    except Exception as e:
        logger.error(f"Failed to communicate with ML Model Service for RL prediction: {e}")
        raise HTTPException(status_code=502, detail=f"ML Model service unreachable: {str(e)}")
        
    recommended_action = rl_data["recommended_action"]
    recommended_replicas = rl_data["recommended_replicas"]
    reason = rl_data["reason"]
    risk_score = rl_data["risk_score"]
    
    # 2. Enforce active mode logic
    mode_upper = RL_AUTOSCALING_MODE.upper()
    
    if mode_upper == "SIMULATION":
        reason = f"[SIMULATION] {reason}"
        
    elif mode_upper == "APPROVAL":
        reason = f"[APPROVAL REQUIRED] {reason}"
        
    elif mode_upper == "AUTONOMOUS":
        # Run proposed action step through the safety validator layer
        proposed_step = action_to_step(recommended_action)
        safe_replicas, safe_step, safety_reason = gateway_safety_validator.validate_action(
            current_replicas=payload.current_servers,
            proposed_step=proposed_step,
            metrics=payload.dict()
        )
        
        if safe_step != proposed_step:
            # Override recommended action to safe action
            recommended_replicas = safe_replicas
            # Map safe step back to action string
            if safe_step == 2: safe_idx = 4
            elif safe_step == 1: safe_idx = 3
            elif safe_step == -1: safe_idx = 1
            elif safe_step == -2: safe_idx = 0
            else: safe_idx = 2
            recommended_action = idx_to_action(safe_idx)
            reason = f"[AUTONOMOUS OVERRIDDEN] {safety_reason} (Original proposed step: {proposed_step})"
            risk_score = min(1.0, risk_score + 0.3)  # Increase risk since it was overridden
        else:
            reason = f"[AUTONOMOUS APPROVED] {reason}"
            
    return RLPredictResponse(
        current_replicas=payload.current_servers,
        recommended_action=recommended_action,
        recommended_replicas=recommended_replicas,
        expected_reward=rl_data["expected_reward"],
        risk_score=risk_score,
        reason=reason
    )

@app.post("/rl/evaluate", response_model=RLEvaluateResponse)
def rl_evaluate(payload: RLEvaluateRequest):
    """
    Gateway RL Evaluate/Train Route. Triggers simulator policy training 
    and returns baseline benchmarking comparison table.
    """
    try:
        ml_res = requests.post(f"{ML_SERVICE_URL}/rl/evaluate_raw", json=payload.dict(), timeout=600)  # Long timeout for training loops
        if ml_res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ML Service RL evaluate error: {ml_res.text}")
        return RLEvaluateResponse(**ml_res.json())
    except Exception as e:
        logger.error(f"Failed to run RL evaluation benchmark: {e}")
        raise HTTPException(status_code=502, detail=f"ML Model service unreachable/failed: {str(e)}")

@app.get("/rl/status", response_model=RLStatusResponse)
def rl_status():
    """Returns the Reinforcement Learning model state and configuration mode details."""
    try:
        ml_res = requests.get(f"{ML_SERVICE_URL}/rl/status_raw", timeout=5)
        if ml_res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ML Service RL status error: {ml_res.text}")
        status_data = ml_res.json()
        return RLStatusResponse(
            model_loaded=status_data["model_loaded"],
            checkpoint_exists=status_data["checkpoint_exists"],
            state_dimension=status_data["state_dimension"],
            action_dimension=status_data["action_dimension"],
            active_mode=RL_AUTOSCALING_MODE
        )
    except Exception as e:
        # Fallback if ml_service is down
        return RLStatusResponse(
            model_loaded=False,
            checkpoint_exists=False,
            state_dimension=15,
            action_dimension=5,
            active_mode=RL_AUTOSCALING_MODE
        )



# =====================================================================
# DIGITAL TWIN SIMULATION ROUTERS
# =====================================================================

@app.post("/simulation/run-scenario", response_model=SimulationScenarioResponse)
def run_simulation_scenario(payload: SimulationScenarioRequest):
    """
    Gateway Simulation Scenario Route. Evaluates the custom stress scenario 
    against the in-memory Digital Twin cloud engine.
    """
    try:
        ml_res = requests.post(f"{ML_SERVICE_URL}/simulation/scenario_raw", json=payload.dict(), timeout=10)
        if ml_res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ML Service simulation scenario error: {ml_res.text}")
        return SimulationScenarioResponse(**ml_res.json())
    except Exception as e:
        logger.error(f"Failed to execute Digital Twin scenario run: {e}")
        raise HTTPException(status_code=502, detail=f"ML Model service unreachable/failed: {str(e)}")

@app.post("/simulation/evaluate-policies", response_model=SimulationCompareResponse)
def evaluate_simulation_policies(payload: SimulationCompareRequest):
    """
    Gateway Simulation Policy Compare Route. Runs comparison benchmarks 
    comparing Static, Threshold, HPA, ML Predictive, and PPO RL autoscaling.
    """
    try:
        ml_res = requests.post(f"{ML_SERVICE_URL}/simulation/evaluate_raw", json=payload.dict(), timeout=30)
        if ml_res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ML Service simulation compare error: {ml_res.text}")
        return SimulationCompareResponse(**ml_res.json())
    except Exception as e:
        logger.error(f"Failed to execute Digital Twin policy evaluation: {e}")
        raise HTTPException(status_code=502, detail=f"ML Model service unreachable/failed: {str(e)}")


# =====================================================================
# MULTI-OBJECTIVE OPTIMIZER ROUTERS
# =====================================================================

@app.post("/optimizer/optimize", response_model=MultiObjectiveOptimizeResponse)
def optimizer_optimize(payload: MultiObjectiveOptimizeRequest):
    """
    Gateway Multi-Objective Sizing Optimizer Route. Proxies telemetry payload 
    to ML Model Service and returns scenario comparisons alongside Pareto optimal lists.
    """
    try:
        ml_res = requests.post(f"{ML_SERVICE_URL}/optimizer/multi_objective_raw", json=payload.dict(), timeout=15)
        if ml_res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ML Service multi-objective optimization error: {ml_res.text}")
        return MultiObjectiveOptimizeResponse(**ml_res.json())
    except Exception as e:
        logger.error(f"Failed to execute multi-objective optimization sizing: {e}")
        raise HTTPException(status_code=502, detail=f"ML Model service unreachable/failed: {str(e)}")



