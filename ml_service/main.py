from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import numpy as np
import os
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

# Import core business logic from src
from src.pipeline import BASE_FEATURES, preprocess_single_record
from src.forecasting import forecast_next_workloads
from src.capacity import calculate_required_servers, estimate_prediction_uncertainty
from src.anomaly import detect_anomaly_record
from src.explainability import explain_prediction_shap
import shap

# Import RL modules
import torch
from rl.agent import PPOAgent
from rl.state import get_observation
from rl.actions import idx_to_action, idx_to_step
from rl.evaluator import Evaluator
from rl.trainer import train_ppo_agent
from rl.safety import SafetyValidator

# Import Simulation modules
from simulation.scenarios import WhatIfAnalyzer
from simulation.experiments import ExperimentSuite

from src.multi_objective_optimizer import run_multi_objective_optimization

from src.model_monitor import (
    log_prediction_and_resolve_actuals,
    calculate_drift,
    get_performance_metrics,
    trigger_retraining_pipeline,
    rollback_to_previous_stable,
    REGISTRY_PATH,
    init_registry_and_dirs
)

import time




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

# Reinforcement Learning Globals
rl_agent = None
rl_safety = None
rl_checkpoint_path = "rl/models/ppo_autoscaler.pth"
rl_model_loaded = False

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
        
    # 6. Initialize RL Agent and Safety Validator
    global rl_agent, rl_safety, rl_model_loaded
    rl_agent = PPOAgent(state_dim=15, action_dim=5)
    rl_safety = SafetyValidator()
    if os.path.exists(rl_checkpoint_path):
        try:
            rl_agent.load(rl_checkpoint_path)
            rl_model_loaded = True
            print("Successfully loaded PPO autoscaler checkpoint.")
        except Exception as e:
            print(f"Error loading PPO checkpoint: {e}")
            rl_model_loaded = False
    else:
        print("PPO checkpoint not found. Agent must train first in simulation.")
        rl_model_loaded = False


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
    start_time = time.perf_counter()
    
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
        
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        log_prediction_and_resolve_actuals(input_dict, capacity["recommended_servers"], latency_ms)

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


# =====================================================================
# REINFORCEMENT LEARNING ROUTERS
# =====================================================================

class RLPredictRawInput(BaseModel):
    cpu_usage: float
    memory_usage: float
    network_traffic: float
    active_users: int
    request_rate: float
    response_time: float
    error_rate: float
    current_servers: int
    predicted_workload: float
    predicted_required_servers: int
    hourly_cost: float
    sla_status: str
    is_anomaly: bool
    prev_step: int
    hour: float

class RLEvaluateRawRequest(BaseModel):
    episodes: int = 5
    seed: int = 42

@app.post("/rl/predict_raw")
def rl_predict_raw(payload: RLPredictRawInput):
    global rl_agent, rl_model_loaded
    
    if rl_agent is None:
        raise HTTPException(status_code=503, detail="RL agent not initialized.")
        
    metrics = payload.dict()
    # Normalize inputs to observation vector (15 dimensions)
    obs = get_observation(metrics)
    
    # Run PPO forward pass to get action and critic estimation
    with torch.no_grad():
        state_t = torch.FloatTensor(obs)
        action_probs = rl_agent.policy_old.actor(state_t)
        dist = torch.distributions.Categorical(action_probs)
        
        # Select action index
        if rl_model_loaded:
            action_idx = action_probs.argmax().item()
        else:
            action_idx = 2  # default NO_ACTION
            
        # Expected reward estimation from critic
        critic_val = rl_agent.policy_old.critic(state_t).item()
        
        # Risk score proportional to categorical distribution entropy (pi uncertainty)
        dist_entropy = dist.entropy().item()
        # Max entropy for 5 actions is ln(5) ~ 1.609. Normalize to [0, 1]
        risk_score = min(1.0, max(0.0, float(dist_entropy) / 1.61))
        
    recommended_action = idx_to_action(action_idx)
    step_change = idx_to_step(action_idx)
    recommended_replicas = int(np.clip(payload.current_servers + step_change, 1, 20))
    
    # Reason message
    if step_change > 0:
        reason = f"RL Agent recommends scaling UP by {step_change} server(s) to optimize workload performance."
    elif step_change < 0:
        reason = f"RL Agent recommends scaling DOWN by {abs(step_change)} server(s) to reduce hosting costs."
    else:
        reason = "RL Agent recommends NO_ACTION as server capacity is currently optimal."
        
    return {
        "current_replicas": payload.current_servers,
        "recommended_action": recommended_action,
        "recommended_replicas": recommended_replicas,
        "expected_reward": float(critic_val),
        "risk_score": float(risk_score),
        "reason": reason
    }

@app.post("/rl/evaluate_raw")
def rl_evaluate_raw(payload: RLEvaluateRawRequest):
    global rl_agent, rl_model_loaded
    
    try:
        # 1. Train agent in simulation if episodes > 0
        if payload.episodes > 0:
            print(f"Triggering RL simulator training loop for {payload.episodes} episodes...")
            train_ppo_agent(episodes=payload.episodes, seed=payload.seed)
            
            # Reload updated checkpoint weights
            if os.path.exists(rl_checkpoint_path):
                rl_agent.load(rl_checkpoint_path)
                rl_model_loaded = True
                
        # 2. Execute benchmark comparison across all autoscaler algorithms
        evaluator = Evaluator(checkpoint_path=rl_checkpoint_path)
        df_results = evaluator.run_benchmark(seed=payload.seed)
        
        # Convert pandas DataFrame comparison to dict list
        benchmark_list = df_results.to_dict(orient="records")
        return {"benchmark_results": benchmark_list}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RL evaluation cycle failed: {str(e)}")

@app.get("/rl/status_raw")
def rl_status_raw():
    global rl_model_loaded, rl_checkpoint_path
    checkpoint_exists = os.path.exists(rl_checkpoint_path)
    return {
        "model_loaded": rl_model_loaded,
        "checkpoint_exists": checkpoint_exists,
        "state_dimension": 15,
        "action_dimension": 5
    }


# =====================================================================
# DIGITAL TWIN SIMULATION ROUTERS
# =====================================================================

class SimulationScenarioRawRequest(BaseModel):
    policy_name: str = "HPA"
    initial_replicas: int = 5
    max_steps: int = 288
    traffic_multiplier: float = 1.0
    users_multiplier: float = 1.0
    workload_patterns: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

class SimulationCompareRawRequest(BaseModel):
    initial_replicas: int = 5
    max_steps: int = 288
    traffic_multiplier: float = 1.0
    users_multiplier: float = 1.0
    workload_patterns: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

@app.post("/simulation/scenario_raw")
def simulation_scenario_raw(payload: SimulationScenarioRawRequest):
    global rl_agent, rl_model_loaded
    try:
        analyzer = WhatIfAnalyzer()
        res = analyzer.run_custom_scenario(
            config=payload.dict(),
            ppo_agent=rl_agent,
            model_loaded=rl_model_loaded
        )
        return {
            "summary": res["summary"],
            "history": res["history"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation scenario failed: {str(e)}")

@app.post("/simulation/evaluate_raw")
def simulation_evaluate_raw(payload: SimulationCompareRawRequest):
    global rl_agent, rl_model_loaded
    try:
        suite = ExperimentSuite()
        results = suite.run_policy_comparison(
            scenario_config=payload.dict(),
            ppo_agent=rl_agent,
            model_loaded=rl_model_loaded
        )
        return {"benchmark_results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Simulation comparison failed: {str(e)}")


# =====================================================================
# MULTI-OBJECTIVE OPTIMIZER ROUTERS
# =====================================================================

class MultiObjectiveOptimizeRawRequest(BaseModel):
    workload: Dict[str, Any]
    current_configuration: Dict[str, Any]
    weights: Dict[str, float] = {}
    constraints: Dict[str, Any] = {}

@app.post("/optimizer/multi_objective_raw")
def optimizer_multi_objective_raw(payload: MultiObjectiveOptimizeRawRequest):
    try:
        res = run_multi_objective_optimization(
            workload=payload.workload,
            current_config=payload.current_configuration,
            weights=payload.weights,
            constraints=payload.constraints
        )
        return res
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Multi-Objective optimization execution failed: {str(e)}")


# =====================================================================
# MODEL SELF-MONITORING & RETRAINING ROUTERS
# =====================================================================

@app.get("/model/status")
def model_status():
    """Returns active model details, registry metadata, and historical rollback events."""
    init_registry_and_dirs()
    try:
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
        active_ver = registry["active_version"]
        champion_meta = registry["history"][active_ver]
        
        # Calculate model age
        created_dt = datetime.fromisoformat(champion_meta["created_at"])
        age_days = (datetime.now() - created_dt).days
        
        # Check last retraining
        retrain_history = [v for k, v in registry["history"].items() if k != "v1"]
        last_retrained = retrain_history[-1]["created_at"] if retrain_history else "Never"
        
        return {
            "active_version": active_ver,
            "algorithm": champion_meta["algorithm"],
            "created_at": champion_meta["created_at"],
            "age_days": age_days,
            "last_retrained": last_retrained,
            "rollback_events": registry.get("rollback_events", []),
            "champion_model": champion_meta,
            "challenger_model": {
                "status": "IDLE",
                "ready": True
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model registry: {str(e)}")

@app.get("/model/metrics")
def model_metrics():
    """Retrieves rolling prediction errors (MAE, RMSE, R2) based on logged feedback."""
    return get_performance_metrics()

@app.get("/model/drift")
def model_drift():
    """Computes Kolmogorov-Smirnov drift detection for all input features and predictions."""
    return calculate_drift()

class RetrainRequest(BaseModel):
    force: bool = False
    authorized: bool = False

@app.post("/model/retrain")
def model_retrain(payload: RetrainRequest):
    """Executes the automated retraining pipeline and reloads models in memory upon promotion."""
    try:
        res = trigger_retraining_pipeline(force=payload.force, authorized=payload.authorized)
        if res.get("success") and res.get("promoted"):
            # Dynamically reload active assets in memory
            load_assets()
            print("Retraining completed and Challenger promoted! Assets reloaded in memory.")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.post("/model/rollback")
def model_rollback():
    """Rolls back the active production model to the previous stable version and reloads assets."""
    try:
        res = rollback_to_previous_stable()
        if res.get("success"):
            load_assets()
            print("Rollback completed! Previous assets reloaded in memory.")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rollback failed: {str(e)}")




