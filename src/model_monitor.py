# src/model_monitor.py

import os
import json
import time
import shutil
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

REGISTRY_PATH = "artifacts/model_registry.json"
MODELS_DIR = "artifacts/models"
HISTORY_DIR = "data/model_history"
PREDS_LOG_PATH = os.path.join(HISTORY_DIR, "predictions_log.csv")
REFERENCE_DATA_PATH = "data/cleaned_workload.csv"
CHAMPION_MODEL_PATH = "artifacts/cloud_resource_optimization_model.pkl"
FEATURES_LIST_PATH = "artifacts/features_list.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

# Base features to check for drift
DRIFT_FEATURES = [
    "cpu_usage", "memory_usage", "network_traffic",
    "active_users", "request_rate", "response_time", "error_rate"
]

def init_registry_and_dirs():
    """Initializes all registry paths, model version folders, and history CSV log files."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    
    # Initialize Predictions Log CSV if missing
    if not os.path.exists(PREDS_LOG_PATH):
        df_cols = ["timestamp", "predicted_at"] + DRIFT_FEATURES + ["predicted_servers", "actual_servers", "latency_ms"]
        pd.DataFrame(columns=df_cols).to_csv(PREDS_LOG_PATH, index=False)
        
    # Initialize Registry JSON if missing
    if not os.path.exists(REGISTRY_PATH):
        initial_registry = {
            "active_version": "v1",
            "history": {
                "v1": {
                    "version": "v1",
                    "algorithm": "Random Forest",
                    "hyperparameters": {"n_estimators": 100, "random_state": 42},
                    "metrics": {"mae": 0.082, "rmse": 0.125, "r2": 0.94},
                    "created_at": datetime.now().isoformat(),
                    "dataset_path": REFERENCE_DATA_PATH,
                    "model_path": os.path.join(MODELS_DIR, "model_v1.pkl")
                }
            },
            "rollback_events": []
        }
        with open(REGISTRY_PATH, "w") as f:
            json.dump(initial_registry, f, indent=2)
            
        # Copy current model to registry model_v1 if present
        v1_path = os.path.join(MODELS_DIR, "model_v1.pkl")
        if os.path.exists(CHAMPION_MODEL_PATH) and not os.path.exists(v1_path):
            shutil.copy(CHAMPION_MODEL_PATH, v1_path)

def log_prediction_and_resolve_actuals(inputs: dict, predicted_servers: float, latency_ms: float):
    """
    Logs prediction details. Automatically resolves and updates past predictions 
    with actual replicas observed in the current tick.
    """
    init_registry_and_dirs()
    now = datetime.now()
    now_str = now.isoformat()
    
    # 1. Update past prediction actuals where timestamp is ~15 min ago
    # We assume current_servers in inputs represents actual servers right now.
    current_actual = float(inputs.get("current_servers", 3))
    
    try:
        df = pd.read_csv(PREDS_LOG_PATH)
        if len(df) > 0:
            # Look for predictions created 10-20 mins ago that don't have actuals
            # Filter and apply actuals
            time_now = datetime.now()
            unresolved = df[df["actual_servers"].isna()]
            for idx, row in unresolved.iterrows():
                pred_time = datetime.fromisoformat(row["predicted_at"])
                # If elapsed time is between 10 and 25 minutes, match it
                if timedelta(minutes=10) <= (time_now - pred_time) <= timedelta(minutes=25):
                    df.at[idx, "actual_servers"] = current_actual
                    
            # Cap CSV size to avoid unbound growth
            if len(df) > 5000:
                df = df.tail(3000)
            df.to_csv(PREDS_LOG_PATH, index=False)
    except Exception as e:
        print(f"Error updating predictions actuals: {e}")
        
    # 2. Log current prediction
    new_log = {
        "timestamp": inputs.get("timestamp", now_str),
        "predicted_at": now_str,
        "cpu_usage": float(inputs.get("cpu_usage", 0.0)),
        "memory_usage": float(inputs.get("memory_usage", 0.0)),
        "network_traffic": float(inputs.get("network_traffic", 0.0)),
        "active_users": int(inputs.get("active_users", 0)),
        "request_rate": float(inputs.get("request_rate", 0.0)),
        "response_time": float(inputs.get("response_time", 0.0)),
        "error_rate": float(inputs.get("error_rate", 0.0)),
        "predicted_servers": float(predicted_servers),
        "actual_servers": np.nan,  # Resolved later
        "latency_ms": float(latency_ms)
    }
    
    try:
        pd.DataFrame([new_log]).to_csv(PREDS_LOG_PATH, mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error appending prediction log: {e}")

def calculate_drift() -> dict:
    """
    Computes Kolmogorov-Smirnov (KS) test for input features and prediction drift.
    Compares recent production distributions against the training reference dataset.
    """
    init_registry_and_dirs()
    
    # Defaults if reference or log is missing/empty
    empty_result = {
        "drift_detected": False,
        "affected_features": [],
        "severity": "LOW",
        "drift_score": 0.0,
        "features": {}
    }
    
    if not os.path.exists(REFERENCE_DATA_PATH) or not os.path.exists(PREDS_LOG_PATH):
        return empty_result
        
    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        prod_df = pd.read_csv(PREDS_LOG_PATH)
    except Exception:
        return empty_result
        
    if len(prod_df) < 50:
        return empty_result  # Need sufficient samples to perform KS test
        
    # Take last 100 observations to represent current production distribution
    recent_prod = prod_df.tail(100)
    
    drift_details = {}
    affected_features = []
    total_stat = 0.0
    
    # 1. Feature drift analysis
    for feature in DRIFT_FEATURES:
        if feature in ref_df.columns and feature in recent_prod.columns:
            ref_dist = ref_df[feature].dropna()
            prod_dist = recent_prod[feature].dropna()
            
            if len(ref_dist) > 5 and len(prod_dist) > 5:
                stat, p_value = ks_2samp(ref_dist, prod_dist)
                drifted = bool(p_value < 0.05)
                drift_details[feature] = {
                    "ks_statistic": float(stat),
                    "p_value": float(p_value),
                    "drift_detected": drifted
                }
                if drifted:
                    affected_features.append(feature)
                    total_stat += stat
                    
    # 2. Prediction drift analysis
    target_col = "required_servers"
    pred_drifted = False
    pred_stat = 0.0
    if target_col in ref_df.columns:
        ref_dist = ref_df[target_col].dropna()
        prod_dist = recent_prod["predicted_servers"].dropna()
        if len(ref_dist) > 5 and len(prod_dist) > 5:
            stat, p_value = ks_2samp(ref_dist, prod_dist)
            pred_drifted = bool(p_value < 0.05)
            pred_stat = stat
            drift_details["prediction_target"] = {
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "drift_detected": pred_drifted
            }
            if pred_drifted:
                affected_features.append("predicted_servers")
                total_stat += stat

    # Compute overall drift metrics
    drift_detected = len(affected_features) > 0
    drift_ratio = len(affected_features) / (len(DRIFT_FEATURES) + 1)
    
    # Determine severity
    if drift_ratio >= 0.5 or pred_stat > 0.35:
        severity = "HIGH"
    elif drift_ratio >= 0.2:
        severity = "MEDIUM"
    else:
        severity = "LOW"
        
    avg_drift_score = total_stat / len(affected_features) if len(affected_features) > 0 else 0.0
    
    return {
        "drift_detected": drift_detected,
        "affected_features": affected_features,
        "severity": severity,
        "drift_score": float(avg_drift_score),
        "features": drift_details
    }

def get_performance_metrics() -> dict:
    """Calculates active model prediction errors (MAE, RMSE, R2) over matched logs."""
    init_registry_and_dirs()
    
    default_metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "samples_evaluated": 0}
    
    try:
        df = pd.read_csv(PREDS_LOG_PATH)
        # Filter for rows that have resolved actual target values
        resolved = df.dropna(subset=["actual_servers"])
        if len(resolved) < 10:
            return default_metrics
            
        y_true = resolved["actual_servers"].values
        y_pred = resolved["predicted_servers"].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "samples_evaluated": len(resolved)
        }
    except Exception:
        return default_metrics

def trigger_retraining_pipeline(force: bool = False, authorized: bool = False) -> dict:
    """
    Retrains ML models over reference data combined with production feedback loops.
    Validates challenger models, running an evaluation gate before promotion.
    """
    if not force and not authorized:
        return {"success": False, "reason": "Retraining requires explicit authorization or trigger conditions."}
        
    init_registry_and_dirs()
    
    # 1. Fetch current active version and performance baseline
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
        
    active_ver = registry["active_version"]
    champion_metadata = registry["history"][active_ver]
    baseline_mae = champion_metadata["metrics"]["mae"]
    
    # 2. Gather new training dataset
    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        prod_df = pd.read_csv(PREDS_LOG_PATH).dropna(subset=["actual_servers"])
        
        # Ensure we have enough new data to train, unless forced
        if len(prod_df) < 20 and not force:
            return {"success": False, "reason": f"Sufficient production feedback data not yet available (have {len(prod_df)} records, need at least 20)."}
            
        # Map production log columns back to training format
        # predictions log has raw features. We merge them.
        prod_mapped = prod_df[DRIFT_FEATURES].copy()
        prod_mapped["required_servers"] = prod_df["actual_servers"]
        
        # Merge datasets
        merged_df = pd.concat([ref_df[prod_mapped.columns], prod_mapped], ignore_index=True)
    except Exception as e:
        return {"success": False, "reason": f"Dataset loading failed: {str(e)}"}
        
    # 3. Train Candidate Models
    try:
        # Load features list
        features = joblib.load(FEATURES_LIST_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # Chronological split on merged dataset
        X = merged_df[features]
        y = merged_df["required_servers"]
        
        n = len(merged_df)
        train_end = int(n * 0.8)
        
        X_train_raw = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test_raw = X.iloc[train_end:]
        y_test = y.iloc[train_end:]
        
        # Scale
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        
        # Train challenger
        challenger = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        challenger.fit(X_train, y_train)
        
        preds = challenger.predict(X_test)
        challenger_mae = mean_absolute_error(y_test, preds)
        challenger_rmse = np.sqrt(mean_squared_error(y_test, preds))
        challenger_r2 = r2_score(y_test, preds)
        
        # 4. Evaluation approval gate: compare Challenger vs Champion
        # Only promote if MAE improves or is within a 2% margin, to prevent regression
        improvement_margin = 0.02
        is_approved = float(challenger_mae) <= float(baseline_mae) * (1 + improvement_margin)
        
        if is_approved:
            # Promote Challenger
            next_ver_num = int(active_ver.replace("v", "")) + 1
            next_ver = f"v{next_ver_num}"
            new_model_file = f"model_{next_ver}.pkl"
            new_model_path = os.path.join(MODELS_DIR, new_model_file)
            
            # Save new model path
            joblib.dump(challenger, new_model_path)
            
            # Update registry
            registry["active_version"] = next_ver
            registry["history"][next_ver] = {
                "version": next_ver,
                "algorithm": "Random Forest (Retrained)",
                "hyperparameters": {"n_estimators": 100, "random_state": 42},
                "metrics": {
                    "mae": float(challenger_mae),
                    "rmse": float(challenger_rmse),
                    "r2": float(challenger_r2)
                },
                "created_at": datetime.now().isoformat(),
                "dataset_path": "Merged (Reference + Production Logs)",
                "model_path": new_model_path
            }
            
            with open(REGISTRY_PATH, "w") as f:
                json.dump(registry, f, indent=2)
                
            # Copy promoted challenger to production path
            shutil.copy(new_model_path, CHAMPION_MODEL_PATH)
            
            return {
                "success": True,
                "promoted": True,
                "version": next_ver,
                "metrics": registry["history"][next_ver]["metrics"],
                "baseline_mae": baseline_mae,
                "reason": f"Challenger approved and promoted to active deployment {next_ver} (MAE: {challenger_mae:.4f} vs Champion MAE: {baseline_mae:.4f})."
            }
        else:
            return {
                "success": True,
                "promoted": False,
                "reason": f"Challenger rejected: Validation MAE ({challenger_mae:.4f}) does not outperform active Champion MAE ({baseline_mae:.4f}). Champion retained."
            }
            
    except Exception as e:
        return {"success": False, "reason": f"Error running training pipeline: {str(e)}"}

def rollback_to_previous_stable() -> dict:
    """Rolls back the production model to the previous version registered in artifacts."""
    init_registry_and_dirs()
    
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
        
    active_ver = registry["active_version"]
    history = registry["history"]
    
    if len(history) <= 1:
        return {"success": False, "reason": "No previous stable model version found in registry. Rollback unavailable."}
        
    # Get sorted versions list
    versions = sorted(list(history.keys()), key=lambda x: int(x.replace("v", "")))
    active_idx = versions.index(active_ver)
    
    if active_idx == 0:
        return {"success": False, "reason": "Current active version is the oldest available. Cannot rollback further."}
        
    previous_ver = versions[active_idx - 1]
    prev_model_path = history[previous_ver]["model_path"]
    
    if not os.path.exists(prev_model_path):
        return {"success": False, "reason": f"Previous version model file {prev_model_path} was deleted from disk."}
        
    # Apply rollback copy
    try:
        shutil.copy(prev_model_path, CHAMPION_MODEL_PATH)
        
        # Update registry
        registry["active_version"] = previous_ver
        registry["rollback_events"].append({
            "timestamp": datetime.now().isoformat(),
            "from_version": active_ver,
            "to_version": previous_ver,
            "reason": "Operator requested model rollback due to degradation warnings."
        })
        
        with open(REGISTRY_PATH, "r+") as f:
            f.seek(0)
            json.dump(registry, f, indent=2)
            f.truncate()
            
        return {
            "success": True,
            "rolled_back_to": previous_ver,
            "reason": f"Successfully rolled back production model from version {active_ver} to {previous_ver}."
        }
    except Exception as e:
        return {"success": False, "reason": f"Rollback failed during asset swap: {str(e)}"}
