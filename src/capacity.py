import math
import numpy as np

def calculate_required_servers(
    prediction: float, 
    current_servers: int, 
    min_servers: int = 1, 
    max_servers: int = 20, 
    safety_margin: float = 0.10
) -> dict:
    """
    Implements a safe, risk-managed capacity planning strategy.
    
    Rather than blindly rounding statistical predictions, this function:
    1. Uses ceiling logic (ceil) to avoid under-provisioning.
    2. Incorporates a percentage safety margin to absorb sudden workload surges.
    3. Clamps recommendations between MIN_SERVERS and MAX_SERVERS bounds.
    4. Guarantees server recommendations never drop to negative or zero counts.
    """
    # 1. Enforce non-negative predictions
    raw_pred = max(0.0, prediction)
    
    # 2. Calculate safety buffer
    safety_buffer = raw_pred * safety_margin
    
    # 3. Apply safety margin and ceiling logic
    # Ceil is standard for capacity planning because under-provisioning causes 
    # resource starvation, whereas minor over-provisioning simply increases cost.
    projected = raw_pred + safety_buffer
    recommended = math.ceil(projected)
    
    # 4. Enforce strict minimum/maximum server boundaries
    # Guarantees that recommended server count is at least min_servers (which is >= 1)
    min_servers = max(1, min_servers)
    max_servers = max(min_servers, max_servers)
    recommended_servers = max(min_servers, min(max_servers, recommended))
    
    return {
        "predicted_servers": float(raw_pred),
        "recommended_servers": int(recommended_servers),
        "safety_margin": float(safety_margin),
        "safety_buffer": float(safety_buffer)
    }

def estimate_prediction_uncertainty(model, X_scaled) -> dict:
    """
    Estimates the statistical prediction uncertainty of the machine learning model.
    For Random Forest models, this computes the standard deviation and 95% confidence
    intervals across predictions from all individual decision trees in the ensemble.
    """
    # Check if the model is an ensemble forest (e.g., RandomForestRegressor, ExtraTreesRegressor)
    if hasattr(model, "estimators_") and model.estimators_:
        try:
            # Gather predictions from each individual estimator tree
            tree_predictions = []
            for tree in model.estimators_:
                pred = tree.predict(X_scaled)[0]
                tree_predictions.append(pred)
                
            tree_predictions = np.array(tree_predictions)
            
            # Compute statistical uncertainty metrics
            std_dev = float(np.std(tree_predictions))
            lower_ci = float(np.percentile(tree_predictions, 2.5))  # 95% CI Lower Bound
            upper_ci = float(np.percentile(tree_predictions, 97.5))  # 95% CI Upper Bound
            
            return {
                "uncertainty_std": std_dev,
                "confidence_interval_lower": lower_ci,
                "confidence_interval_upper": upper_ci,
                "estimators_evaluated": len(tree_predictions)
            }
        except Exception as e:
            # Fallback on evaluation error
            pass
            
    # Default fallback for models that do not support estimators (e.g. LinearRegression, XGBoost)
    return {
        "uncertainty_std": 0.0,
        "confidence_interval_lower": float(model.predict(X_scaled)[0]) if hasattr(model, "predict") else 0.0,
        "confidence_interval_upper": float(model.predict(X_scaled)[0]) if hasattr(model, "predict") else 0.0,
        "estimators_evaluated": 0
    }
