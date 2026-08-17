import pandas as pd
import numpy as np
from app.models.manager import model_manager
from src.anomaly import ANOMALY_METRICS

def detect_anomaly(record_df: pd.DataFrame, history_df: pd.DataFrame) -> dict:
    """
    Evaluates the anomaly status of the current telemetry record using 
    the model manager's cached Isolation Forest model.
    """
    model = model_manager.anomaly_detector
    scaler = model_manager.anomaly_scaler
    features = model_manager.anomaly_features
    
    if model is None or scaler is None or features is None:
        raise RuntimeError("Anomaly detector assets are not loaded in model manager.")
        
    # Align record columns and scale
    X_raw = record_df[features].copy()
    X_scaled = scaler.transform(X_raw)
    
    # Predict anomaly status
    pred = model.predict(X_scaled)[0]  # 1 = normal, -1 = anomaly
    is_anomaly = bool(pred == -1)
    
    # Calculate Normalized Anomaly Score
    raw_score = model.decision_function(X_scaled)[0]
    anomaly_score = max(0.0, min(1.0, 0.5 - (raw_score * 2.0)))
    
    # Severity
    if is_anomaly:
        if anomaly_score >= 0.55:
            severity = "CRITICAL"
        elif anomaly_score >= 0.35:
            severity = "HIGH"
        else:
            severity = "MEDIUM"
    else:
        if anomaly_score >= 0.15:
            severity = "MEDIUM"
        else:
            severity = "LOW"
            
    # Z-score root-cause attribution
    affected_metrics = []
    reasons = []
    
    if len(history_df) >= 5:
        for col in ANOMALY_METRICS:
            mean_val = history_df[col].mean()
            std_val = history_df[col].std()
            curr_val = record_df[col].iloc[0]
            
            if std_val < 1e-5:
                std_val = 1e-5
                
            z_score = (curr_val - mean_val) / std_val
            
            if abs(z_score) >= 2.5:
                direction = "spike" if z_score > 0 else "drop"
                affected_metrics.append(col)
                reasons.append(f"{col} {direction} (Z-score: {z_score:.2f}, Value: {curr_val:.1f} vs Mean: {mean_val:.1f})")
                
    if is_anomaly:
        affected_str = ", ".join(affected_metrics) if affected_metrics else "suspicious workload pattern"
        recommendation = (
            f"ALERT: Anomaly detected! Affected indicators: [{affected_str}]. "
            f"Severity: {severity}. Activating proactive capacity scaling overrides."
        )
    else:
        recommendation = "System operation is normal. No anomalous metrics detected."
        
    reason_str = "; ".join(reasons) if reasons else "All metrics are behaving within normal seasonal workload distributions."
    
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(anomaly_score, 4),
        "severity": severity,
        "affected_metrics": affected_metrics,
        "recommendation": recommendation,
        "reason": reason_str
    }
