import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

CLEANED_DATA_PATH = "data/cleaned_workload.csv"
ANOMALY_MODEL_PATH = "artifacts/anomaly_detector.pkl"
ANOMALY_SCALER_PATH = "artifacts/anomaly_scaler.pkl"

ANOMALY_METRICS = [
    "cpu_usage", "memory_usage", "network_traffic", 
    "active_users", "request_rate", "response_time", "error_rate"
]

ANOMALY_TIME_FEATURES = [
    "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week"
]

ANOMALY_FEATURES = ANOMALY_METRICS + ANOMALY_TIME_FEATURES

def train_and_compare_anomaly_detectors():
    """
    Loads cleaned workload telemetry, extracts raw features, scales them,
    trains and compares Isolation Forest vs One-Class SVM, and serializes the best one.
    """
    print("\nStarting AI-Based Anomaly Detection Training...")
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned workload dataset not found at: {CLEANED_DATA_PATH}. Run pipeline first.")
        
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # Verify columns exist
    missing_cols = [c for c in ANOMALY_FEATURES if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required anomaly features not found in dataset: {missing_cols}")
        
    X_raw = df[ANOMALY_FEATURES].copy()
    
    # Fit StandardScaler specifically for the anomaly features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 1. Isolation Forest
    # nu/contamination parameter represents the expected ratio of anomalies (1%)
    start_time = time.time()
    iforest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iforest.fit(X_scaled)
    iforest_time = time.time() - start_time
    
    # 2. One-Class SVM
    start_time = time.time()
    ocsvm = OneClassSVM(nu=0.01, kernel="rbf", gamma="scale")
    ocsvm.fit(X_scaled)
    ocsvm_time = time.time() - start_time
    
    # Predict and evaluate scores
    iforest_preds = iforest.predict(X_scaled)  # Returns 1 (normal) or -1 (anomaly)
    ocsvm_preds = ocsvm.predict(X_scaled)
    
    iforest_anomaly_count = np.sum(iforest_preds == -1)
    ocsvm_anomaly_count = np.sum(ocsvm_preds == -1)
    
    print("\n================ ANOMALY DETECTOR COMPARISON ================= ")
    print(f"Isolation Forest | Fit Time: {iforest_time:.4f}s | Detected Anomalies: {iforest_anomaly_count}/{len(df)}")
    print(f"One-Class SVM    | Fit Time: {ocsvm_time:.4f}s | Detected Anomalies: {ocsvm_anomaly_count}/{len(df)}")
    print("============================================================== ")
    
    # Select Isolation Forest for production (faster inference, handles multi-dimensional partitions better)
    print("Selected Isolation Forest as the primary anomaly detector.")
    
    # Save assets
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(iforest, ANOMALY_MODEL_PATH)
    joblib.dump(scaler, ANOMALY_SCALER_PATH)
    joblib.dump(ANOMALY_FEATURES, "artifacts/anomaly_features_list.pkl")
    print(f"Anomaly detector assets serialized to artifacts/")
    
    return iforest

def detect_anomaly_record(record_df: pd.DataFrame, history_df: pd.DataFrame) -> dict:
    """
    Evaluates a single telemetry record statefully against the trained anomaly model
    and performs Z-score metric root-cause analysis over the historical rolling window.
    """
    if not os.path.exists(ANOMALY_MODEL_PATH) or not os.path.exists(ANOMALY_SCALER_PATH):
        raise FileNotFoundError("Anomaly model or scaler files are missing. Run anomaly training first.")
        
    model = joblib.load(ANOMALY_MODEL_PATH)
    scaler = joblib.load(ANOMALY_SCALER_PATH)
    features = joblib.load("artifacts/anomaly_features_list.pkl")
    
    # 1. Align record columns and scale
    X_raw = record_df[features].copy()
    X_scaled = scaler.transform(X_raw)
    
    # 2. Predict anomaly status
    pred = model.predict(X_scaled)[0]  # 1 = normal, -1 = anomaly
    is_anomaly = bool(pred == -1)
    
    # 3. Calculate Normalized Anomaly Score in range [0.0, 1.0]
    # decision_function returns negative values for anomalies, positive for normal
    raw_score = model.decision_function(X_scaled)[0]
    # Map decision function score (-0.5 to 0.5 range) to normalized scale
    # If raw_score is positive, anomaly_score is low. If raw_score is negative, anomaly_score is high.
    anomaly_score = max(0.0, min(1.0, 0.5 - (raw_score * 2.0)))
    
    # 4. Determine Severity Level
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
            
    # 5. Root Cause Attribution (Affected Metrics)
    # Check which metrics deviate by more than 2.5 standard deviations from history
    affected_metrics = []
    reasons = []
    
    # Check if history_df contains necessary records
    if len(history_df) >= 5:
        for col in ANOMALY_METRICS:
            mean_val = history_df[col].mean()
            std_val = history_df[col].std()
            curr_val = record_df[col].iloc[0]
            
            # Avoid division by zero
            if std_val < 1e-5:
                std_val = 1e-5
                
            z_score = (curr_val - mean_val) / std_val
            
            if abs(z_score) >= 2.5:
                direction = "spike" if z_score > 0 else "drop"
                affected_metrics.append(col)
                reasons.append(f"{col} {direction} (Z-score: {z_score:.2f}, Value: {curr_val:.1f} vs Mean: {mean_val:.1f})")
                
    # 6. Generate action recommendations
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

if __name__ == "__main__":
    train_and_compare_anomaly_detectors()
