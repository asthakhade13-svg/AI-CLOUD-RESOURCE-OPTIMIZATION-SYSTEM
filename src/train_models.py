import pandas as pd
import numpy as np
import os
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

CLEANED_DATA_PATH = "data/cleaned_workload.csv"
SCALER_PATH = "artifacts/scaler.pkl"
MODEL_PATH = "artifacts/cloud_resource_optimization_model.pkl"

FEATURES = [
    "cpu_usage", "memory_usage", "network_in", "network_out", 
    "network_traffic", "disk_read", "disk_write", "active_users", 
    "request_rate", "response_time", "error_rate", "current_servers", 
    "server_cost"
]
TARGET = "required_servers"

def train_and_compare_models():
    """
    Trains Random Forest and XGBoost regression models on the preprocessed 15-column schema dataset.
    Compares their performance, outputs metrics, and saves the best model.
    """
    print("\nLoading preprocessed dataset for model training...")
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at: {CLEANED_DATA_PATH}. Please run pipeline first.")
        
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # 1. Feature Target Split
    X = df[FEATURES]
    y = df[TARGET]
    
    # 2. Train-Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Scale Features using the previously fitted and saved scaler
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Fitted scaler not found at: {SCALER_PATH}. Run pipeline first.")
        
    scaler = joblib.load(SCALER_PATH)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 4. Train Random Forest
    print("\nTraining Random Forest Regressor...")
    rf_start = time.time()
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - rf_start
    rf_preds = rf.predict(X_test)
    
    # 5. Train XGBoost
    print("Training XGBoost Regressor...")
    xgb_start = time.time()
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - xgb_start
    xgb_preds = xgb.predict(X_test)
    
    # 6. Evaluation helper
    def evaluate(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2
        
    rf_mae, rf_rmse, rf_r2 = evaluate(y_test, rf_preds)
    xgb_mae, xgb_rmse, xgb_r2 = evaluate(y_test, xgb_preds)
    
    print("\n======================= MODEL TRAIN & EVALUATION COMPARISON =======================")
    print(f"{'Algorithm':<20} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10} | {'Train Time (s)':<15}")
    print("-" * 75)
    print(f"{'Random Forest':<20} | {rf_mae:<10.4f} | {rf_rmse:<10.4f} | {rf_r2:<10.4f} | {rf_time:<15.4f}")
    print(f"{'XGBoost':<20} | {xgb_mae:<10.4f} | {xgb_rmse:<10.4f} | {xgb_r2:<10.4f} | {xgb_time:<15.4f}")
    print("===================================================================================")
    
    # 7. Select and save the best model
    if xgb_r2 > rf_r2:
        best_model = xgb
        best_name = "XGBoost"
        best_r2 = xgb_r2
    else:
        best_model = rf
        best_name = "Random Forest"
        best_r2 = rf_r2
        
    print(f"\nSelect model with highest validation R2: {best_name} (R2={best_r2:.4f})")
    
    # Save best model to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best performing model to: {MODEL_PATH}")
    
    return best_model

if __name__ == "__main__":
    train_and_compare_models()
