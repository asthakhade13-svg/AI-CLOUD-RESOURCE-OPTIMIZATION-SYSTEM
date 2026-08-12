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
FEATURES_LIST_PATH = "artifacts/features_list.pkl"
MODEL_PATH = "artifacts/cloud_resource_optimization_model.pkl"
TARGET = "required_servers"

def print_feature_importance_analysis(rf_model, xgb_model, feature_names):
    """
    Extracts and prints the top 15 features by importance for both models.
    Provides explanations for why these features are useful.
    """
    print("\n======================= FEATURE IMPORTANCE ANALYSIS =======================")
    
    # Extract importances
    rf_importances = rf_model.feature_importances_
    xgb_importances = xgb_model.feature_importances_
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Random Forest": rf_importances,
        "XGBoost": xgb_importances
    })
    
    # Sort by Random Forest importance
    rf_sorted = importance_df.sort_values(by="Random Forest", ascending=False).head(15)
    
    print(f"{'Rank':<4} | {'Feature Name':<30} | {'RF MDI':<10} | {'XGBoost Weight':<14}")
    print("-" * 68)
    for rank, (_, row) in enumerate(rf_sorted.iterrows(), 1):
        print(f"{rank:<4} | {row['Feature']:<30} | {row['Random Forest']:<10.4f} | {row['XGBoost']:<14.4f}")
    
    print("\n* Key Insights on Feature Utility for Predictive Scaling:")
    print("1. Lag Features (e.g. active_users_lag_1, cpu_usage_lag_5):")
    print("   -> Captures autocorrelation. A high CPU usage in the immediate past (t-1) strongly indicates high resource demands at time t.")
    print("2. Rolling Features (e.g. cpu_usage_moving_average_30, active_users_moving_average_5):")
    print("   -> Smooths out transient workload spikes. Moving averages allow the model to understand the longer-term trend rather than micro-bursts.")
    print("3. Trend Features (e.g. cpu_usage_growth_rate):")
    print("   -> Represents acceleration/velocity of workloads. A positive growth rate means server scaling must be proactive before CPU hits limits.")
    print("4. Cyclical Time Features (e.g. sin_hour, cos_day_of_week):")
    print("   -> Allows the model to predict resource scale based on daily/weekly cycles (e.g. lunch hour peaks, night drops).")
    print("===========================================================================")

def train_and_compare_models():
    """
    Trains Random Forest and XGBoost regression models on the engineered feature dataset.
    Performs feature importance analysis, outputs performance metrics, and saves the best model.
    """
    print("\nLoading dataset and features list for training...")
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at: {CLEANED_DATA_PATH}. Run pipeline first.")
    if not os.path.exists(FEATURES_LIST_PATH):
        raise FileNotFoundError(f"Features list not found at: {FEATURES_LIST_PATH}. Run pipeline first.")
        
    df = pd.read_csv(CLEANED_DATA_PATH)
    features = joblib.load(FEATURES_LIST_PATH)
    
    # 1. Feature Target Split
    X = df[features]
    y = df[TARGET]
    
    # 2. Train-Test Split (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Scale Features using the saved standardizer
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Fitted scaler not found at: {SCALER_PATH}")
        
    scaler = joblib.load(SCALER_PATH)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 4. Train Random Forest
    print("\nTraining Random Forest Regressor on engineered features...")
    rf_start = time.time()
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - rf_start
    rf_preds = rf.predict(X_test)
    
    # 5. Train XGBoost
    print("Training XGBoost Regressor on engineered features...")
    xgb_start = time.time()
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - xgb_start
    xgb_preds = xgb.predict(X_test)
    
    # 6. Evaluation metrics
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
    
    # 7. Select best model
    if xgb_r2 > rf_r2:
        best_model = xgb
        best_name = "XGBoost"
        best_r2 = xgb_r2
    else:
        best_model = rf
        best_name = "Random Forest"
        best_r2 = rf_r2
        
    print(f"\nSelecting model with highest validation R2: {best_name} (R2={best_r2:.4f})")
    
    # Save best model to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best performing model to: {MODEL_PATH}")
    
    # 8. Print Feature Importance Analysis
    print_feature_importance_analysis(rf, xgb, features)
    
    return best_model

if __name__ == "__main__":
    train_and_compare_models()
