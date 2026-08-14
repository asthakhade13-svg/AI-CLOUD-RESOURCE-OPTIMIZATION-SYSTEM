import pandas as pd
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# File paths
CLEANED_DATA_PATH = "data/cleaned_workload.csv"
FORECASTER_5MIN_PATH = "artifacts/forecaster_5min.pkl"
FORECASTER_10MIN_PATH = "artifacts/forecaster_10min.pkl"
FORECASTER_15MIN_PATH = "artifacts/forecaster_15min.pkl"

# Telemetry metrics we want to forecast
FORECAST_METRICS = [
    "cpu_usage", "memory_usage", "network_traffic", 
    "active_users", "request_rate", "response_time"
]

def prepare_forecasting_data(df: pd.DataFrame, lags=6):
    """
    Creates historical lags (1 to 6 steps) for features, and shifted targets 
    representing workloads in 5 minutes (+1 step), 10 minutes (+2 steps), and 15 minutes (+3 steps).
    Ensures zero future data leakage.
    """
    df = df.copy()
    
    # Ensure sorted chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # 1. Create Lag Features (history from t-1 down to t-lags)
    feature_cols = []
    for col in FORECAST_METRICS:
        for lag in range(1, lags + 1):
            lag_name = f"{col}_lag_{lag}"
            df[lag_name] = df[col].shift(lag)
            feature_cols.append(lag_name)
            
    # Also keep time characteristics of current time t
    # hour, day_of_week, and cyclical encodings
    time_cols = ["hour", "day_of_week", "sin_hour", "cos_hour", "sin_day_of_week", "cos_day_of_week", "is_weekend"]
    for col in time_cols:
        if col in df.columns:
            feature_cols.append(col)
            
    # 2. Create target columns for horizons: +1 (5m), +2 (10m), +3 (15m)
    # Shifting by negative numbers looks into the future
    targets = {
        "5min": [f"{col}_lead_1" for col in FORECAST_METRICS],
        "10min": [f"{col}_lead_2" for col in FORECAST_METRICS],
        "15min": [f"{col}_lead_3" for col in FORECAST_METRICS]
    }
    
    for col in FORECAST_METRICS:
        df[f"{col}_lead_1"] = df[col].shift(-1)
        df[f"{col}_lead_2"] = df[col].shift(-2)
        df[f"{col}_lead_3"] = df[col].shift(-3)
        
    # Drop rows with NaNs caused by lagging/leading
    # Maximum lag is 6, maximum lead is 3
    df_clean = df.dropna().reset_index(drop=True)
    
    return df_clean, feature_cols, targets

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE) handling zeros with epsilon."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    epsilon = 1e-5
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def evaluate_forecast(y_true, y_pred, metric_names):
    """Computes MAE, RMSE, R2, and MAPE for each forecasted variable."""
    results = {}
    for i, col in enumerate(metric_names):
        true = y_true[:, i]
        pred = y_pred[:, i]
        mae = mean_absolute_error(true, pred)
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        mape = calculate_mape(true, pred)
        
        results[col] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        }
    return results

def train_and_compare_forecasters():
    """
    Loads clean workload data, creates forecasting sets, compares RF, XGBoost, and GBDT,
    saves the best forecaster model, and generates evaluation charts.
    """
    print("\nStarting Workload Forecasting Model Comparison...")
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at: {CLEANED_DATA_PATH}. Run pipeline first.")
        
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # 1. Prepare data
    df_feat, feature_cols, targets_dict = prepare_forecasting_data(df)
    print(f"Total processed dataset shape for forecasting: {df_feat.shape}")
    print(f"Number of lag and time features: {len(feature_cols)}")
    
    # 2. Chronological Split (Train: 70%, Val: 15%, Test: 15%)
    n = len(df_feat)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df_feat.iloc[:train_end]
    val_df = df_feat.iloc[train_end:val_end]
    test_df = df_feat.iloc[val_end:]
    
    print(f"Time-based split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    X_train, y_train_all = train_df[feature_cols], train_df
    X_val, y_val_all = val_df[feature_cols], val_df
    X_test, y_test_all = test_df[feature_cols], test_df
    
    # We will train separate model sets for each horizon: 5min, 10min, 15min
    horizons = ["5min", "10min", "15min"]
    best_models = {}
    
    # Dictionary to store performance reports
    performance_reports = {}
    
    for h in horizons:
        print(f"\n================ TRAINING HORIZON: {h} ================= ")
        target_cols = targets_dict[h]
        
        y_train = y_train_all[target_cols].values
        y_val = y_val_all[target_cols].values
        y_test = y_test_all[target_cols].values
        
        # Define candidate architectures
        # MultiOutputRegressor is required since we predict 6 targets simultaneously
        models = {
            "Random Forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42, n_jobs=-1),
            "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
        }
        
        best_r2 = -999.0
        best_name = None
        best_model_obj = None
        best_test_preds = None
        
        for name, model in models.items():
            start_t = time.time()
            model.fit(X_train, y_train)
            train_t = time.time() - start_t
            
            # Predict
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)
            
            # Evaluate on Test
            eval_res = evaluate_forecast(y_test, test_preds, FORECAST_METRICS)
            
            # Use mean R2 score across all 6 metrics to pick the best model
            mean_r2 = np.mean([res["R2"] for res in eval_res.values()])
            mean_mae = np.mean([res["MAE"] for res in eval_res.values()])
            
            print(f"[{name}] Test Mean R2: {mean_r2:.4f} | Mean MAE: {mean_mae:.4f} | Time: {train_t:.2f}s")
            
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_name = name
                best_model_obj = model
                best_test_preds = test_preds
                
        print(f"Winner for {h} horizon: {best_name} (Mean R2: {best_r2:.4f})")
        best_models[h] = (best_model_obj, best_name, best_test_preds, y_test)
        
        # Save model
        save_path = FORECASTER_5MIN_PATH if h == "5min" else (FORECASTER_10MIN_PATH if h == "10min" else FORECASTER_15MIN_PATH)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(best_model_obj, save_path)
        print(f"Saved {h} forecaster to {save_path}")
        
    # Save the feature list for forecasting
    joblib.dump(feature_cols, "artifacts/forecasting_features_list.pkl")
    
    # 3. Create Visualizations (using the best models selected)
    generate_forecasting_plots(best_models, test_df)

def generate_forecasting_plots(best_models, test_df):
    """Creates and saves the 3 requested forecasting visualization plots."""
    plots_dir = "data/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # We'll use CPU usage for plotting
    cpu_index = FORECAST_METRICS.index("cpu_usage")
    
    # A. Actual vs Predicted Workload Plot
    # We will display the 5-minute horizon prediction
    model_5m, name_5m, preds_5m, actual_5m = best_models["5min"]
    
    plt.figure(figsize=(12, 5))
    # Plot first 150 points for visual clarity
    plt.plot(actual_5m[:150, cpu_index], label="Actual CPU Usage", color="black", linewidth=2)
    plt.plot(preds_5m[:150, cpu_index], label=f"Predicted CPU ({name_5m})", color="#1f77b4", linestyle="--", alpha=0.9)
    plt.title("Workload Forecasting Evaluation: Actual vs. 5-Min Predicted CPU Usage", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Sample Timeline (5-Min Steps)", fontsize=11)
    plt.ylabel("CPU Usage (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "forecast_actual_vs_predicted.png"), dpi=150)
    plt.close()
    print("-> Saved forecast_actual_vs_predicted.png")
    
    # B. Forecast Horizon Progression Plot
    # Shows predictions at +5, +10, and +15 min for a slice of time
    preds_10m = best_models["10min"][2]
    preds_15m = best_models["15min"][2]
    
    plt.figure(figsize=(12, 5))
    slice_idx = 100
    timeline = np.arange(50)
    
    plt.plot(timeline, actual_5m[slice_idx:slice_idx+50, cpu_index], label="Actual CPU", color="black", linewidth=2.5)
    plt.plot(timeline, preds_5m[slice_idx:slice_idx+50, cpu_index], label="5-Min Forecast (t+1)", color="#2ca02c", linestyle="-.", alpha=0.8)
    plt.plot(timeline, preds_10m[slice_idx:slice_idx+50, cpu_index], label="10-Min Forecast (t+2)", color="#ff7f0e", linestyle=":", alpha=0.8)
    plt.plot(timeline, preds_15m[slice_idx:slice_idx+50, cpu_index], label="15-Min Forecast (t+3)", color="#d62728", linestyle="--", alpha=0.8)
    
    plt.title("Multi-Horizon Workload Forecasting Progression (5, 10, 15 Minutes)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Test Timeline Slice (5-Min Steps)", fontsize=11)
    plt.ylabel("CPU Usage (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "forecast_horizon_progression.png"), dpi=150)
    plt.close()
    print("-> Saved forecast_horizon_progression.png")
    
    # C. Prediction Errors Distribution Plot
    # Residuals for 5m, 10m, 15m predictions
    err_5m = actual_5m[:, cpu_index] - preds_5m[:, cpu_index]
    err_10m = best_models["10min"][3][:, cpu_index] - preds_10m[:, cpu_index]
    err_15m = best_models["15min"][3][:, cpu_index] - preds_15m[:, cpu_index]
    
    plt.figure(figsize=(10, 5))
    plt.hist(err_5m, bins=40, alpha=0.5, label="5-Min Error Residuals", color="#2ca02c")
    plt.hist(err_10m, bins=40, alpha=0.5, label="10-Min Error Residuals", color="#ff7f0e")
    plt.hist(err_15m, bins=40, alpha=0.5, label="15-Min Error Residuals", color="#d62728")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1.5)
    
    plt.title("Workload Forecasting Error Residual Distributions", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Prediction Error (Actual - Predicted %)", fontsize=11)
    plt.ylabel("Frequency Count", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "forecast_prediction_errors.png"), dpi=150)
    plt.close()
    print("-> Saved forecast_prediction_errors.png")

def forecast_next_workloads(history_df: pd.DataFrame):
    """
    Reusable prediction function. Takes a DataFrame of the last 6 observations
    (minimum needed for lags) and returns a dictionary with predictions for 5, 10, and 15 mins.
    """
    if len(history_df) < 6:
        raise ValueError("history_df must contain at least 6 chronological observations to compute lag features.")
        
    # Align and order columns
    history = history_df.tail(6).copy()
    
    # Check features list
    feature_list_path = "artifacts/forecasting_features_list.pkl"
    if not os.path.exists(feature_list_path):
        raise FileNotFoundError(f"Forecasting feature names file not found at: {feature_list_path}. Run training first.")
    features = joblib.load(feature_list_path)
    
    # Construct lag features from history
    row_data = {}
    for col in FORECAST_METRICS:
        # Lags 1 to 6
        for lag in range(1, 7):
            # history is sorted, so index -lag retrieves correct lag
            row_data[f"{col}_lag_{lag}"] = [history.iloc[-lag][col]]
            
    # Extract time features of the latest observation (at index -1)
    latest_ts = pd.to_datetime(history.iloc[-1]["timestamp"])
    hour = latest_ts.hour
    day_of_week = latest_ts.dayofweek
    
    row_data["hour"] = [hour]
    row_data["day_of_week"] = [day_of_week]
    row_data["day_of_month"] = [latest_ts.day]
    row_data["is_weekend"] = [1 if day_of_week >= 5 else 0]
    row_data["sin_hour"] = [np.sin(2 * np.pi * hour / 24.0)]
    row_data["cos_hour"] = [np.cos(2 * np.pi * hour / 24.0)]
    row_data["sin_day_of_week"] = [np.sin(2 * np.pi * day_of_week / 7.0)]
    row_data["cos_day_of_week"] = [np.cos(2 * np.pi * day_of_week / 7.0)]
    
    X_pred = pd.DataFrame(row_data)[features]
    
    # Load model files
    model_5m = joblib.load(FORECASTER_5MIN_PATH)
    model_10m = joblib.load(FORECASTER_10MIN_PATH)
    model_15m = joblib.load(FORECASTER_15MIN_PATH)
    
    # Predict (each outputs a 2D array of shape [1, 6])
    pred_5 = model_5m.predict(X_pred)[0]
    pred_10 = model_10m.predict(X_pred)[0]
    pred_15 = model_15m.predict(X_pred)[0]
    
    # Convert predictions back to dictionary structures mapping to metrics
    forecast_results = {
        "5min": {col: float(val) for col, val in zip(FORECAST_METRICS, pred_5)},
        "10min": {col: float(val) for col, val in zip(FORECAST_METRICS, pred_10)},
        "15min": {col: float(val) for col, val in zip(FORECAST_METRICS, pred_15)}
    }
    
    return forecast_results

if __name__ == "__main__":
    train_and_compare_forecasters()
