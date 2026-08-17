import pandas as pd
import numpy as np
from app.models.manager import model_manager
from src.forecasting import FORECAST_METRICS

def forecast_workloads(history_df: pd.DataFrame) -> dict:
    """
    Statefully predicts telemetry metrics (CPU, Memory, Users, Traffic, Requests, response_time)
    for 5, 10, and 15 minutes ahead based on history window.
    """
    if len(history_df) < 6:
        raise ValueError("history_df must contain at least 6 observations to compute lag features.")
        
    history = history_df.tail(6).copy()
    features = model_manager.forecasting_features
    
    if features is None:
        raise RuntimeError("Forecasting feature list is not loaded in model manager.")
        
    # Construct lag features
    row_data = {}
    for col in FORECAST_METRICS:
        for lag in range(1, 7):
            row_data[f"{col}_lag_{lag}"] = [history.iloc[-lag][col]]
            
    # Time properties
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
    
    # Predict
    pred_5 = model_manager.forecaster_5min.predict(X_pred)[0]
    pred_10 = model_manager.forecaster_10min.predict(X_pred)[0]
    pred_15 = model_manager.forecaster_15min.predict(X_pred)[0]
    
    return {
        "5min": {col: float(val) for col, val in zip(FORECAST_METRICS, pred_5)},
        "10min": {col: float(val) for col, val in zip(FORECAST_METRICS, pred_10)},
        "15min": {col: float(val) for col, val in zip(FORECAST_METRICS, pred_15)}
    }
