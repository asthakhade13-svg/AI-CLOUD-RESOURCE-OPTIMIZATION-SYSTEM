import os
from typing import List

class Settings:
    # App Settings
    APP_TITLE: str = "AI Cloud Resource Optimization API"
    APP_VERSION: str = "10.0.0"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS Origins configuration
    CORS_ORIGINS: List[str] = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
    
    # ML & Asset Paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "artifacts/cloud_resource_optimization_model.pkl")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "artifacts/scaler.pkl")
    CLEANED_DATA_PATH: str = os.getenv("CLEANED_DATA_PATH", "data/cleaned_workload.csv")
    FEATURES_LIST_PATH: str = os.getenv("FEATURES_LIST_PATH", "artifacts/features_list.pkl")
    
    # Forecasting Models Paths
    FORECASTER_5MIN_PATH: str = os.getenv("FORECASTER_5MIN_PATH", "artifacts/forecaster_5min.pkl")
    FORECASTER_10MIN_PATH: str = os.getenv("FORECASTER_10MIN_PATH", "artifacts/forecaster_10min.pkl")
    FORECASTER_15MIN_PATH: str = os.getenv("FORECASTER_15MIN_PATH", "artifacts/forecaster_15min.pkl")
    FORECASTING_FEATURES_PATH: str = os.getenv("FORECASTING_FEATURES_PATH", "artifacts/forecasting_features_list.pkl")
    
    # Anomaly Detection Paths
    ANOMALY_MODEL_PATH: str = os.getenv("ANOMALY_MODEL_PATH", "artifacts/anomaly_detector.pkl")
    ANOMALY_SCALER_PATH: str = os.getenv("ANOMALY_SCALER_PATH", "artifacts/anomaly_scaler.pkl")
    ANOMALY_FEATURES_PATH: str = os.getenv("ANOMALY_FEATURES_PATH", "artifacts/anomaly_features_list.pkl")

settings = Settings()
