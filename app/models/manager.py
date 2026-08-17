import os
import joblib
import shap
from app.config import settings
from app.utils.logging import logger

class ModelManager:
    """
    Stateful manager class responsible for loading all machine learning models,
    scalers, and explainer assets once on startup, caching them in-memory.
    """
    def __init__(self):
        self.capacity_model = None
        self.capacity_scaler = None
        self.capacity_features = None
        
        self.forecaster_5min = None
        self.forecaster_10min = None
        self.forecaster_15min = None
        self.forecasting_features = None
        
        self.anomaly_detector = None
        self.anomaly_scaler = None
        self.anomaly_features = None
        
        self.shap_explainer = None
        self.assets_loaded = False

    def load_all_assets(self):
        """Loads all serialized pickled artifacts into memory."""
        logger.info("Initializing ML asset loading routine on startup...")
        
        try:
            # 1. Load Capacity Predictor (Stage 2)
            if os.path.exists(settings.MODEL_PATH):
                self.capacity_model = joblib.load(settings.MODEL_PATH)
                logger.info(f"Loaded capacity model: {settings.MODEL_PATH}")
            else:
                logger.warning(f"Capacity model path '{settings.MODEL_PATH}' not found.")
                
            # 2. Load StandardScaler
            if os.path.exists(settings.SCALER_PATH):
                self.capacity_scaler = joblib.load(settings.SCALER_PATH)
                logger.info(f"Loaded StandardScaler: {settings.SCALER_PATH}")
            else:
                logger.warning(f"StandardScaler path '{settings.SCALER_PATH}' not found.")
                
            # 3. Load Capacity Features List
            if os.path.exists(settings.FEATURES_LIST_PATH):
                self.capacity_features = joblib.load(settings.FEATURES_LIST_PATH)
                logger.info(f"Loaded capacity features list: {settings.FEATURES_LIST_PATH}")
                
            # 4. Load Workload Forecasters (Stage 1)
            if os.path.exists(settings.FORECASTER_5MIN_PATH):
                self.forecaster_5min = joblib.load(settings.FORECASTER_5MIN_PATH)
                logger.info(f"Loaded 5-min workload forecaster: {settings.FORECASTER_5MIN_PATH}")
            if os.path.exists(settings.FORECASTER_10MIN_PATH):
                self.forecaster_10min = joblib.load(settings.FORECASTER_10MIN_PATH)
                logger.info(f"Loaded 10-min workload forecaster: {settings.FORECASTER_10MIN_PATH}")
            if os.path.exists(settings.FORECASTER_15MIN_PATH):
                self.forecaster_15min = joblib.load(settings.FORECASTER_15MIN_PATH)
                logger.info(f"Loaded 15-min workload forecaster: {settings.FORECASTER_15MIN_PATH}")
            if os.path.exists(settings.FORECASTING_FEATURES_PATH):
                self.forecasting_features = joblib.load(settings.FORECASTING_FEATURES_PATH)
                logger.info(f"Loaded forecasting features list: {settings.FORECASTING_FEATURES_PATH}")
                
            # 5. Load Anomaly Detection Assets
            if os.path.exists(settings.ANOMALY_MODEL_PATH):
                self.anomaly_detector = joblib.load(settings.ANOMALY_MODEL_PATH)
                logger.info(f"Loaded anomaly detector model: {settings.ANOMALY_MODEL_PATH}")
            if os.path.exists(settings.ANOMALY_SCALER_PATH):
                self.anomaly_scaler = joblib.load(settings.ANOMALY_SCALER_PATH)
                logger.info(f"Loaded anomaly scaler: {settings.ANOMALY_SCALER_PATH}")
            if os.path.exists(settings.ANOMALY_FEATURES_PATH):
                self.anomaly_features = joblib.load(settings.ANOMALY_FEATURES_PATH)
                logger.info(f"Loaded anomaly features list: {settings.ANOMALY_FEATURES_PATH}")
                
            # 6. Initialize SHAP Explainer
            if self.capacity_model is not None:
                self.shap_explainer = shap.TreeExplainer(self.capacity_model)
                logger.info("SHAP TreeExplainer initialized successfully.")
                
            self.assets_loaded = True
            logger.info("All machine learning assets successfully cached in-memory.")
            
        except Exception as e:
            logger.error(f"Failed to load ML assets on startup: {str(e)}")
            self.assets_loaded = False
            raise e

model_manager = ModelManager()
