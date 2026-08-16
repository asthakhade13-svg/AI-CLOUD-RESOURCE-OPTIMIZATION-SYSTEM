import pytest
import pandas as pd
import numpy as np
import os
import joblib
import shap
from src.explainability import (
    explain_prediction_shap,
    MODEL_PATH,
    SCALER_PATH,
    FEATURES_LIST_PATH,
    CLEANED_DATA_PATH
)

def test_shap_explanation_basic():
    # Verify that local SHAP calculations work correctly using a mocked setup
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(CLEANED_DATA_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_LIST_PATH)
        df = pd.read_csv(CLEANED_DATA_PATH)
        
        # Instantiate TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Take a single scaled row
        X_scaled = scaler.transform(df[features].tail(1))
        
        # Compute explanation
        res = explain_prediction_shap(
            explainer=explainer,
            X_scaled_record=X_scaled,
            feature_names=features,
            recommended_servers=5
        )
        
        assert "shap_explanation" in res
        assert "category_contributions" in res
        assert "top_feature_contributions" in res
        
        # Check that we categorized features
        assert "CPU utilization" in res["category_contributions"]
        assert "Active users" in res["category_contributions"]
        assert "Memory utilization" in res["category_contributions"]
        
        # Verify it generates a non-empty human readable string
        assert len(res["shap_explanation"]) > 0
        assert "recommends 5 servers" in res["shap_explanation"]
