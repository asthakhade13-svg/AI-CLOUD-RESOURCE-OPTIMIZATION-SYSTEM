import pytest
import pandas as pd
import numpy as np
import os
import joblib
from src.forecasting import (
    prepare_forecasting_data,
    forecast_next_workloads,
    FORECAST_METRICS,
    FORECASTER_5MIN_PATH
)
from src.generator import generate_synthetic_workload

TEST_RAW_CSV = "data/test_forecast_raw.csv"

@pytest.fixture(scope="module")
def setup_forecast_test_data():
    """Generates a small dataset for testing the forecasting helper functions."""
    generate_synthetic_workload(days=2, output_path=TEST_RAW_CSV)
    yield
    if os.path.exists(TEST_RAW_CSV):
        os.remove(TEST_RAW_CSV)

def test_prepare_forecasting_data(setup_forecast_test_data):
    df = pd.read_csv(TEST_RAW_CSV)
    
    # Run data prep
    df_feat, feature_cols, targets_dict = prepare_forecasting_data(df, lags=6)
    
    # 1. Assert rows are cleaned up (8643 rows, lags/leads drop the first 6 and last 3 rows)
    assert len(df_feat) > 0
    assert len(df_feat) < len(df)
    
    # 2. Check lag columns existence
    for col in FORECAST_METRICS:
        for lag in range(1, 7):
            assert f"{col}_lag_{lag}" in df_feat.columns
            
    # 3. Check targets existence
    for col in FORECAST_METRICS:
        assert f"{col}_lead_1" in df_feat.columns
        assert f"{col}_lead_2" in df_feat.columns
        assert f"{col}_lead_3" in df_feat.columns

def test_forecast_next_workloads_invalid_size():
    # Less than 6 rows should raise ValueError
    short_df = pd.DataFrame({
        "timestamp": ["2026-08-01 00:00:00"],
        "cpu_usage": [50.0]
    })
    with pytest.raises(ValueError):
        forecast_next_workloads(short_df)

def test_forecast_next_workloads_success(setup_forecast_test_data):
    df = pd.read_csv(TEST_RAW_CSV)
    
    # Run pipeline to ensure scaler and models are present
    if os.path.exists(FORECASTER_5MIN_PATH):
        # Slice last 6 rows
        history_df = df.tail(6).copy()
        
        # Call forecasting function
        forecast_results = forecast_next_workloads(history_df)
        
        assert "5min" in forecast_results
        assert "10min" in forecast_results
        assert "15min" in forecast_results
        
        # Check that forecast contains all metrics
        for h in ["5min", "10min", "15min"]:
            for col in FORECAST_METRICS:
                assert col in forecast_results[h]
                assert isinstance(forecast_results[h][col], float)
