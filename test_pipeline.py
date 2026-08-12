import pytest
import pandas as pd
import numpy as np
import os
import joblib
from src.validation import validate_schema, validate_data_ranges, validate_dataset
from src.pipeline import (
    run_preprocessing_pipeline, 
    preprocess_single_record, 
    CLEANED_DATA_PATH, 
    SCALER_PATH,
    FEATURES
)
from src.generator import generate_synthetic_workload

TEST_RAW_CSV = "data/test_raw.csv"

@pytest.fixture(scope="module")
def setup_test_data():
    """Generates a small dataset and runs pipeline to fit the scaler."""
    # Ensure raw file exists
    generate_synthetic_workload(days=2, output_path=TEST_RAW_CSV)
    # Run pipeline
    run_preprocessing_pipeline(TEST_RAW_CSV, output_csv_path="data/test_cleaned.csv")
    yield
    # Cleanup test files
    if os.path.exists(TEST_RAW_CSV):
        os.remove(TEST_RAW_CSV)
    if os.path.exists("data/test_cleaned.csv"):
        os.remove("data/test_cleaned.csv")

def test_validation_schema_missing_columns():
    df_missing = pd.DataFrame({"cpu_usage": [50.0], "memory_usage": [60.0]})
    errors = validate_schema(df_missing)
    assert len(errors) > 0
    assert any("Missing required column" in err for err in errors)

def test_validation_ranges_out_of_bounds():
    df_invalid = pd.DataFrame({
        "timestamp": ["2026-08-01 00:00:00"],
        "cpu_usage": [150.0], # Out of bounds (>100)
        "memory_usage": [65.0],
        "network_in": [50.0],
        "network_out": [50.0],
        "network_traffic": [100.0],
        "disk_read": [10.0],
        "disk_write": [5.0],
        "active_users": [100],
        "request_rate": [250.0],
        "response_time": [120.0],
        "error_rate": [0.0],
        "current_servers": [2],
        "server_cost": [0.24],
        "required_servers": [2]
    })
    errors = validate_data_ranges(df_invalid)
    assert len(errors) > 0
    assert any("cpu_usage" in err for err in errors)

def test_pipeline_deduplication_and_null_imputation(setup_test_data):
    # Load cleaned data
    df = pd.read_csv("data/test_cleaned.csv")
    
    # 1. Assert no NaNs are present in features
    assert df[FEATURES].isnull().sum().sum() == 0
    
    # 2. Check schema and range validity
    assert validate_dataset(df, raise_exception=False) is True

def test_preprocess_single_record(setup_test_data):
    scaler = joblib.load(SCALER_PATH)
    
    valid_record = {
        "cpu_usage": 80.0,
        "memory_usage": 70.0,
        "network_in": 120.0,
        "network_out": 200.0,
        "network_traffic": 320.0,
        "disk_read": 150.0,
        "disk_write": 80.0,
        "active_users": 300,
        "request_rate": 750.0,
        "response_time": 220.0,
        "error_rate": 0.0,
        "current_servers": 4,
        "server_cost": 0.50
    }
    
    scaled_vector = preprocess_single_record(valid_record, scaler)
    assert scaled_vector.shape == (1, len(FEATURES))
    assert isinstance(scaled_vector, np.ndarray)
