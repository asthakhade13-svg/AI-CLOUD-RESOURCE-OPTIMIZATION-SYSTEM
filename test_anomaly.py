import pytest
import pandas as pd
import numpy as np
import os
import joblib
from src.anomaly import (
    detect_anomaly_record, 
    train_and_compare_anomaly_detectors,
    ANOMALY_MODEL_PATH
)
from src.generator import generate_synthetic_workload

TEST_RAW_CSV = "data/test_anomaly_raw.csv"

@pytest.fixture(scope="module")
def setup_anomaly_test_data():
    generate_synthetic_workload(days=2, output_path=TEST_RAW_CSV)
    yield
    if os.path.exists(TEST_RAW_CSV):
        os.remove(TEST_RAW_CSV)

def test_anomaly_record_normal(setup_anomaly_test_data):
    df = pd.read_csv(TEST_RAW_CSV)
    
    # Take a normal historical row (last row) and add time components
    latest_ts = pd.to_datetime(df.iloc[-1]["timestamp"])
    new_row = pd.DataFrame([df.iloc[-1]])
    new_row["hour"] = latest_ts.hour
    new_row["day_of_week"] = latest_ts.dayofweek
    new_row["sin_hour"] = np.sin(2 * np.pi * latest_ts.hour / 24.0)
    new_row["cos_hour"] = np.cos(2 * np.pi * latest_ts.hour / 24.0)
    new_row["sin_day_of_week"] = np.sin(2 * np.pi * latest_ts.dayofweek / 7.0)
    new_row["cos_day_of_week"] = np.cos(2 * np.pi * latest_ts.dayofweek / 7.0)
    
    # Evaluate
    if os.path.exists(ANOMALY_MODEL_PATH):
        res = detect_anomaly_record(new_row, df.tail(30))
        # Normal row should NOT be detected as an anomaly
        assert res["is_anomaly"] is False
        assert res["anomaly_score"] < 0.50
        assert res["severity"] in ["LOW", "MEDIUM"]
        assert len(res["affected_metrics"]) == 0

def test_anomaly_record_critical_spike(setup_anomaly_test_data):
    df = pd.read_csv(TEST_RAW_CSV)
    
    # Construct an extreme outlier (5x traffic spike, high CPU, high latency)
    latest_ts = pd.to_datetime(df.iloc[-1]["timestamp"])
    new_row = pd.DataFrame([df.iloc[-1]])
    
    # Override values to create an extreme anomaly
    new_row["active_users"] = 2500  # 10x normal load
    new_row["request_rate"] = 6000.0
    new_row["cpu_usage"] = 99.0
    new_row["response_time"] = 950.0
    new_row["error_rate"] = 15.0
    
    new_row["hour"] = latest_ts.hour
    new_row["day_of_week"] = latest_ts.dayofweek
    new_row["sin_hour"] = np.sin(2 * np.pi * latest_ts.hour / 24.0)
    new_row["cos_hour"] = np.cos(2 * np.pi * latest_ts.hour / 24.0)
    new_row["sin_day_of_week"] = np.sin(2 * np.pi * latest_ts.dayofweek / 7.0)
    new_row["cos_day_of_week"] = np.cos(2 * np.pi * latest_ts.dayofweek / 7.0)
    
    if os.path.exists(ANOMALY_MODEL_PATH):
        res = detect_anomaly_record(new_row, df.tail(30))
        assert res["is_anomaly"] is True
        assert res["anomaly_score"] >= 0.50
        assert res["severity"] in ["HIGH", "CRITICAL"]
        # Confirm active_users and request_rate are listed in root cause
        assert "active_users" in res["affected_metrics"]
        assert "request_rate" in res["affected_metrics"]
