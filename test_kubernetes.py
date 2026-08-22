# test_kubernetes.py

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module", autouse=True)
def mock_requests_post():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "predicted_servers": 6.0,
        "recommended_servers": 6,
        "uncertainty_std": 0.5,
        "is_anomaly": False,
        "anomaly_score": 0.08,
        "severity": "LOW",
        "affected_metrics": [],
        "recommendation": "System normal.",
        "shap_explanation": "Predicted workload is stable.",
        "shap_contributions": {
            "CPU utilization": 0.02,
            "Memory utilization": 0.01,
            "Active users": 0.05
        },
        "forecasts": {
            "5min": {"cpu_usage": 45.0, "memory_usage": 50.0, "network_traffic": 120.0, "active_users": 180.0, "request_rate": 200.0, "response_time": 100.0, "latency": 120.0, "error_rate": 0.0},
            "10min": {"cpu_usage": 45.0, "memory_usage": 50.0, "network_traffic": 120.0, "active_users": 180.0, "request_rate": 200.0, "response_time": 100.0, "latency": 120.0, "error_rate": 0.0},
            "15min": {"cpu_usage": 45.0, "memory_usage": 50.0, "network_traffic": 120.0, "active_users": 180.0, "request_rate": 200.0, "response_time": 100.0, "latency": 120.0, "error_rate": 0.0}
        }
    }
    with patch("requests.post", return_value=mock_response) as mock:
        yield mock

def test_metrics_expose_recommended_servers():
    client_tc = TestClient(app)
    
    # Trigger a predict call to update the recommended servers count
    predict_payload = {
        "cpu_usage": 78.0,
        "memory_usage": 62.0,
        "network_traffic": 800.0,
        "active_users": 1500,
        "current_servers": 4
    }
    resp = client_tc.post("/predict", json=predict_payload)
    assert resp.status_code == 200
    pred_data = resp.json()
    recommended = pred_data["recommended_servers"]
    
    # Query Prometheus metrics endpoint
    metrics_resp = client_tc.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_text = metrics_resp.text
    
    # Ensure recommended servers metric exists and matches the returned value
    assert "cloud_resource_recommended_servers_count" in metrics_text
    
    # Find the specific line containing the metric value
    found_metric = False
    for line in metrics_text.splitlines():
        if line.startswith("cloud_resource_recommended_servers_count"):
            val = float(line.split()[-1])
            assert val == float(recommended)
            found_metric = True
            break
            
    assert found_metric is True

def test_k8s_status_endpoint_structure():
    client_tc = TestClient(app)
    resp = client_tc.get("/k8s/status")
    assert resp.status_code == 200
    data = resp.json()
    
    # Check that required keys are present
    assert "current_replicas" in data
    assert "predicted_replicas" in data
    assert "desired_replicas" in data
    assert "hpa_keda_status" in data
    assert "scaling_events" in data
    assert "fallback_active" in data
    assert "mode" in data
    assert "method" in data
