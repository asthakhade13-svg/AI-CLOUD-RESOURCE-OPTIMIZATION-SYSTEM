import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

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

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "cloud_resource_cpu_usage_ratio" in response.text
    assert "cloud_resource_scaling_actions_total" in response.text

def test_predict_endpoint_success(client):
    payload = {
        "cpu_usage": 78,
        "memory_usage": 82,
        "network_traffic": 390,
        "active_users": 290,
        "current_servers": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_servers" in data
    assert "recommended_servers" in data
    assert "action" in data
    assert data["current_servers"] == 5
    assert data["action"] in ["SCALE_UP", "SCALE_DOWN", "NO_ACTION"]

def test_forecast_endpoint_success(client):
    payload = {
        "cpu_usage": 70.0,
        "memory_usage": 65.0,
        "network_traffic": 250.0,
        "active_users": 150,
        "current_servers": 3
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "forecasts" in data
    assert "15min" in data["forecasts"]

def test_autoscale_endpoint_success(client):
    payload = {
        "cpu_usage": 85.0,
        "current_servers": 5,
        "predicted_servers": 6.5,
        "recommended_servers": 7,
        "sla_status": "AT_RISK",
        "anomaly_severity": "LOW"
    }
    response = client.post("/autoscale", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] in ["SCALE_UP", "SCALE_DOWN", "NO_ACTION"]
    assert "reason" in data

def test_anomaly_endpoint_success(client):
    payload = {
        "cpu_usage": 95.0,
        "memory_usage": 88.0,
        "active_users": 1200,
        "current_servers": 5
    }
    response = client.post("/anomaly", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "severity" in data
    assert "affected_metrics" in data

def test_optimize_endpoint_success(client):
    payload = {
        "predicted_required_servers": 6.2,
        "current_servers": 5,
        "server_cost_per_hour": 0.60
    }
    response = client.post("/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommended_servers" in data
    assert "hourly_cost" in data
    assert "estimated_savings" in data

def test_root_dashboard_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Cloud Resource Optimization Console" in response.text

def test_k8s_status_endpoint(client):
    response = client.get("/k8s/status")
    assert response.status_code == 200
    data = response.json()
    assert "current_replicas" in data
    assert "predicted_replicas" in data
    assert "hpa_keda_status" in data
    assert "scaling_events" in data
    assert "mode" in data
    assert "method" in data

