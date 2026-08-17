import pytest
import os
from fastapi.testclient import TestClient
from app.main import app, model_manager
from app.config import settings

@pytest.fixture(scope="module")
def client():
    # Trigger model manager load assets
    if not model_manager.assets_loaded:
        model_manager.load_all_assets()
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    if model_manager.assets_loaded:
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["assets_loaded"] is True
    else:
        assert response.status_code == 503

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predict_requests" in data
    assert "uptime_seconds" in data
    assert data["model_version"] == settings.APP_VERSION

def test_predict_endpoint_success(client):
    payload = {
        "cpu_usage": 78,
        "memory_usage": 82,
        "network_traffic": 390,
        "active_users": 290,
        "current_servers": 5
    }
    response = client.post("/predict", json=payload)
    if model_manager.assets_loaded:
        assert response.status_code == 200
        data = response.json()
        assert "predicted_servers" in data
        assert "recommended_servers" in data
        assert "action" in data
        assert data["current_servers"] == 5
        assert data["action"] in ["SCALE_UP", "SCALE_DOWN", "NO_ACTION"]
    else:
        assert response.status_code == 503

def test_forecast_endpoint_success(client):
    payload = {
        "cpu_usage": 70.0,
        "memory_usage": 65.0,
        "network_traffic": 250.0,
        "active_users": 150,
        "current_servers": 3
    }
    response = client.post("/forecast", json=payload)
    if model_manager.assets_loaded:
        assert response.status_code == 200
        data = response.json()
        assert "forecasts" in data
        assert "15min" in data["forecasts"]
    else:
        assert response.status_code == 500

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
    if model_manager.assets_loaded:
        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly" in data
        assert "severity" in data
        assert "affected_metrics" in data
    else:
        assert response.status_code == 500

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
