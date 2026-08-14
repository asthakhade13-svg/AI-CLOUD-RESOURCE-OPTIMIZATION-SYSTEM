import pytest
from fastapi.testclient import TestClient
from main import app, MODEL_PATH
import os

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "docs_url" in data
    assert "model_loaded" in data

def test_health_check(client):
    response = client.get("/health")
    if os.path.exists(MODEL_PATH):
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_type" in data
    else:
        assert response.status_code == 503

def test_predict_success(client):
    payload = {
        "cpu_usage": 70.0,
        "memory_usage": 65.0,
        "network_in": 100.0,
        "network_out": 150.0,
        "network_traffic": 250.0,
        "disk_read": 50.0,
        "disk_write": 25.0,
        "active_users": 150,
        "request_rate": 375.0,
        "response_time": 180.0,
        "error_rate": 0.01,
        "current_servers": 3,
        "server_cost": 0.365
    }
    response = client.post("/predict", json=payload)
    if os.path.exists(MODEL_PATH) and os.path.exists("artifacts/scaler.pkl"):
        assert response.status_code == 200
        data = response.json()
        assert "recommended_servers" in data
        assert "predicted_servers" in data
        assert "scaling_action" in data
        assert "reasoning" in data
        assert "predicted_cpu_5min" in data
        assert "predicted_cpu_15min" in data
        assert "forecasts" in data
        assert data["current_servers"] == 3
        assert data["recommended_servers"] >= 1
        assert data["scaling_action"] in ["SCALE UP", "SCALE DOWN", "NO ACTION NEEDED"]
    else:
        # If assets aren't trained/loaded, backend will raise 503
        assert response.status_code == 503

def test_predict_invalid_input(client):
    # cpu_usage must be <= 100
    payload = {
        "cpu_usage": 150.0,  # Invalid (>100)
        "memory_usage": 65.0,
        "network_in": 100.0,
        "network_out": 150.0,
        "network_traffic": 250.0,
        "disk_read": 50.0,
        "disk_write": 25.0,
        "active_users": 150,
        "request_rate": 375.0,
        "response_time": 180.0,
        "error_rate": 0.01,
        "current_servers": 3,
        "server_cost": 0.365
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_features(client):
    response = client.get("/features")
    if os.path.exists(MODEL_PATH):
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert "importances" in data
        assert "cpu_usage" in data["importances"]
    else:
        assert response.status_code == 503
