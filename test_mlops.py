# test_mlops.py

import pytest
import os
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module", autouse=True)
def mock_ml_service_mlops():
    # Mock ML service responses for gateway calls to decouple tests from background service
    mock_status = {
        "active_version": "v1",
        "algorithm": "Random Forest",
        "created_at": "2026-08-22T14:48:00",
        "age_days": 1,
        "last_retrained": "Never",
        "rollback_events": [],
        "champion_model": {
            "version": "v1",
            "algorithm": "Random Forest",
            "metrics": {"mae": 0.082, "rmse": 0.125, "r2": 0.94}
        }
    }
    mock_metrics = {"mae": 0.085, "rmse": 0.13, "r2": 0.93, "samples_evaluated": 15}
    mock_drift = {
        "drift_detected": True,
        "affected_features": ["cpu_usage"],
        "severity": "LOW",
        "drift_score": 0.12,
        "features": {
            "cpu_usage": {"ks_statistic": 0.15, "p_value": 0.02, "drift_detected": True}
        }
    }
    mock_retrain = {
        "success": True,
        "promoted": True,
        "version": "v2",
        "metrics": {"mae": 0.079, "rmse": 0.118, "r2": 0.95},
        "reason": "Challenger approved and promoted."
    }
    mock_rollback = {
        "success": True,
        "rolled_back_to": "v1",
        "reason": "Successfully rolled back production model."
    }

    def mock_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "/model/status" in url:
            resp.json.return_value = mock_status
        elif "/model/metrics" in url:
            resp.json.return_value = mock_metrics
        elif "/model/drift" in url:
            resp.json.return_value = mock_drift
        else:
            resp.status_code = 404
        return resp

    def mock_post(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "/model/retrain" in url:
            resp.json.return_value = mock_retrain
        elif "/model/rollback" in url:
            resp.json.return_value = mock_rollback
        else:
            resp.status_code = 404
        return resp

    with patch("requests.get", side_effect=mock_get), patch("requests.post", side_effect=mock_post):
        yield

def test_gateway_model_status():
    client = TestClient(app)
    resp = client.get("/model/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_version"] == "v1"
    assert "algorithm" in data
    assert "age_days" in data
    assert "champion_model" in data

def test_gateway_model_metrics():
    client = TestClient(app)
    resp = client.get("/model/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "mae" in data
    assert "rmse" in data
    assert "r2" in data

def test_gateway_model_drift():
    client = TestClient(app)
    resp = client.get("/model/drift")
    assert resp.status_code == 200
    data = resp.json()
    assert "drift_detected" in data
    assert "affected_features" in data
    assert "severity" in data

def test_gateway_model_retrain():
    client = TestClient(app)
    resp = client.post("/model/retrain", json={"force": True, "authorized": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["promoted"] is True
    assert "version" in data

def test_gateway_model_rollback():
    client = TestClient(app)
    resp = client.post("/model/rollback")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "rolled_back_to" in data
