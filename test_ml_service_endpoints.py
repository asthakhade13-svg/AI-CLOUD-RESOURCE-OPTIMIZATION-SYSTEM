import pytest
from fastapi.testclient import TestClient
from ml_service.main import app

@pytest.fixture(scope="module")
def ml_client():
    with TestClient(app) as c:
        yield c

def test_ml_service_health(ml_client):
    resp = ml_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_ml_service_risk(ml_client):
    # Verify risk status works and does not raise NameError for safety_layer
    resp = ml_client.get("/risk")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "score" in data
    assert "metrics" in data

def test_ml_service_policies(ml_client):
    resp = ml_client.get("/policies")
    assert resp.status_code == 200
    assert "min_replicas" in resp.json()
    assert "max_replicas" in resp.json()

def test_ml_service_emergency_stop_flow(ml_client):
    # Test emergency stop deactivates / activates global safety layer
    resp = ml_client.post("/emergency-stop")
    assert resp.status_code == 200
    assert resp.json()["emergency_stop"] is True

    resp = ml_client.post("/emergency-start")
    assert resp.status_code == 200
    assert resp.json()["emergency_stop"] is False

def test_ml_service_operating_mode(ml_client):
    resp = ml_client.post("/mode/set", json={"mode": "APPROVAL"})
    assert resp.status_code == 200
    assert resp.json()["operating_mode"] == "APPROVAL"

    resp = ml_client.post("/mode/set", json={"mode": "SIMULATION"})
    assert resp.status_code == 200
    assert resp.json()["operating_mode"] == "SIMULATION"

    # Invalid mode
    resp = ml_client.post("/mode/set", json={"mode": "INVALID"})
    assert resp.status_code == 400
