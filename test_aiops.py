# test_aiops.py

import pytest
import os
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from src.aiops_engine import AIOpsEngine

def test_aiops_engine_initialization():
    engine = AIOpsEngine()
    assert "Frontend" in engine.graph
    assert "API" in engine.graph
    assert len(engine.graph["Frontend"].dependencies) == 1
    assert engine.graph["Frontend"].health == "healthy"
    assert engine.graph["API"].health == "healthy"

def test_aiops_engine_fault_injection_cpu():
    engine = AIOpsEngine()
    incident = engine.inject_fault("CPU_SATURATION")
    
    assert incident["status"] == "active"
    assert incident["severity"] == "CRITICAL"
    assert "API" in incident["affected_services"]
    assert "Frontend" in incident["affected_services"]
    assert incident["recommended_action"] == "SCALE_SERVICE_API"
    assert engine.graph["API"].cpu == 99.8
    assert engine.graph["API"].health == "unhealthy"
    assert engine.graph["Frontend"].health == "degraded"

def test_aiops_engine_fault_injection_memory():
    engine = AIOpsEngine()
    incident = engine.inject_fault("MEMORY_EXHAUSTION")
    
    assert incident["status"] == "active"
    assert incident["severity"] == "CRITICAL"
    assert "Authentication" in incident["affected_services"]
    assert incident["recommended_action"] == "RESTART_UNHEALTHY_POD"
    assert engine.graph["Authentication"].memory == 98.5
    assert engine.graph["Authentication"].health == "unhealthy"

def test_aiops_engine_fault_resolution():
    engine = AIOpsEngine()
    incident = engine.inject_fault("DATABASE_SLOWDOWN")
    incident_id = incident["id"]
    
    # Assert nodes are degraded
    assert engine.graph["Database"].health == "degraded"
    
    # Resolve
    success = engine.resolve_incident(incident_id)
    assert success is True
    assert engine.graph["Database"].health == "healthy"

@pytest.fixture(scope="module", autouse=True)
def mock_ml_service_aiops():
    # Mock AIOps ML Service responses for endpoints
    mock_graph = {
        "Frontend": {"name": "Frontend", "health": "healthy", "cpu": 12.0, "latency": 45.0, "error_rate": 0.0, "traffic": 100.0},
        "API": {"name": "API", "health": "healthy", "cpu": 15.0, "latency": 50.0, "error_rate": 0.0, "traffic": 100.0}
    }
    mock_incidents = [
        {
            "id": "INC-12345",
            "timestamp": "2026-08-23T21:00:00",
            "incident": "API Latency Spike",
            "root_cause": "API CPU Saturation",
            "confidence": 0.88,
            "severity": "CRITICAL",
            "sla_impact_pct": 85.0,
            "estimated_recovery": "5 min",
            "affected_services": ["API", "Frontend"],
            "evidence": ["API CPU at 99%"],
            "recommended_action": "SCALE_SERVICE_API",
            "status": "active"
        }
    ]
    mock_experiments = {
        "scenarios": [
            {"policy": "AI + AIOps Self-Healing", "detection_time_sec": 8.0, "root_cause_id_time_sec": 12.0, "recovery_time_sec": 35.0, "sla_violations": 0, "cost_multiplier": 0.98, "scaling_events": 3}
        ],
        "evidence": ["Evidence 1"]
    }

    def mock_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "/aiops/graph" in url:
            resp.json.return_value = mock_graph
        elif "/aiops/incidents" in url:
            resp.json.return_value = mock_incidents
        elif "/aiops/experiments" in url:
            resp.json.return_value = mock_experiments
        else:
            resp.status_code = 404
        return resp

    def mock_post(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "/aiops/fault_inject" in url:
            resp.json.return_value = mock_incidents[0]
        elif "/aiops/resolve" in url:
            resp.json.return_value = {"success": True, "message": "Resolved"}
        else:
            resp.status_code = 404
        return resp

    with patch("requests.get", side_effect=mock_get), patch("requests.post", side_effect=mock_post):
        yield

def test_gateway_aiops_graph():
    client = TestClient(app)
    resp = client.get("/aiops/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert "Frontend" in data
    assert data["Frontend"]["health"] == "healthy"

def test_gateway_aiops_incidents():
    client = TestClient(app)
    resp = client.get("/aiops/incidents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "INC-12345"

def test_gateway_aiops_fault_inject():
    client = TestClient(app)
    resp = client.post("/aiops/fault_inject", json={"fault_type": "CPU_SATURATION"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["root_cause"] == "API CPU Saturation"

def test_gateway_aiops_resolve():
    client = TestClient(app)
    resp = client.post("/aiops/resolve", json={"incident_id": "INC-12345"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True

def test_gateway_aiops_experiments():
    client = TestClient(app)
    resp = client.get("/aiops/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert "scenarios" in data
