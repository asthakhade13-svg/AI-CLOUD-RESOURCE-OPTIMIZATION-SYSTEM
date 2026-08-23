# test_safety.py

import pytest
import os
import json
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, MagicMock
from src.safety_layer import RiskEngine, PolicyEngine, SafetyControlLayer

@pytest.fixture(scope="module", autouse=True)
def mock_ml_service_safety():
    mock_decision_resp = {
        "decision": {
            "timestamp": "2026-08-23T21:00:00",
            "model_version": "v1.2.0-champion",
            "input_summary": {"current_replicas": 4, "predicted_traffic": 250.0},
            "prediction": {"recommended_servers": 6, "confidence": 0.92},
            "action": "SCALE_UP_2",
            "risk_score": "LOW",
            "policy_result": "PASSED",
            "reason": "Scale from 4 to 6 replicas",
            "operator": "SYSTEM_AI",
            "execution_result": "SIMULATED_SUCCESS"
        },
        "risk_value": 0.15,
        "operating_mode": "SIMULATION",
        "emergency_stop": False,
        "is_approved": True
    }
    mock_explain_resp = {
        "human_explanation": "Scale from 4 to 6 replicas",
        "contributing_factors": {},
        "shap_attributions": {"predicted_traffic": 1.2, "current_cpu": 0.5}
    }
    mock_simulate_resp = {
        "traffic_increase_pct": 50.0,
        "current_configuration": {"replicas": 4},
        "predicted_configuration": {"replicas": 6},
        "cost_difference": 0.24,
        "sla_impact": "HEALTHY",
        "carbon_impact_ghr": 3.0,
        "risk": "LOW",
        "recommended_action": "SCALE_UP_2"
    }
    mock_risk_resp = {"status": "LOW", "score": 0.15, "metrics": {"sla_risk_pct": 10.0, "prediction_uncertainty": 0.05}}
    mock_policies_resp = {"min_replicas": 2, "max_replicas": 15, "scaling_step_limit": 3}
    mock_audit_resp = [mock_decision_resp["decision"]]

    def mock_get(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "/risk" in url:
            resp.json.return_value = mock_risk_resp
        elif "/policies" in url:
            resp.json.return_value = mock_policies_resp
        elif "/audit" in url:
            resp.json.return_value = mock_audit_resp
        else:
            resp.status_code = 404
        return resp

    def mock_post(url, *args, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "/emergency-stop" in url:
            resp.json.return_value = {"emergency_stop": True}
        elif "/emergency-start" in url:
            resp.json.return_value = {"emergency_stop": False}
        elif "/decision/explain" in url:
            resp.json.return_value = mock_explain_resp
        elif "/decision/simulate" in url:
            resp.json.return_value = mock_simulate_resp
        elif "/decision" in url:
            resp.json.return_value = mock_decision_resp
        else:
            resp.status_code = 404
        return resp

    with patch("requests.get", side_effect=mock_get), patch("requests.post", side_effect=mock_post):
        yield

def test_risk_engine_calculations():
    # Test low risk
    risk, val = RiskEngine.calculate_risk(
        prediction_uncertainty=0.05,
        sla_risk_pct=10.0,
        anomaly_severity="NONE",
        workload_volatility=0.10,
        infrastructure_health="healthy",
        model_confidence=0.95,
        recent_scaling_freq=0
    )
    assert risk == "LOW"
    assert val < 0.25

    # Test critical risk
    risk, val = RiskEngine.calculate_risk(
        prediction_uncertainty=0.45,
        sla_risk_pct=90.0,
        anomaly_severity="CRITICAL",
        workload_volatility=0.80,
        infrastructure_health="unhealthy",
        model_confidence=0.45,
        recent_scaling_freq=5
    )
    assert risk == "CRITICAL"
    assert val >= 0.75

def test_policy_engine_validation():
    policies = {
        "min_replicas": 2,
        "max_replicas": 10,
        "scaling_step_limit": 3,
        "cooldown_seconds": 60,
        "budget_limit_hourly": 2.0,
        "emergency_restrictions": False
    }
    
    # Test valid approval
    approved, msg = PolicyEngine.validate_action(
        current_replicas=4,
        target_replicas=6,
        policies=policies,
        cooldown_elapsed=True,
        is_emergency=False
    )
    assert approved is True
    assert "APPROVED" in msg

    # Test step limit violation
    approved, msg = PolicyEngine.validate_action(
        current_replicas=4,
        target_replicas=9,
        policies=policies,
        cooldown_elapsed=True,
        is_emergency=False
    )
    assert approved is False
    assert "step size" in msg

    # Test emergency stop block
    approved, msg = PolicyEngine.validate_action(
        current_replicas=4,
        target_replicas=6,
        policies=policies,
        cooldown_elapsed=True,
        is_emergency=True
    )
    assert approved is False
    assert "Emergency Stop" in msg

def test_safety_orchestrator():
    control = SafetyControlLayer()
    res = control.process_decision(
        current_replicas=4,
        predicted_traffic=300.0,
        current_cpu=75.0,
        current_latency=120.0
    )
    assert "decision" in res
    assert "risk_value" in res
    assert res["decision"]["prediction"]["recommended_servers"] == 6
    assert "Scale from 4 to 6 replicas" in res["decision"]["reason"]

def test_what_if_scenarios():
    control = SafetyControlLayer()
    res = control.generate_what_if(traffic_increase_pct=50.0, current_replicas=4)
    assert res["traffic_increase_pct"] == 50.0
    assert res["predicted_configuration"]["replicas"] == 6
    assert res["cost_difference"] > 0.0
    assert res["sla_impact"] == "HEALTHY"

def test_gateway_safety_endpoints():
    client = TestClient(app)
    
    # Enable emergency stop
    resp = client.post("/emergency-stop")
    assert resp.status_code == 200
    assert resp.json()["emergency_stop"] is True

    # Disable emergency stop
    resp = client.post("/emergency-start")
    assert resp.status_code == 200
    assert resp.json()["emergency_stop"] is False

    # Get policies
    resp = client.get("/policies")
    assert resp.status_code == 200
    assert "min_replicas" in resp.json()

    # Get active calculated risk stats
    resp = client.get("/risk")
    assert resp.status_code == 200
    assert "status" in resp.json()
    assert "score" in resp.json()

    # Process decision via API
    resp = client.post("/decision", json={
        "current_replicas": 4,
        "predicted_traffic": 250.0,
        "current_cpu": 65.0,
        "current_latency": 150.0
    })
    assert resp.status_code == 200
    assert "decision" in resp.json()

    # Get decision explanation
    resp = client.post("/decision/explain", json={
        "current_replicas": 4,
        "predicted_traffic": 350.0,
        "current_cpu": 82.0,
        "current_latency": 190.0
    })
    assert resp.status_code == 200
    assert "human_explanation" in resp.json()
    assert "shap_attributions" in resp.json()

    # Get audit timeline log
    resp = client.get("/audit")
    assert resp.status_code == 200
    assert len(resp.json()) > 0
