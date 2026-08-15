import pytest
from src.sla import evaluate_sla

def test_sla_healthy():
    res = evaluate_sla(
        response_time=120.0,
        error_rate=0.05,
        cpu_usage=55.0,
        memory_usage=60.0,
        target_response_time=200.0,
        maximum_error_rate=1.0,
        minimum_availability=99.0
    )
    
    assert res["status"] == "HEALTHY"
    # Max ratio is memory: 60/100 = 0.6 and response time: 120/200 = 0.6
    assert res["risk_score"] == pytest.approx(0.6)
    assert res["availability"] == 99.95

def test_sla_violation_response_time():
    res = evaluate_sla(
        response_time=250.0,  # exceeds 200
        error_rate=0.05,
        cpu_usage=55.0,
        memory_usage=60.0,
        target_response_time=200.0,
        maximum_error_rate=1.0,
        minimum_availability=99.0
    )
    
    assert res["status"] == "VIOLATED"
    assert res["risk_score"] == 1.0
    assert "Response time violated" in res["reason"]

def test_sla_violation_availability():
    res = evaluate_sla(
        response_time=120.0,
        error_rate=1.5,  # availability = 98.5% < 99.0%
        cpu_usage=55.0,
        memory_usage=60.0,
        target_response_time=200.0,
        maximum_error_rate=2.0,
        minimum_availability=99.0
    )
    
    assert res["status"] == "VIOLATED"
    assert res["risk_score"] == 1.0
    assert "Availability violated" in res["reason"]

def test_sla_at_risk_cpu():
    res = evaluate_sla(
        response_time=120.0,
        error_rate=0.05,
        cpu_usage=82.0,  # exceeds 80% saturation threshold
        memory_usage=60.0,
        target_response_time=200.0,
        maximum_error_rate=1.0,
        minimum_availability=99.0
    )
    
    assert res["status"] == "AT_RISK"
    assert res["risk_score"] == pytest.approx(0.82)
    assert "CPU resource saturated" in res["reason"]
