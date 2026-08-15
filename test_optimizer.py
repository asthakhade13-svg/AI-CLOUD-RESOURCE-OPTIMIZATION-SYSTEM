import pytest
from src.optimizer import optimize_capacity_cost

def test_optimizer_basic_savings():
    # Predicted required = 5, Current servers = 7. Server cost per hour = 0.50
    # The optimizer should recommend 5 servers because it has no deficit and no over-provisioning penalty.
    # Estimated savings: (7 - 5) * 0.50 * 24 = $24.00/day
    opt = optimize_capacity_cost(
        predicted_required_servers=5.0,
        current_servers=7,
        server_cost_per_hour=0.50,
        min_servers=1,
        max_servers=10,
        sla_penalty_weight=5.0,
        overprovisioning_weight=0.5
    )
    
    assert opt["recommended_servers"] == 5
    assert opt["hourly_cost"] == 2.50
    assert opt["estimated_daily_cost"] == 60.00
    assert opt["estimated_monthly_cost"] == 1800.00
    assert opt["estimated_savings_daily"] == 24.00
    assert opt["sla_status"] == "SATISFIED"
    assert "daily_savings" not in opt  # Confirming our key name is estimated_savings_daily

def test_optimizer_high_sla_priority():
    # Predicted = 5.4.
    # Candidate 5: infra_cost = 5 * 0.50 = 2.50. Deficit = 0.4. sla_penalty = 0.4 * 10 * 0.50 = 2.00. Total = 4.50
    # Candidate 6: infra_cost = 6 * 0.50 = 3.00. Surplus = 0.6. overprovisioning_penalty = 0.6 * 0.5 * 0.50 = 0.15. Total = 3.15
    # The optimizer should select 6 because 3.15 < 4.50.
    opt = optimize_capacity_cost(
        predicted_required_servers=5.4,
        current_servers=5,
        server_cost_per_hour=0.50,
        min_servers=1,
        max_servers=10,
        sla_penalty_weight=10.0,
        overprovisioning_weight=0.5
    )
    
    assert opt["recommended_servers"] == 6
    assert opt["sla_status"] == "SATISFIED"

def test_optimizer_budget_constrained():
    # If SLA penalty weight is very low (e.g. 0.1) and overprovisioning penalty weight is very high (e.g. 5.0)
    # The optimizer will rather under-provision and violate SLA to avoid overprovisioning costs.
    # Predicted = 5.4.
    # Candidate 5: infra_cost = 2.50. Deficit = 0.4. sla_penalty = 0.4 * 0.1 * 0.50 = 0.02. Total = 2.52
    # Candidate 6: infra_cost = 3.00. Surplus = 0.6. over_penalty = 0.6 * 5.0 * 0.50 = 1.50. Total = 4.50
    # The optimizer should select 5 (violating SLA) because 2.52 < 4.50.
    opt = optimize_capacity_cost(
        predicted_required_servers=5.4,
        current_servers=5,
        server_cost_per_hour=0.50,
        min_servers=5,
        max_servers=10,
        sla_penalty_weight=0.1,
        overprovisioning_weight=5.0
    )
    
    assert opt["recommended_servers"] == 5
    assert opt["sla_status"] == "VIOLATED"
