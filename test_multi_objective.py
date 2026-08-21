# test_multi_objective.py

import pytest
from src.multi_objective_optimizer import (
    evaluate_config,
    is_dominated,
    solve_pareto_grid_search,
    run_multi_objective_optimization
)

def test_evaluate_config_metrics():
    workload = {"request_rate": 600.0, "active_users": 1500, "predicted_required_servers": 3.0}
    current_config = {"replicas": 4, "instance_type": "m5.large", "cpu_alloc": 2.0, "mem_alloc": 8.0, "region": "us-east-1"}
    
    # Evaluate 3x m5.large in us-east-1
    res = evaluate_config(
        replicas=3,
        instance_type="m5.large",
        cpu_alloc=2.0,
        mem_alloc=8.0,
        region="us-east-1",
        workload=workload,
        current_config=current_config
    )
    
    # 3 replicas * 0.096 price = 0.288
    assert abs(res["cost"] - 0.288) < 1e-5
    assert res["replicas"] == 3
    assert res["instance_type"] == "m5.large"
    assert res["region"] == "us-east-1"
    assert "energy" in res
    assert "carbon" in res

def test_carbon_intensities():
    workload = {"request_rate": 300.0, "active_users": 500, "predicted_required_servers": 2.0}
    current = {"replicas": 2, "instance_type": "t3.medium", "cpu_alloc": 2.0, "mem_alloc": 4.0, "region": "us-east-1"}
    
    # N. Virginia (us-east-1) vs Stockholm (eu-west-6)
    res_va = evaluate_config(2, "t3.medium", 2.0, 4.0, "us-east-1", workload, current)
    res_st = evaluate_config(2, "t3.medium", 2.0, 4.0, "eu-west-6", workload, current)
    
    # Carbon emissions in Va should be way higher than Stockholm (380 g vs 20 g per kWh)
    assert res_va["carbon"] > res_st["carbon"]

def test_domination_logic():
    # Candidate A: Cost: 0.1, Latency: 100ms, SLA: 0, Carbon: 10
    cand_a = {
        "cost": 0.1, "latency": 100.0, "sla_violated": 0.0, "energy": 0.05, 
        "carbon": 10.0, "overprovisioning": 1.0, "underprovisioning": 0.0, "instability": 0.0
    }
    
    # Candidate B is worse in every aspect (Cost: 0.2, Latency: 150ms, Carbon: 20)
    cand_b = {
        "cost": 0.2, "latency": 150.0, "sla_violated": 0.0, "energy": 0.10, 
        "carbon": 20.0, "overprovisioning": 2.0, "underprovisioning": 0.0, "instability": 1.0
    }
    
    # Candidate C is better in cost, but worse in latency (Cost: 0.05, Latency: 250ms)
    cand_c = {
        "cost": 0.05, "latency": 250.0, "sla_violated": 1.0, "energy": 0.03, 
        "carbon": 5.0, "overprovisioning": 0.0, "underprovisioning": 1.0, "instability": 0.0
    }
    
    # A dominates B
    assert is_dominated(cand_b, cand_a) is True
    # B does not dominate A
    assert is_dominated(cand_a, cand_b) is False
    # A does not dominate C (since C has lower cost)
    assert is_dominated(cand_c, cand_a) is False
    # C does not dominate A (since A has lower latency)
    assert is_dominated(cand_a, cand_c) is False

def test_optimizer_priorities():
    workload = {"request_rate": 800.0, "active_users": 2000, "predicted_required_servers": 4.0}
    current = {"replicas": 4, "instance_type": "m5.large", "cpu_alloc": 2.0, "mem_alloc": 8.0, "region": "us-east-1"}
    
    # 1. Cost-critical optimizer priorities
    opt_cost = run_multi_objective_optimization(
        workload=workload,
        current_config=current,
        weights={"cost": 2.0, "latency": 0.1, "sla": 0.1, "carbon": 0.1},
        constraints={"min_replicas": 2, "max_replicas": 6}
    )
    
    # 2. Performance-critical optimizer priorities
    opt_perf = run_multi_objective_optimization(
        workload=workload,
        current_config=current,
        weights={"cost": 0.1, "latency": 2.0, "sla": 2.0, "carbon": 0.1},
        constraints={"min_replicas": 2, "max_replicas": 6}
    )
    
    # Performance-optimal setup should have higher cost or lower latency than cost-optimal
    assert opt_perf["estimated_latency"] <= opt_cost["estimated_latency"]
    assert opt_perf["optimization_score"] > 0
