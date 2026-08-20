# test_simulation.py

import pytest
import numpy as np
from simulation.workload import WorkloadGenerator
from simulation.infrastructure import ClusterSimulator
from simulation.failures import ChaosInjector
from simulation.metrics import MetricsTracker
from simulation.environment import DigitalTwinEnv
from simulation.scenarios import WhatIfAnalyzer
from simulation.experiments import ExperimentSuite

def test_workload_generator():
    gen = WorkloadGenerator()
    patterns = [
        {"type": "constant", "params": {"users": 150.0, "requests_per_user": 2.0}},
        {"type": "linear", "params": {"slope": 5.0, "base": 50.0, "requests_per_user": 2.0}}
    ]
    
    # At step 10: constant: 150 users. linear: 50 + 5*10 = 100 users. Total = 250 users.
    w = gen.generate_step(10, patterns)
    assert w["users"] == 250
    assert w["requests"] == 250 * 2.0
    assert w["network"] == w["requests"] * 0.6

def test_cluster_simulator():
    sim = ClusterSimulator(config={"startup_delay": 2, "shutdown_delay": 1})
    sim.reset(5)
    assert sim.current_replicas == 5
    assert sim.active_replicas == 5
    
    # Scale up target to 7 (2 steps delay)
    diff, msg = sim.scale_to(7, current_step=0)
    assert diff == 2
    assert sim.current_replicas == 7
    assert sim.active_replicas == 5 # remains 5 during boot
    
    # Advance state machine step 1
    sim.update_states(current_step=1)
    assert sim.active_replicas == 5
    
    # Advance state machine step 2 (pods become active)
    sim.update_states(current_step=2)
    assert sim.active_replicas == 7

def test_chaos_injector():
    injector = ChaosInjector()
    injector.inject_cpu_leak(True)
    injector.inject_network_degradation(25.0)
    
    cpu, mem, lat, err, pods = injector.apply_failures(
        cpu=50.0, memory=40.0, latency=100.0, error_rate=0.0, active_pods=4
    )
    
    assert cpu == 100.0
    assert err >= 20.0
    assert lat >= 1200.0

def test_metrics_tracker():
    tracker = MetricsTracker(target_latency=200.0, unit_cost=0.50)
    
    # Normal performance
    lat, err, sla = tracker.compute_performance(cpu=50.0, memory=50.0, requests=1000.0, active_pods=5)
    assert lat < 200.0
    assert err == 0.0
    assert sla == "HEALTHY"
    
    # High CPU latency queuing
    lat_high, _, sla_high = tracker.compute_performance(cpu=95.0, memory=50.0, requests=1000.0, active_pods=5)
    assert lat_high > 200.0
    assert sla_high == "VIOLATED"
    
    # Cost calculation: 5 pods for 5 mins (5/60 hours)
    # cost = 5 * 0.50 * 5/60 = 2.5 * 1/12 = 0.208
    cost = tracker.calculate_cost(5)
    assert round(cost, 3) == 0.208

def test_digital_twin_env():
    env = DigitalTwinEnv(config={"max_steps": 10})
    patterns = [{"type": "constant", "params": {"users": 100.0}}]
    env.set_workload_patterns(patterns)
    
    obs = env.reset(5)
    assert len(obs) == 15
    assert env.cluster_simulator.current_replicas == 5
    
    next_obs, cost, done, log = env.step("HPA")
    assert len(next_obs) == 15
    assert not done
    assert "cpu" in log
    assert "latency" in log

def test_what_if_analyzer():
    analyzer = WhatIfAnalyzer(simulator_config={"max_steps": 20})
    config = {
        "policy_name": "THRESHOLD",
        "initial_replicas": 4,
        "traffic_multiplier": 1.5,
        "failures": [{"step": 5, "type": "pod_crash", "value": 1}]
    }
    
    res = analyzer.run_custom_scenario(config)
    assert "summary" in res
    assert "history" in res
    assert len(res["history"]) == 20
    assert res["summary"]["total_cost"] > 0

def test_experiment_suite():
    suite = ExperimentSuite(simulator_config={"max_steps": 10})
    config = {
        "initial_replicas": 5,
        "traffic_multiplier": 1.0,
        "workload_patterns": [{"type": "constant", "params": {"users": 100.0}}]
    }
    
    results = suite.run_policy_comparison(config, ppo_agent=None, model_loaded=False)
    assert len(results) == 5
    policies = [r["policy"] for r in results]
    assert "STATIC" in policies
    assert "HPA" in policies
    assert "RL_PPO" in policies
