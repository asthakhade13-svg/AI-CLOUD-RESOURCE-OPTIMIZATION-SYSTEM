# test_rl.py

import pytest
import numpy as np
import torch
from rl.actions import idx_to_action, idx_to_step, action_to_step, get_action_count
from rl.state import get_observation, STATE_DIM
from rl.reward import calculate_reward
from rl.environment import CloudAutoscalingEnv
from rl.safety import SafetyValidator
from rl.agent import PPOAgent

def test_actions_mapping():
    assert get_action_count() == 5
    assert idx_to_action(2) == "NO_ACTION"
    assert idx_to_action(4) == "SCALE_UP_2"
    assert idx_to_step(0) == -2
    assert idx_to_step(3) == 1
    assert action_to_step("SCALE_DOWN_1") == -1

def test_state_normalization():
    metrics = {
        "cpu_usage": 80.0,
        "memory_usage": 60.0,
        "network_traffic": 1000.0,
        "active_users": 5000,
        "request_rate": 2500.0,
        "response_time": 150.0,
        "error_rate": 0.5,
        "current_servers": 4,
        "predicted_workload": 85.0,
        "predicted_required_servers": 6,
        "hourly_cost": 2.4,
        "sla_status": "AT_RISK",
        "is_anomaly": True,
        "prev_step": -1,
        "hour": 18.0
    }
    
    obs = get_observation(metrics)
    assert len(obs) == STATE_DIM
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    
    # Check bounds
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
    
    # Check specific normalizations
    assert obs[0] == 0.8  # CPU 80/100
    assert obs[7] == 0.2  # current servers 4/20
    assert obs[11] == 0.5 # SLA status AT_RISK
    assert obs[12] == 1.0 # Anomaly is True -> 1.0

def test_reward_computation():
    metrics = {
        "cpu_usage": 50.0,
        "response_time": 100.0,
        "target_response_time": 200.0,
        "sla_status": "HEALTHY",
        "current_servers": 5,
        "max_servers": 20,
        "predicted_required_servers": 5,
        "error_rate": 0.0
    }
    
    # Step change 0 -> no thrashing
    reward_no_action = calculate_reward(metrics, action_step=0)
    # Step change 1 -> thrashing penalty applied
    reward_action = calculate_reward(metrics, action_step=1)
    
    assert reward_no_action < 0.0
    assert reward_action < reward_no_action  # Thrashing should increase penalty (lower reward)
    
    # Test SLA violation penalty
    metrics_violated = metrics.copy()
    metrics_violated["sla_status"] = "VIOLATED"
    reward_violated = calculate_reward(metrics_violated, action_step=0)
    assert reward_violated < reward_no_action  # SLA penalty should drop reward significantly

def test_environment_loop():
    env = CloudAutoscalingEnv()
    obs = env.reset()
    
    assert len(obs) == STATE_DIM
    assert env.current_step == 0
    assert env.current_servers == 5
    
    # Take a step (NO_ACTION)
    next_obs, reward, done, info = env.step(2)
    assert len(next_obs) == STATE_DIM
    assert isinstance(reward, float)
    assert not done
    assert env.current_step == 1
    assert "cpu_usage" in info
    assert "response_time" in info

def test_safety_validator():
    validator = SafetyValidator(config={"min_servers": 2, "max_servers": 10, "max_scale_up_step": 2})
    metrics = {
        "cpu_usage": 50.0,
        "response_time": 100.0,
        "target_response_time": 200.0,
        "sla_status": "HEALTHY"
    }
    
    # 1. Normal validation
    rec, step, reason = validator.validate_action(current_replicas=5, proposed_step=1, metrics=metrics)
    assert rec == 6
    assert step == 1
    assert "Approved" in reason
    
    # 2. Clamping min limits
    validator.last_scale_time = 0.0
    rec, step, reason = validator.validate_action(current_replicas=2, proposed_step=-1, metrics=metrics)
    assert rec == 2
    assert step == 0
    assert "Boundary" in reason
    
    # 3. Clamping max step limits
    validator.last_scale_time = 0.0
    rec, step, reason = validator.validate_action(current_replicas=5, proposed_step=3, metrics=metrics)
    assert rec == 7  # limited to max step of +2
    assert step == 2
    
    # 4. Emergency override on SLA breach
    validator.last_scale_time = 0.0
    metrics_bad = metrics.copy()
    metrics_bad["sla_status"] = "VIOLATED"
    rec, step, reason = validator.validate_action(current_replicas=5, proposed_step=-1, metrics=metrics_bad)
    assert rec == 6  # Override scale-down to scale-up
    assert step == 1
    assert "Override" in reason

def test_ppo_agent_update():
    agent = PPOAgent(state_dim=15, action_dim=5)
    
    # Simulate a rollout trajectory sequence
    for _ in range(10):
        dummy_state = np.random.uniform(0.0, 1.0, size=(15,)).astype(np.float32)
        # select_action saves tensors to agent.buffer
        action = agent.select_action(dummy_state)
        agent.buffer.rewards.append(-0.5)
        agent.buffer.is_terminals.append(False)
        
    # Set final step terminal
    agent.buffer.is_terminals[-1] = True
    
    assert len(agent.buffer.states) == 10
    
    # Run PPO policy optimization update
    try:
        agent.update()
        update_successful = True
    except Exception as e:
        print(f"PPO update failed: {e}")
        update_successful = False
        
    assert update_successful
    assert len(agent.buffer.states) == 0  # Buffer must be cleared after update
