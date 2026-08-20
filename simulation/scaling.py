# simulation/scaling.py

import numpy as np
from typing import Dict, Any, Optional

def get_static_recommendation(current_replicas: int, fixed_replicas: int = 5) -> int:
    """Always targets a fixed number of replica instances."""
    return fixed_replicas - current_replicas

def get_threshold_recommendation(cpu_usage: float, current_replicas: int) -> int:
    """Reactive scaling based on threshold rules: CPU > 80% or CPU < 35%."""
    if cpu_usage > 80.0:
        return 1
    elif cpu_usage < 35.0:
        return -1
    return 0

def get_hpa_recommendation(
    cpu_usage: float, 
    current_replicas: int, 
    target_cpu_util: float = 60.0,
    min_pods: int = 2,
    max_pods: int = 10
) -> int:
    """Kubernetes HPA replica ratio equation."""
    if current_replicas <= 0:
        return min_pods
    desired = int(np.ceil(current_replicas * (cpu_usage / target_cpu_util)))
    desired = int(np.clip(desired, min_pods, max_pods))
    return desired - current_replicas

def get_ml_predictive_recommendation(
    current_replicas: int, 
    predicted_required_servers: int
) -> int:
    """ML predictive sizing: targets predicted count directly."""
    return predicted_required_servers - current_replicas

def get_rl_recommendation(
    ppo_agent: Any, 
    obs: np.ndarray, 
    model_loaded: bool = True
) -> int:
    """Queries PPO Actor networks for decisions."""
    if ppo_agent is None or not model_loaded:
        return 0 # default NO_ACTION
        
    action_idx = ppo_agent.select_action(obs)
    # Clear memory (since we evaluate without training updates)
    ppo_agent.buffer.clear()
    
    from rl.actions import idx_to_step
    return idx_to_step(action_idx)
