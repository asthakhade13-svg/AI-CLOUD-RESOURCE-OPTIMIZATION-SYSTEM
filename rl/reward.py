# rl/reward.py

from typing import Dict, Any

DEFAULT_REWARD_CONFIG: Dict[str, float] = {
    "w_cost": 1.0,
    "w_latency": 1.5,
    "w_sla": 5.0,
    "w_overprovisioning": 0.5,
    "w_underprovisioning": 2.0,
    "w_thrashing": 0.8
}

def calculate_reward(
    metrics: Dict[str, Any], 
    action_step: int, 
    config: Dict[str, float] = None
) -> float:
    """
    Calculates a scalar reward for a step, penalizing high infrastructure cost,
    response latency, SLA violations, over/underprovisioning, and scaling thrashing.
    
    Returns a negative cost/penalty score. High reward means optimal balancing.
    """
    if config is None:
        config = DEFAULT_REWARD_CONFIG
        
    w_cost = config.get("w_cost", 1.0)
    w_latency = config.get("w_latency", 1.5)
    w_sla = config.get("w_sla", 5.0)
    w_over = config.get("w_overprovisioning", 0.5)
    w_under = config.get("w_underprovisioning", 2.0)
    w_thrash = config.get("w_thrashing", 0.8)
    
    # 1. Cost Penalty (proportional to current servers relative to max 20)
    current_servers = metrics.get("current_servers", 1)
    max_servers = metrics.get("max_servers", 20)
    cost_penalty = float(current_servers) / float(max_servers)
    
    # 2. Latency Penalty (ratio of response time to SLA target 200ms)
    response_time = float(metrics.get("response_time", 100.0))
    target_latency = float(metrics.get("target_response_time", 200.0))
    latency_penalty = min(2.0, response_time / target_latency)
    
    # 3. SLA Violation Penalty
    sla_status = str(metrics.get("sla_status", "HEALTHY")).upper()
    sla_violated = 1.0 if (sla_status == "VIOLATED" or response_time > target_latency or float(metrics.get("error_rate", 0.0)) > 1.0) else 0.0
    sla_penalty = sla_violated
    
    # 4. Over/Under-provisioning Penalties
    pred_required = float(metrics.get("predicted_required_servers", current_servers))
    diff = float(current_servers) - pred_required
    
    overprovisioning_penalty = max(0.0, diff) / float(max_servers)
    underprovisioning_penalty = max(0.0, -diff) / float(max_servers)
    
    # 5. Thrashing Penalty (applied if the agent executed a non-zero scale step)
    thrashing_penalty = 1.0 if action_step != 0 else 0.0
    
    # Compute weighted sum of penalties
    total_penalty = (
        w_cost * cost_penalty +
        w_latency * latency_penalty +
        w_sla * sla_penalty +
        w_over * overprovisioning_penalty +
        w_under * underprovisioning_penalty +
        w_thrash * thrashing_penalty
    )
    
    # Return negative penalty as reward
    return -total_penalty
