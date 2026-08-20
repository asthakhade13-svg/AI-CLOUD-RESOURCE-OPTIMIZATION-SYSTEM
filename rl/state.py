# rl/state.py

import numpy as np
from typing import Dict, Any

STATE_DIM = 15

def get_observation(metrics: Dict[str, Any]) -> np.ndarray:
    """
    Parses and normalizes telemetry metrics into a 15-dimensional observation vector [0.0, 1.0].
    
    The vector components:
    0. CPU usage (0-100) -> /100.0
    1. Memory usage (0-100) -> /100.0
    2. Network traffic (0-2000) -> /2000.0 (capped)
    3. Active users (0-10000) -> /10000.0 (capped)
    4. Request rate (0-5000) -> /5000.0 (capped)
    5. Response time (0-2000 ms) -> /2000.0 (capped)
    6. Error rate (0-100) -> /100.0
    7. Current replicas (1-20) -> /20.0
    8. Predicted workload (future CPU forecast 0-100) -> /100.0
    9. Predicted required servers (1-20) -> /20.0
    10. Current hourly cost (0-12.0 $/hr) -> /12.0 (capped)
    11. SLA status ('HEALTHY': 0.0, 'AT_RISK': 0.5, 'VIOLATED': 1.0)
    12. Anomaly status (bool or 0/1) -> float
    13. Previous scaling step (-2 to 2) -> (prev_step + 2) / 4.0
    14. Time of day (0-23 hours) -> /24.0
    """
    # 0. CPU usage
    cpu = float(metrics.get("cpu_usage", 0.0)) / 100.0
    
    # 1. Memory usage
    memory = float(metrics.get("memory_usage", 0.0)) / 100.0
    
    # 2. Network traffic
    traffic = min(2000.0, float(metrics.get("network_traffic", 0.0))) / 2000.0
    
    # 3. Active users
    users = min(10000.0, float(metrics.get("active_users", 0.0))) / 10000.0
    
    # 4. Request rate
    req_rate = min(5000.0, float(metrics.get("request_rate", 0.0))) / 5000.0
    
    # 5. Response time
    resp_time = min(2000.0, float(metrics.get("response_time", 0.0))) / 2000.0
    
    # 6. Error rate
    err_rate = float(metrics.get("error_rate", 0.0)) / 100.0
    
    # 7. Current servers
    current_servers = float(metrics.get("current_servers", 1.0)) / 20.0
    
    # 8. Predicted workload (CPU 5min ahead)
    pred_workload = float(metrics.get("predicted_workload", 0.0)) / 100.0
    
    # 9. Predicted required servers
    pred_servers = float(metrics.get("predicted_required_servers", 1.0)) / 20.0
    
    # 10. Hourly cost
    cost = min(12.0, float(metrics.get("hourly_cost", 0.0))) / 12.0
    
    # 11. SLA Status
    sla_str = str(metrics.get("sla_status", "HEALTHY")).upper()
    if sla_str == "VIOLATED":
        sla_val = 1.0
    elif sla_str == "AT_RISK":
        sla_val = 0.5
    else:
        sla_val = 0.0
        
    # 12. Anomaly status
    is_anomaly = float(bool(metrics.get("is_anomaly", False)))
    
    # 13. Previous step
    prev_step = float(metrics.get("prev_step", 0))
    prev_step_norm = (np.clip(prev_step, -2, 2) + 2) / 4.0
    
    # 14. Time of day
    hour = float(metrics.get("hour", 12.0)) / 24.0
    
    obs = np.array([
        cpu, memory, traffic, users, req_rate, resp_time, err_rate,
        current_servers, pred_workload, pred_servers, cost,
        sla_val, is_anomaly, prev_step_norm, hour
    ], dtype=np.float32)
    
    return obs
