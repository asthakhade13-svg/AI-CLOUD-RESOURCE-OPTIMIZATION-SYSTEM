# simulation/metrics.py

import numpy as np
from typing import Dict, Any, List, Tuple

class MetricsTracker:
    """
    Computes system response latency, error rates, SLA status, 
    and gathers statistics for algorithm benchmarking.
    """
    def __init__(self, target_latency: float = 200.0, unit_cost: float = 0.60):
        self.target_latency = target_latency
        self.unit_cost = unit_cost
        self.history: List[Dict[str, Any]] = []
        
    def reset(self):
        """Clears metric accumulation history."""
        self.history.clear()
        
    def compute_performance(
        self, 
        cpu: float, 
        memory: float, 
        requests: float, 
        active_pods: int
    ) -> Tuple[float, float, str]:
        """
        Calculates response time (latency), error rate, and SLA status
        based on resource utilization.
        """
        base_latency = 75.0 # ms
        
        # 1. Latency: queuing curve modeled with exponential tail above 75% CPU load
        if cpu <= 75.0:
            latency = base_latency + (cpu / 75.0) * 20.0
        else:
            exponent = (cpu - 75.0) / 8.0
            latency = base_latency + 20.0 + 15.0 * np.exp(exponent)
            
        latency = float(np.clip(latency, 30.0, 2000.0))
        
        # 2. Error rate: scales up when CPU or Memory saturates
        error_rate = 0.0
        if cpu > 88.0:
            error_rate += (cpu - 88.0) * 0.6
        if memory > 92.0:
            error_rate += (memory - 92.0) * 0.9
            
        error_rate = float(np.clip(error_rate, 0.0, 100.0))
        
        # 3. SLA Evaluation
        if latency > self.target_latency or error_rate > 1.0:
            sla_status = "VIOLATED"
        elif latency > self.target_latency * 0.8 or error_rate > 0.5:
            sla_status = "AT_RISK"
        else:
            sla_status = "HEALTHY"
            
        return latency, error_rate, sla_status
        
    def calculate_cost(self, replicas: int) -> float:
        """Returns cost per timestep (5-minute interval = 5/60 of hourly cost)."""
        hourly = float(replicas) * self.unit_cost
        return hourly * (5.0 / 60.0)

    def record_step(self, step_metrics: Dict[str, Any]):
        """Records a single simulation timestep telemetry record."""
        self.history.append(step_metrics.copy())

    def get_summary(self) -> Dict[str, Any]:
        """Computes summary metrics for comparative benchmarks."""
        if not self.history:
            return {}
            
        costs = [h["cost"] for h in self.history]
        latencies = [h["latency"] for h in self.history]
        replicas = [h["replicas"] for h in self.history]
        cpu_utils = [h["cpu"] for h in self.history]
        sla_statuses = [h["sla_status"] for h in self.history]
        
        total_cost = sum(costs)
        avg_cost = np.mean(costs) * (60.0 / 5.0) # Scale back to hourly average
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        sla_violations = sla_statuses.count("VIOLATED")
        
        # Scaling events count (differences in consecutive steps)
        scaling_events = 0
        for i in range(1, len(replicas)):
            if replicas[i] != replicas[i-1]:
                scaling_events += 1
                
        # Recovery time: max consecutive steps spent in VIOLATED status
        max_consecutive_sla = 0
        curr_consecutive = 0
        for status in sla_statuses:
            if status == "VIOLATED":
                curr_consecutive += 1
                max_consecutive_sla = max(max_consecutive_sla, curr_consecutive)
            else:
                curr_consecutive = 0
                
        # Over/Under provisioning steps compared to optimal count
        # Optimal count is request_rate / 400 confort capacity
        over_steps = 0
        under_steps = 0
        for h in self.history:
            optimal = int(np.clip(np.ceil(h["requests"] / 400.0), 2, 10))
            if h["replicas"] > optimal:
                over_steps += 1
            elif h["replicas"] < optimal:
                under_steps += 1
                
        return {
            "total_cost": float(round(total_cost, 2)),
            "avg_cost_per_hour": float(round(avg_cost, 2)),
            "avg_latency": float(round(avg_latency, 1)),
            "p95_latency": float(round(p95_latency, 1)),
            "sla_violations": int(sla_violations),
            "scaling_events": int(scaling_events),
            "over_provisioning_steps": int(over_steps),
            "under_provisioning_steps": int(under_steps),
            "avg_cpu_utilization": float(round(np.mean(cpu_utils), 1)),
            "recovery_steps": int(max_consecutive_sla)
        }
