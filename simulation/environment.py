# simulation/environment.py

import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from simulation.workload import WorkloadGenerator
from simulation.infrastructure import ClusterSimulator
from simulation.failures import ChaosInjector
from simulation.metrics import MetricsTracker
from rl.state import get_observation, STATE_DIM
from rl.safety import SafetyValidator

class DigitalTwinEnv:
    """
    The orchestrator environment for the Cloud Digital Twin.
    Simulates operational timelines, schedules workloads, models resources,
    triggers chaos injects, and runs autoscale controllers.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Instantiate sub-components
        self.workload_generator = WorkloadGenerator()
        self.cluster_simulator = ClusterSimulator(self.config)
        self.chaos_injector = ChaosInjector()
        
        target_latency = float(self.config.get("target_response_time", 200.0))
        unit_cost = float(self.config.get("server_unit_cost", 0.60))
        self.metrics_tracker = MetricsTracker(target_latency, unit_cost)
        self.safety_validator = SafetyValidator(self.config)
        
        # Episode parameters
        self.max_steps = int(self.config.get("max_steps", 288))
        self.current_step = 0
        self.prev_action_step = 0
        
        # Workload and failure schedules
        self.workload_patterns: List[Dict[str, Any]] = []
        self.failure_schedule: List[Dict[str, Any]] = [] # list of {"step": int, "type": str, "value": Any}
        
    def set_workload_patterns(self, patterns: List[Dict[str, Any]]):
        """Sets the active combination workload patterns."""
        self.workload_patterns = patterns
        
    def schedule_failure(self, step: int, failure_type: str, value: Any = True):
        """Schedules a chaos failure injection at a specific simulation timestep."""
        self.failure_schedule.append({
            "step": step,
            "type": failure_type,
            "value": value
        })
        
    def reset(self, initial_replicas: int = 5) -> np.ndarray:
        """Resets the simulation to the initial state."""
        self.current_step = 0
        self.prev_action_step = 0
        self.cluster_simulator.reset(initial_replicas)
        self.chaos_injector.reset()
        self.metrics_tracker.reset()
        
        # Initialize initial observation
        metrics = self._get_current_metrics(0, {"users": 100, "requests": 400.0, "network": 240.0})
        return get_observation(metrics)
        
    def _get_current_metrics(self, proposed_step: int, workload: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the metrics calculations for CPU/Mem/Latency and cost."""
        # 1. Base utilization
        cpu, mem = self.cluster_simulator.calculate_utilization(workload)
        
        # 2. Base latency/errors
        latency, err_rate, sla = self.metrics_tracker.compute_performance(
            cpu, mem, workload["requests"], self.cluster_simulator.active_replicas
        )
        
        # 3. Apply active failure modifiers
        cpu, mem, latency, err_rate, active_pods = self.chaos_injector.apply_failures(
            cpu, mem, latency, err_rate, self.cluster_simulator.active_replicas,
            self.cluster_simulator.min_replicas
        )
        
        # Re-evaluate SLA status after chaos modifiers
        if latency > self.metrics_tracker.target_latency or err_rate > 1.0:
            sla = "VIOLATED"
        elif latency > self.metrics_tracker.target_latency * 0.8:
            sla = "AT_RISK"
        else:
            sla = "HEALTHY"
            
        # 4. Hourly cost
        hourly = float(self.cluster_simulator.current_replicas) * self.metrics_tracker.unit_cost
        
        # 5. Compile observation telemetry dict
        # Optimal predicted servers representation
        pred_servers = int(np.clip(np.ceil(workload["requests"] / 400.0), self.cluster_simulator.min_replicas, self.cluster_simulator.max_replicas))
        
        # Forecast workload (next step requests)
        next_w = self.workload_generator.generate_step(self.current_step + 1, self.workload_patterns)
        next_cpu = self.cluster_simulator.base_cpu_idle + (next_w["requests"] / (self.cluster_simulator.active_replicas * self.cluster_simulator.comfort_requests_per_pod) * 60.0)
        next_cpu = float(np.clip(next_cpu, 5.0, 100.0))
        
        metrics_dict = {
            "cpu_usage": cpu,
            "memory_usage": mem,
            "network_traffic": workload["network"],
            "active_users": workload["users"],
            "request_rate": workload["requests"],
            "response_time": latency,
            "error_rate": err_rate,
            "current_servers": self.cluster_simulator.current_replicas,
            "max_servers": self.cluster_simulator.max_replicas,
            "predicted_workload": next_cpu,
            "predicted_required_servers": pred_servers,
            "hourly_cost": hourly,
            "sla_status": sla,
            "target_response_time": self.metrics_tracker.target_latency,
            "is_anomaly": True if (cpu > 90.0 or err_rate > 2.0) else False,
            "prev_step": proposed_step,
            "hour": float((self.current_step % 288) // 12)
        }
        return metrics_dict
        
    def step(
        self, 
        policy_name: str, 
        ppo_agent: Optional[Any] = None, 
        model_loaded: bool = True,
        ml_predicted_required: Optional[int] = None
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes a simulation timestep.
        1. Checks failure schedules.
        2. Pulls workload inputs.
        3. Updates container states.
        4. Calculates performance metrics.
        5. Queries autoscale policy recommendations.
        6. Filters through safety validator layers.
        7. Advances step index.
        """
        # 1. Inject scheduled chaos failures
        for fail in self.failure_schedule:
            if fail["step"] == self.current_step:
                f_type = fail["type"]
                f_val = fail["value"]
                if f_type == "pod_crash":
                    self.chaos_injector.inject_pod_crash(int(f_val))
                elif f_type == "cpu_leak":
                    self.chaos_injector.inject_cpu_leak(bool(f_val))
                elif f_type == "mem_leak":
                    self.chaos_injector.inject_mem_leak(bool(f_val))
                elif f_type == "network_degradation":
                    self.chaos_injector.inject_network_degradation(float(f_val))
                elif f_type == "slowdown":
                    self.chaos_injector.inject_slowdown(float(f_val))
                    
        # 2. Advance provisioning state machines
        self.cluster_simulator.update_states(self.current_step)
        
        # 3. Pull workload load
        workload = self.workload_generator.generate_step(self.current_step, self.workload_patterns)
        
        # 4. Compute performance metrics
        metrics = self._get_current_metrics(self.prev_action_step, workload)
        
        # 5. Query active autoscaling policy
        p_name = policy_name.upper()
        proposed_step = 0
        
        if p_name == "STATIC":
            from simulation.scaling import get_static_recommendation
            proposed_step = get_static_recommendation(self.cluster_simulator.current_replicas)
        elif p_name == "THRESHOLD":
            from simulation.scaling import get_threshold_recommendation
            proposed_step = get_threshold_recommendation(metrics["cpu_usage"], self.cluster_simulator.current_replicas)
        elif p_name == "HPA":
            from simulation.scaling import get_hpa_recommendation
            proposed_step = get_hpa_recommendation(
                metrics["cpu_usage"], 
                self.cluster_simulator.current_replicas,
                min_pods=self.cluster_simulator.min_replicas,
                max_pods=self.cluster_simulator.max_replicas
            )
        elif p_name == "ML_PREDICTIVE":
            from simulation.scaling import get_ml_predictive_recommendation
            pred_req = ml_predicted_required if ml_predicted_required is not None else metrics["predicted_required_servers"]
            proposed_step = get_ml_predictive_recommendation(self.cluster_simulator.current_replicas, pred_req)
        elif p_name == "RL_PPO":
            from simulation.scaling import get_rl_recommendation
            obs = get_observation(metrics)
            proposed_step = get_rl_recommendation(ppo_agent, obs, model_loaded)
            
        # 6. Apply Safety validator wraps
        _, safe_step, safety_reason = self.safety_validator.validate_action(
            current_replicas=self.cluster_simulator.current_replicas,
            proposed_step=proposed_step,
            metrics=metrics
        )
        
        # Execute scale action in simulator
        diff, scale_msg = self.cluster_simulator.scale_to(
            self.cluster_simulator.current_replicas + safe_step, 
            self.current_step
        )
        
        # Save step metrics logs
        step_log = {
            "step": self.current_step,
            "cpu": metrics["cpu_usage"],
            "memory": metrics["memory_usage"],
            "network": metrics["network_traffic"],
            "users": metrics["active_users"],
            "requests": metrics["request_rate"],
            "latency": metrics["response_time"],
            "errors": metrics["error_rate"],
            "replicas": self.cluster_simulator.current_replicas,
            "active_replicas": self.cluster_simulator.active_replicas,
            "cost": self.metrics_tracker.calculate_cost(self.cluster_simulator.current_replicas),
            "sla_status": metrics["sla_status"],
            "action_taken": safe_step,
            "scale_reason": safety_reason if safe_step != proposed_step else scale_msg
        }
        self.metrics_tracker.record_step(step_log)
        
        self.prev_action_step = safe_step
        self.current_step += 1
        
        done = self.current_step >= self.max_steps
        next_obs = get_observation(self._get_current_metrics(self.prev_action_step, workload))
        
        return next_obs, step_log["cost"], done, step_log
