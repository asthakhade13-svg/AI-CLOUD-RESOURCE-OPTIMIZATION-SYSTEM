# simulation/scenarios.py

from typing import Dict, Any, List, Optional
from simulation.environment import DigitalTwinEnv

class WhatIfAnalyzer:
    """
    Runs configured stress-testing scenarios and chaos failure simulations
    against the Cloud Digital Twin to calculate performance impact.
    """
    def __init__(self, simulator_config: Dict[str, Any] = None):
        self.simulator_config = simulator_config or {}

    def run_custom_scenario(
        self, 
        config: Dict[str, Any], 
        ppo_agent: Optional[Any] = None,
        model_loaded: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a simulation scenario with custom traffic multipliers 
        and scheduled chaos failures.
        
        config keys:
            policy_name (str): STATIC / THRESHOLD / HPA / ML_PREDICTIVE / RL_PPO
            initial_replicas (int): default 5
            max_steps (int): default 288
            traffic_multiplier (float): default 1.0 (scales requests & traffic)
            users_multiplier (float): default 1.0 (scales active sessions count)
            workload_patterns (list): workload generator patterns list config
            failures (list): scheduled list of chaos triggers: [{"step": 50, "type": "pod_crash", "value": 2}]
        """
        policy_name = config.get("policy_name", "HPA")
        initial_replicas = int(config.get("initial_replicas", 5))
        max_steps = int(config.get("max_steps", self.simulator_config.get("max_steps", 288)))
        
        # Adjust base simulation parameters
        sim_params = self.simulator_config.copy()
        sim_params["max_steps"] = max_steps

        
        # Instantiate environment
        env = DigitalTwinEnv(sim_params)
        
        # 1. Adjust workload patterns with scaling multipliers
        traffic_mult = float(config.get("traffic_multiplier", 1.0))
        users_mult = float(config.get("users_multiplier", 1.0))
        raw_patterns = config.get("workload_patterns", [])
        
        scaled_patterns = []
        for pat in raw_patterns:
            new_pat = pat.copy()
            new_params = pat.get("params", {}).copy()
            
            # Apply multipliers
            if "users" in new_params:
                new_params["users"] = float(new_params["users"]) * users_mult
            if "amplitude" in new_params:
                new_params["amplitude"] = float(new_params["amplitude"]) * users_mult
            if "bias" in new_params:
                new_params["bias"] = float(new_params["bias"]) * users_mult
            if "amp1" in new_params:
                new_params["amp1"] = float(new_params["amp1"]) * users_mult
            if "amp2" in new_params:
                new_params["amp2"] = float(new_params["amp2"]) * users_mult
                
            # If traffic multiplier is set, scale requests_per_user
            if traffic_mult != 1.0:
                base_req_ratio = new_params.get("requests_per_user", 4.2)
                new_params["requests_per_user"] = float(base_req_ratio) * traffic_mult
                
            new_pat["params"] = new_params
            scaled_patterns.append(new_pat)
            
        # Fallback default pattern if empty (sinusoidal)
        if not scaled_patterns:
            scaled_patterns = [{
                "type": "sinusoidal",
                "params": {
                    "amplitude": 600.0 * users_mult,
                    "bias": 1200.0 * users_mult,
                    "period": 288.0,
                    "requests_per_user": 4.2 * traffic_mult
                }
            }]
            
        env.set_workload_patterns(scaled_patterns)
        
        # 2. Schedule chaos failures
        failures = config.get("failures", [])
        for f in failures:
            step = int(f.get("step", 0))
            f_type = str(f.get("type", "pod_crash"))
            f_val = f.get("value", True)
            env.schedule_failure(step, f_type, f_val)
            
        # 3. Reset and execute timeline loop
        env.reset(initial_replicas)
        done = False
        
        while not done:
            _, _, done, _ = env.step(
                policy_name=policy_name,
                ppo_agent=ppo_agent,
                model_loaded=model_loaded
            )
            
        # Get final simulation summary report
        summary = env.metrics_tracker.get_summary()
        
        return {
            "summary": summary,
            "history": env.metrics_tracker.history
        }
