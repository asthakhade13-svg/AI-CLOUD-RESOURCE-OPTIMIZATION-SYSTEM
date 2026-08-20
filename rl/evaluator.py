# rl/evaluator.py

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from rl.environment import CloudAutoscalingEnv
from rl.agent import PPOAgent
from rl.safety import SafetyValidator

class Evaluator:
    """
    Benchmarks and compares the PPO Reinforcement Learning autoscaling engine
    against traditional heuristic, HPA, and predictive ML models.
    """
    def __init__(self, checkpoint_path: str = "rl/models/ppo_autoscaler.pth"):
        self.checkpoint_path = checkpoint_path
        
        # Load PPO agent
        self.agent = PPOAgent(state_dim=15, action_dim=5)
        if os.path.exists(checkpoint_path):
            self.agent.load(checkpoint_path)
            self.model_loaded = True
        else:
            self.model_loaded = False
            
        # Instantiate environment and safety validator
        self.env = CloudAutoscalingEnv()
        self.safety = SafetyValidator()
        
    def evaluate_static(self, seed: int = 100, fixed_servers: int = 5) -> List[Dict[str, Any]]:
        """Simulates static resource allocations."""
        np.random.seed(seed)
        state = self.env.reset()
        self.env.current_servers = fixed_servers
        self.env.target_servers = fixed_servers
        done = False
        
        while not done:
            # Action selection is forced to NO_ACTION (index 2)
            _, _, done, _ = self.env.step(2)
            # Freeze capacity to static target
            self.env.current_servers = fixed_servers
            self.env.target_servers = fixed_servers
            
        return self.env.history

    def evaluate_threshold(self, seed: int = 100) -> List[Dict[str, Any]]:
        """Simulates simple reactive threshold scaling (CPU > 80% or CPU < 35%)."""
        np.random.seed(seed)
        state = self.env.reset()
        done = False
        
        while not done:
            cpu = self.env._get_state_metrics()["cpu_usage"]
            if cpu > 80.0:
                action_idx = 3 # SCALE_UP_1
            elif cpu < 35.0:
                action_idx = 1 # SCALE_DOWN_1
            else:
                action_idx = 2 # NO_ACTION
                
            _, _, done, _ = self.env.step(action_idx)
            
        return self.env.history

    def evaluate_hpa(self, seed: int = 100, target_cpu: float = 60.0) -> List[Dict[str, Any]]:
        """Simulates Kubernetes HPA algorithm: Desired = ceil(Current * (Current_CPU / Target_CPU))."""
        np.random.seed(seed)
        state = self.env.reset()
        done = False
        
        while not done:
            metrics = self.env._get_state_metrics()
            cpu = metrics["cpu_usage"]
            current_servers = self.env.current_servers
            
            desired = int(np.ceil(current_servers * (cpu / target_cpu)))
            desired = np.clip(desired, self.env.min_servers, self.env.max_servers)
            
            diff = desired - current_servers
            if diff >= 2:
                action_idx = 4 # SCALE_UP_2
            elif diff == 1:
                action_idx = 3 # SCALE_UP_1
            elif diff == -1:
                action_idx = 1 # SCALE_DOWN_1
            elif diff <= -2:
                action_idx = 0 # SCALE_DOWN_2
            else:
                action_idx = 2 # NO_ACTION
                
            _, _, done, _ = self.env.step(action_idx)
            
        return self.env.history

    def evaluate_ml_predictive(self, seed: int = 100) -> List[Dict[str, Any]]:
        """Simulates ML predictive model autoscaling targeting predicted capacity directly."""
        np.random.seed(seed)
        state = self.env.reset()
        done = False
        
        while not done:
            metrics = self.env._get_state_metrics()
            target_servers = metrics["predicted_required_servers"]
            diff = target_servers - self.env.current_servers
            
            if diff >= 2:
                action_idx = 4
            elif diff == 1:
                action_idx = 3
            elif diff == -1:
                action_idx = 1
            elif diff <= -2:
                action_idx = 0
            else:
                action_idx = 2
                
            _, _, done, _ = self.env.step(action_idx)
            
        return self.env.history

    def evaluate_rl(self, seed: int = 100, use_safety: bool = True) -> List[Dict[str, Any]]:
        """Simulates trained PPO policy model actions validated through safety constraints."""
        np.random.seed(seed)
        state = self.env.reset()
        done = False
        
        while not done:
            if self.model_loaded:
                action_idx = self.agent.select_action(state)
            else:
                # Untrained / fallback random policies
                action_idx = 2
                
            # Parse proposed step
            from rl.actions import idx_to_step
            proposed_step = idx_to_step(action_idx)
            
            if use_safety:
                metrics = self.env._get_state_metrics()
                # Run through the safety validator
                _, safe_step, _ = self.safety.validate_action(self.env.current_servers, proposed_step, metrics)
                
                # Remap back to action index
                if safe_step == 2: action_idx = 4
                elif safe_step == 1: action_idx = 3
                elif safe_step == -1: action_idx = 1
                elif safe_step == -2: action_idx = 0
                else: action_idx = 2
                
            next_state, _, done, _ = self.env.step(action_idx)
            state = next_state
            
        return self.env.history

    def run_benchmark(self, seed: int = 100) -> pd.DataFrame:
        """Runs evaluation simulation on all algorithms and returns comparative metrics."""
        results = {
            "Static Allocation (5)": self.evaluate_static(seed, fixed_servers=5),
            "Threshold Scaling": self.evaluate_threshold(seed),
            "Kubernetes HPA": self.evaluate_hpa(seed),
            "ML Predictive Sizer": self.evaluate_ml_predictive(seed),
            "RL Autoscaler (PPO)": self.evaluate_rl(seed, use_safety=True)
        }
        
        comparison = []
        
        for name, history in results.items():
            df = pd.DataFrame(history)
            
            total_cost = df["hourly_cost"].sum() * (5.0 / 60.0) # Sum cost of 5-minute ticks in hours
            avg_latency = df["response_time"].mean()
            sla_violations = (df["sla_status"] == "VIOLATED").sum()
            scaling_events = (df["action_taken"] != 0).sum()
            avg_util = df["cpu_usage"].mean()
            
            # Under/Over provisioning count
            pred_req = df["predicted_required_servers"]
            curr_srv = df["current_servers"]
            under_steps = (curr_srv < pred_req).sum()
            over_steps = (curr_srv > pred_req).sum()
            
            # Recovery time: max consecutive steps in SLA violations
            max_consecutive_sla = 0
            curr_consecutive = 0
            for status in df["sla_status"]:
                if status == "VIOLATED":
                    curr_consecutive += 1
                    max_consecutive_sla = max(max_consecutive_sla, curr_consecutive)
                else:
                    curr_consecutive = 0
                    
            comparison.append({
                "Algorithm": name,
                "Total Cost ($)": round(total_cost, 2),
                "Avg Latency (ms)": round(avg_latency, 1),
                "SLA Violations": int(sla_violations),
                "Scaling Events": int(scaling_events),
                "Under-provisioning Steps": int(under_steps),
                "Over-provisioning Steps": int(over_steps),
                "Recovery Steps": int(max_consecutive_sla),
                "Avg CPU Util (%)": round(avg_util, 1)
            })
            
        return pd.DataFrame(comparison)

if __name__ == "__main__":
    evaluator = Evaluator()
    df_compare = evaluator.run_benchmark()
    print("\n--- AUTOSCALING BENCHMARK RESULTS ---")
    print(df_compare.to_markdown(index=False))
