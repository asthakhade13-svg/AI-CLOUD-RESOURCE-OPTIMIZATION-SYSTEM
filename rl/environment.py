# rl/environment.py

import numpy as np
import random
from typing import Dict, Any, Tuple
from rl.actions import idx_to_step
from rl.state import get_observation, STATE_DIM
from rl.reward import calculate_reward

class CloudAutoscalingEnv:
    """
    Simulated cloud environment for training and evaluating reinforcement learning 
    autoscaling policies. Simulates request rates, CPU/memory performance loops,
    provisioning delay lag, SLA violations, and hourly cost calculations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Environmental configuration limits
        self.min_servers = self.config.get("min_servers", 2)
        self.max_servers = self.config.get("max_servers", 10)
        self.max_steps = self.config.get("max_steps", 288)  # 24h of 5m intervals
        self.target_response_time = self.config.get("target_response_time", 200.0) # SLA limit
        self.server_unit_cost = self.config.get("server_unit_cost", 0.60) # $/hr per server
        self.provisioning_delay = self.config.get("provisioning_delay", 1)  # steps for scale action to take effect
        self.cooldown_period = self.config.get("cooldown_period", 3)  # steps cooldown lock
        
        # Dynamic state trackers
        self.current_step = 0
        self.current_servers = 5
        self.target_servers = 5
        self.prev_step_action = 0
        self.steps_since_scale = 999
        self.pending_actions = []  # list of tuples: (activation_step, target_count)
        
        # Metrics history for rendering/evaluation logs
        self.history = []
        
    def _generate_workload(self, step_idx: int) -> Tuple[int, float, float]:
        """Generates active users, request rates, and network traffic for a step."""
        # 1. Base diurnal pattern (sinusoidal day/night cycle)
        # 288 steps is one full day. Mid-day peaks at step 144.
        time_factor = np.sin(2 * np.pi * step_idx / 288.0 - np.pi / 2.0)  # ranges [-1, 1]
        base_users = int(1200 + 1000 * time_factor + 300 * np.sin(4 * np.pi * step_idx / 288.0))
        
        # 2. Add random spikes & noise
        noise = random.randint(-150, 150)
        users = max(50, base_users + noise)
        
        # Simulated spike at peak hours (mid-day)
        if 120 <= step_idx <= 160 and random.random() < 0.15:
            users += random.randint(1500, 3000)
            
        # 3. Derive request rate and network throughput
        request_rate = float(users) * 4.2  # ~4.2 requests per user session
        network_traffic = request_rate * 0.6  # 0.6 Mbps per request/sec
        
        return users, request_rate, network_traffic

    def reset(self) -> np.ndarray:
        """Resets the environment to initial conditions."""
        self.current_step = 0
        self.current_servers = 5
        self.target_servers = 5
        self.prev_step_action = 0
        self.steps_since_scale = 999
        self.pending_actions = []
        self.history = []
        
        # Get initial state metrics
        metrics = self._get_state_metrics()
        obs = get_observation(metrics)
        return obs

    def _get_state_metrics(self) -> Dict[str, Any]:
        """Calculates performance loops based on current capacity and workload."""
        users, req_rate, traffic = self._generate_workload(self.current_step)
        
        # 1. CPU usage simulation (reaches 100% if servers are overloaded)
        # Comfort capacity: 500 requests per server unit
        capacity_comfort = float(self.current_servers) * 500.0
        cpu = (req_rate / capacity_comfort) * 65.0 if capacity_comfort > 0 else 100.0
        # Add baseline CPU of idle servers + noise
        cpu = np.clip(cpu + 15.0 + random.uniform(-3, 3), 5.0, 100.0)
        
        # 2. Memory usage simulation
        mem = 35.0 + 45.0 * (float(users) / (float(self.current_servers) * 1500.0 + 100))
        mem = np.clip(mem + random.uniform(-2, 2), 10.0, 100.0)
        
        # 3. Response time latency (grows exponentially as CPU usage saturates)
        base_latency = 75.0  # 75ms base latency
        if cpu < 70.0:
            latency = base_latency + (cpu / 70.0) * 20.0
        else:
            # Saturated queue modeling
            exponent = (cpu - 70.0) / 10.0
            latency = base_latency + 20.0 + 25.0 * np.exp(exponent)
        # Clamp response time to max 2000ms
        latency = np.clip(latency + random.uniform(-5, 5), 30.0, 2000.0)
        
        # 4. Error rate simulation
        err = 0.0
        if cpu > 85.0:
            err = (cpu - 85.0) * 0.8  # errors start scaling up above 85% CPU load
        err = np.clip(err + random.uniform(-0.1, 0.1), 0.0, 100.0)
        
        # 5. Cost calculation ($/hour for current server allocation)
        hourly_cost = float(self.current_servers) * self.server_unit_cost
        
        # 6. SLA status check
        if latency > self.target_response_time:
            sla_status = "VIOLATED"
        elif latency > self.target_response_time * 0.8:
            sla_status = "AT_RISK"
        else:
            sla_status = "HEALTHY"
            
        # 7. Anomaly status (simulated when load spikes extremely fast)
        is_anomaly = False
        if cpu > 92.0 and random.random() < 0.4:
            is_anomaly = True
            
        # 8. Forecast prediction (simulated next step workload CPU)
        _, next_req_rate, _ = self._generate_workload(self.current_step + 1)
        next_cpu = (next_req_rate / capacity_comfort) * 65.0 if capacity_comfort > 0 else 100.0
        next_cpu = np.clip(next_cpu + 15.0, 5.0, 100.0)
        
        # 9. Predicted required servers (Rule-based helper representation)
        pred_servers = int(np.clip(np.ceil(req_rate / 400.0), self.min_servers, self.max_servers))
        
        # Compile state dictionary
        metrics_dict = {
            "cpu_usage": cpu,
            "memory_usage": mem,
            "network_traffic": traffic,
            "active_users": users,
            "request_rate": req_rate,
            "response_time": latency,
            "error_rate": err,
            "current_servers": self.current_servers,
            "max_servers": self.max_servers,
            "predicted_workload": next_cpu,
            "predicted_required_servers": pred_servers,
            "hourly_cost": hourly_cost,
            "sla_status": sla_status,
            "target_response_time": self.target_response_time,
            "is_anomaly": is_anomaly,
            "prev_step": self.prev_step_action,
            "hour": float((self.current_step % 288) // 12)  # Hour from step index (0-23)
        }
        
        return metrics_dict

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes a step in the environment.
        1. Checks scaling cooldown locks.
        2. Places pending provisioning updates.
        3. Advances time state.
        4. Calculates step reward.
        """
        # Map action index to step replica count change
        step_change = idx_to_step(action_idx)
        
        # Enforce step capacity limits
        # Apply cooldown locks (no capacity changes accepted if locked)
        cooldown_active = self.steps_since_scale < self.cooldown_period
        
        if step_change != 0 and not cooldown_active:
            # Calculate target
            self.target_servers = int(np.clip(
                self.current_servers + step_change, 
                self.min_servers, 
                self.max_servers
            ))
            # Schedule provisioning delay activation
            activation_step = self.current_step + self.provisioning_delay
            self.pending_actions.append((activation_step, self.target_servers))
            
            # Reset counters
            self.steps_since_scale = 0
            self.prev_step_action = step_change
        else:
            # Action blocked by cooldown or NO_ACTION selected
            self.steps_since_scale += 1
            if cooldown_active and step_change != 0:
                # Agent attempted scaling but was blocked
                self.prev_step_action = 0
            else:
                self.prev_step_action = step_change
                
        # Resolve pending scale activations
        resolved_actions = []
        for act_step, servers in self.pending_actions:
            if self.current_step >= act_step:
                self.current_servers = servers
            else:
                resolved_actions.append((act_step, servers))
        self.pending_actions = resolved_actions
        
        # Get metrics before advancing step
        metrics = self._get_state_metrics()
        
        # Calculate step reward
        reward = calculate_reward(metrics, self.prev_step_action)
        
        # Advance simulation timeline step
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Record metrics history
        metrics["step"] = self.current_step
        metrics["action_taken"] = self.prev_step_action
        metrics["cooldown_active"] = cooldown_active
        metrics["reward"] = reward
        self.history.append(metrics)
        
        # Fetch next observation vector
        next_obs = get_observation(self._get_state_metrics())
        
        return next_obs, reward, done, metrics
