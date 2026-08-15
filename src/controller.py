class AutoscalingController:
    """
    Stateful autoscaling controller that enforces thresholds, consecutive-observation 
    confirmations, cooldown periods, and scaling limits to prevent capacity thrashing.
    """
    def __init__(
        self,
        current_servers: int = 5,
        min_servers: int = 1,
        max_servers: int = 20,
        scale_up_cpu_threshold: float = 80.0,
        scale_down_cpu_threshold: float = 35.0,
        cooldown_periods: int = 3,
        scale_up_confirmations: int = 3,
        scale_down_confirmations: int = 6,
        max_scale_up_step: int = 2,
        max_scale_down_step: int = 1
    ):
        # Configuration parameters
        self.min_servers = max(1, min_servers)
        self.max_servers = max(self.min_servers, max_servers)
        self.scale_up_cpu_threshold = scale_up_cpu_threshold
        self.scale_down_cpu_threshold = scale_down_cpu_threshold
        self.cooldown_periods = max(0, cooldown_periods)
        self.scale_up_confirmations = max(1, scale_up_confirmations)
        self.scale_down_confirmations = max(1, scale_down_confirmations)
        self.max_scale_up_step = max(1, max_scale_up_step)
        self.max_scale_down_step = max(1, max_scale_down_step)
        
        # Stateful attributes
        self.current_server_count = max(self.min_servers, min(self.max_servers, current_servers))
        self.ticks_since_last_scaling = self.cooldown_periods  # Seed as ready (not in cooldown)
        self.scale_up_consecutive_ticks = 0
        self.scale_down_consecutive_ticks = 0

    def make_scaling_decision(self, cpu_usage: float, predicted_servers: float, recommended_servers: int) -> dict:
        """
        Evaluates the controller state machine and returns the scaling action decision.
        """
        # 1. Enforce cooldown boundaries
        # If the number of ticks since the last scaling action is less than the cooldown period,
        # lock any subsequent scaling actions.
        if self.ticks_since_last_scaling < self.cooldown_periods:
            self.ticks_since_last_scaling += 1
            # Reset validation counters during cooldown lock
            self.scale_up_consecutive_ticks = 0
            self.scale_down_consecutive_ticks = 0
            return {
                "current_servers": self.current_server_count,
                "predicted_servers": float(predicted_servers),
                "recommended_servers": self.current_server_count,
                "action": "NO_ACTION",
                "reason": f"Autoscaler is locked in cooldown period ({self.ticks_since_last_scaling}/{self.cooldown_periods} ticks elapsed).",
                "cooldown_active": True
            }

        # 2. Check Scale-Up Condition
        # Condition: Capacity recommendation is higher than current count AND CPU usage exceeds threshold
        if recommended_servers > self.current_server_count and cpu_usage >= self.scale_up_cpu_threshold:
            self.scale_up_consecutive_ticks += 1
            self.scale_down_consecutive_ticks = 0  # Reset scale down counters
            
            if self.scale_up_consecutive_ticks >= self.scale_up_confirmations:
                # Calculate scale-up step, limiting to max_scale_up_step
                step = recommended_servers - self.current_server_count
                step = min(step, self.max_scale_up_step)
                
                new_servers = min(self.max_servers, self.current_server_count + step)
                
                if new_servers > self.current_server_count:
                    # Scaling execution state updates
                    old_servers = self.current_server_count
                    self.current_server_count = new_servers
                    self.ticks_since_last_scaling = 0  # Enter cooldown
                    self.scale_up_consecutive_ticks = 0
                    return {
                        "current_servers": old_servers,
                        "predicted_servers": float(predicted_servers),
                        "recommended_servers": self.current_server_count,
                        "action": "SCALE_UP",
                        "reason": f"SCALE_UP triggered: High CPU ({cpu_usage:.1f}%) and capacity deficit confirmed for {self.scale_up_confirmations} consecutive observations. Scaled from {old_servers} to {new_servers} (max step limit applied).",
                        "cooldown_active": False
                    }
            else:
                return {
                    "current_servers": self.current_server_count,
                    "predicted_servers": float(predicted_servers),
                    "recommended_servers": self.current_server_count,
                    "action": "NO_ACTION",
                    "reason": f"High workload detected. Scale-up confirmation in progress ({self.scale_up_consecutive_ticks}/{self.scale_up_confirmations} observations).",
                    "cooldown_active": False
                }

        # 3. Check Scale-Down Condition
        # Condition: Capacity recommendation is lower than current count AND CPU utilization is low
        elif recommended_servers < self.current_server_count and cpu_usage <= self.scale_down_cpu_threshold:
            self.scale_down_consecutive_ticks += 1
            self.scale_up_consecutive_ticks = 0  # Reset scale up counters
            
            if self.scale_down_consecutive_ticks >= self.scale_down_confirmations:
                # Calculate scale-down step, limiting to max_scale_down_step
                step = self.current_server_count - recommended_servers
                step = min(step, self.max_scale_down_step)
                
                new_servers = max(self.min_servers, self.current_server_count - step)
                
                if new_servers < self.current_server_count:
                    old_servers = self.current_server_count
                    self.current_server_count = new_servers
                    self.ticks_since_last_scaling = 0  # Enter cooldown
                    self.scale_down_consecutive_ticks = 0
                    return {
                        "current_servers": old_servers,
                        "predicted_servers": float(predicted_servers),
                        "recommended_servers": self.current_server_count,
                        "action": "SCALE_DOWN",
                        "reason": f"SCALE_DOWN triggered: Low CPU ({cpu_usage:.1f}%) and capacity surplus confirmed for {self.scale_down_confirmations} consecutive observations. Scaled from {old_servers} to {new_servers} (max step limit applied).",
                        "cooldown_active": False
                    }
            else:
                return {
                    "current_servers": self.current_server_count,
                    "predicted_servers": float(predicted_servers),
                    "recommended_servers": self.current_server_count,
                    "action": "NO_ACTION",
                    "reason": f"Low workload detected. Scale-down confirmation in progress ({self.scale_down_consecutive_ticks}/{self.scale_down_confirmations} observations).",
                    "cooldown_active": False
                }

        # 4. Workload Stable (No scaling conditions met)
        else:
            self.scale_up_consecutive_ticks = 0
            self.scale_down_consecutive_ticks = 0
            self.ticks_since_last_scaling += 1  # Increment cooldown timer
            return {
                "current_servers": self.current_server_count,
                "predicted_servers": float(predicted_servers),
                "recommended_servers": self.current_server_count,
                "action": "NO_ACTION",
                "reason": "Workload is stable. Server capacity is optimal.",
                "cooldown_active": False
            }
