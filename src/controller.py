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

    def make_scaling_decision(
        self, 
        cpu_usage: float, 
        predicted_servers: float, 
        recommended_servers: int, 
        sla_status: str = "HEALTHY",
        anomaly_severity: str = "LOW"
    ) -> dict:
        """
        Evaluates the controller state machine and returns the scaling action decision.
        """
        # 1. Enforce cooldown boundaries
        # Lock scaling actions during cooldown unless a CRITICAL anomaly is detected.
        # Outage prevention overrides cooldown limits.
        if self.ticks_since_last_scaling < self.cooldown_periods:
            if anomaly_severity == "CRITICAL":
                # Override cooldown
                pass
            else:
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
        # Trigger scale-up if there is a recommended increase AND (CPU exceeds threshold OR SLA is violated/at-risk OR high anomaly severity)
        if recommended_servers > self.current_server_count and (
            cpu_usage >= self.scale_up_cpu_threshold or 
            sla_status in ["VIOLATED", "AT_RISK"] or 
            anomaly_severity in ["HIGH", "CRITICAL"]
        ):
            self.scale_up_consecutive_ticks += 1
            self.scale_down_consecutive_ticks = 0  # Reset scale down counters
            
            # Anomaly severity HIGH or CRITICAL bypasses the consecutive ticks confirmation check
            is_anomaly_override = anomaly_severity in ["HIGH", "CRITICAL"]
            
            if self.scale_up_consecutive_ticks >= self.scale_up_confirmations or is_anomaly_override:
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
                    
                    if is_anomaly_override:
                        trigger_msg = f"AI Anomaly Alert ({anomaly_severity} Severity)"
                    else:
                        trigger_msg = f"High CPU ({cpu_usage:.1f}%)" if cpu_usage >= self.scale_up_cpu_threshold else f"SLA Warning ({sla_status})"
                        
                    return {
                        "current_servers": old_servers,
                        "predicted_servers": float(predicted_servers),
                        "recommended_servers": self.current_server_count,
                        "action": "SCALE_UP",
                        "reason": f"SCALE_UP triggered: {trigger_msg} and capacity deficit confirmed. Scaled from {old_servers} to {new_servers}.",
                        "cooldown_active": False
                    }
            else:
                return {
                    "current_servers": self.current_server_count,
                    "predicted_servers": float(predicted_servers),
                    "recommended_servers": self.current_server_count,
                    "action": "NO_ACTION",
                    "reason": f"High workload or SLA risk detected. Scale-up confirmation in progress ({self.scale_up_consecutive_ticks}/{self.scale_up_confirmations} observations).",
                    "cooldown_active": False
                }

        # 3. Check Scale-Down Condition
        # Trigger scale-down if capacity surplus exists AND CPU is low AND SLA status is healthy AND no active anomalies
        elif recommended_servers < self.current_server_count and cpu_usage <= self.scale_down_cpu_threshold and sla_status == "HEALTHY" and anomaly_severity not in ["HIGH", "CRITICAL"]:
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
            reason_msg = "Workload is stable. Server capacity is optimal."
            if recommended_servers < self.current_server_count and sla_status != "HEALTHY":
                reason_msg = f"Scale down prevented: SLA is {sla_status}."
            return {
                "current_servers": self.current_server_count,
                "predicted_servers": float(predicted_servers),
                "recommended_servers": self.current_server_count,
                "action": "NO_ACTION",
                "reason": reason_msg,
                "cooldown_active": False
            }
