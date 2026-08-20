# rl/safety.py

import time
from typing import Dict, Any, Tuple

class SafetyValidator:
    """
    Enforces critical safety envelopes and constraints to prevent uncontrolled autoscaling,
    cooldown violations, capacity bounds breaches, or SLA-threatening scale-down decisions.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_replicas = self.config.get("min_servers", 2)
        self.max_replicas = self.config.get("max_servers", 10)
        self.max_step = self.config.get("max_scale_up_step", 2)
        self.cooldown_period = self.config.get("cooldown_seconds", 60) # Cooldown window in seconds
        self.last_scale_time = 0.0

    def validate_action(
        self, 
        current_replicas: int, 
        proposed_step: int, 
        metrics: Dict[str, Any]
    ) -> Tuple[int, int, str]:
        """
        Validates the proposed scale step change against safety policies.
        
        Returns:
            recommended_replicas (int): Clamped target replicas count.
            validated_step (int): Safe action step to execute.
            reason (str): Decisive safety logic citation.
        """
        now = time.time()
        
        # 1. Check Cooldown active (if time elapsed < cooldown seconds)
        time_since_scale = now - self.last_scale_time
        cooldown_active = time_since_scale < self.cooldown_period
        
        if proposed_step != 0 and cooldown_active:
            # Overrides active scaling step to 0 (NO_ACTION) during cooldown lock
            return current_replicas, 0, f"Blocked: Cooldown active ({int(time_since_scale)}s elapsed of {self.cooldown_period}s)"
            
        # 2. Emergency Override for system protection (SLA breaches or extreme CPU loads)
        cpu_usage = float(metrics.get("cpu_usage", 0.0))
        latency = float(metrics.get("response_time", 100.0))
        target_latency = float(metrics.get("target_response_time", 200.0))
        sla_violated = metrics.get("sla_status", "HEALTHY") == "VIOLATED" or latency > target_latency
        
        # If overloaded but agent proposes scaling down or NO_ACTION, override to safety scale-up
        if (cpu_usage > 90.0 or sla_violated) and proposed_step <= 0:
            emergency_step = 1 if current_replicas + 1 <= self.max_replicas else 0
            if emergency_step > 0:
                self.last_scale_time = now
                return current_replicas + emergency_step, emergency_step, "Override: Emergency SLA/CPU overload scaling trigger"
                
        # 3. Limit single scale steps (clamping action values within boundaries)
        clamped_step = max(-self.max_step, min(self.max_step, proposed_step))
        
        # 4. Enforce Replica boundaries
        recommended_replicas = current_replicas + clamped_step
        if recommended_replicas < self.min_replicas:
            clamped_step = self.min_replicas - current_replicas
            recommended_replicas = self.min_replicas
            reason = "Boundary: Clamped to min replicas count"
        elif recommended_replicas > self.max_replicas:
            clamped_step = self.max_replicas - current_replicas
            recommended_replicas = self.max_replicas
            reason = "Boundary: Clamped to max replicas count"
        else:
            reason = "Approved: Safety validator passed"
            
        if clamped_step != 0:
            self.last_scale_time = now
            
        return recommended_replicas, clamped_step, reason
