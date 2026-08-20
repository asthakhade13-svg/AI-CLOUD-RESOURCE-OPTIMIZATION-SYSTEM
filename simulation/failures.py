# simulation/failures.py

from typing import Dict, Any, Tuple

class ChaosInjector:
    """
    Simulates operational hazards and failure modes (chaos engineering) 
    such as pod crashes, resource saturation, network degradation, and backend dependencies slowdown.
    """
    def __init__(self):
        # Maps failure type string to its active parameter value
        self.active_failures: Dict[str, Any] = {}
        
    def reset(self):
        """Clears all active failure modes."""
        self.active_failures.clear()
        
    def inject_pod_crash(self, count: int = 1):
        """Schedules pod crashes that kill active replicas."""
        self.active_failures["pod_crash"] = self.active_failures.get("pod_crash", 0) + count
        
    def inject_cpu_leak(self, active: bool = True):
        """Simulates CPU saturation lockup."""
        self.active_failures["cpu_leak"] = active
        
    def inject_mem_leak(self, active: bool = True):
        """Simulates memory leak leading to OOM saturation."""
        self.active_failures["mem_leak"] = active
        
    def inject_network_degradation(self, loss_pct: float = 30.0):
        """Degrades network, creating packet drop rate."""
        self.active_failures["network_degradation"] = loss_pct
        
    def inject_slowdown(self, multiplier: float = 2.5):
        """Simulates slow database query execution or API dependencies."""
        self.active_failures["slowdown"] = multiplier
        
    def apply_failures(
        self, 
        cpu: float, 
        memory: float, 
        latency: float, 
        error_rate: float, 
        active_pods: int,
        min_pods: int = 2
    ) -> Tuple[float, float, float, float, int]:
        """
        Modifies system telemetry outputs based on the scheduled chaos failures.
        """
        # 1. Pod Crash: immediately reduces the active pod count
        crashed_count = self.active_failures.get("pod_crash", 0)
        if crashed_count > 0:
            active_pods = max(min_pods, active_pods - crashed_count)
            # Reset crash counter (triggered once per step call)
            self.active_failures["pod_crash"] = 0
            
        # 2. CPU Saturation Override
        if self.active_failures.get("cpu_leak", False):
            cpu = 100.0
            latency = max(latency, 1200.0) # lockup latency
            error_rate = max(error_rate, 5.0)
            
        # 3. Memory Saturation Override
        if self.active_failures.get("mem_leak", False):
            memory = 100.0
            error_rate = max(error_rate, 8.0) # OOM kills
            
        # 4. Network degradation (packet drop increases request failures and latency)
        net_pct = self.active_failures.get("network_degradation", 0.0)
        if net_pct > 0.0:
            error_rate = max(error_rate, net_pct * 0.8)
            latency = latency * (1.0 + (net_pct / 100.0) * 1.5)
            
        # 5. Service dependency slowdown
        slow_multiplier = self.active_failures.get("slowdown", 1.0)
        if slow_multiplier > 1.0:
            latency = latency * slow_multiplier
            
        return cpu, memory, latency, error_rate, active_pods
