# simulation/infrastructure.py

import numpy as np
from typing import Dict, Any, List, Tuple

class ClusterSimulator:
    """
    Simulates Kubernetes cluster infrastructure resource capacity, active pods, 
    provisioning boot delays, and CPU/memory resource utilization.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Scaling limits
        self.min_replicas = int(self.config.get("min_servers", 2))
        self.max_replicas = int(self.config.get("max_servers", 10))
        
        # Startup and shutdown delays (timesteps)
        self.startup_delay = int(self.config.get("startup_delay", 1))   # default 1 step (5 mins)
        self.shutdown_delay = int(self.config.get("shutdown_delay", 1)) # default 1 step (5 mins)
        
        # Capacity constants
        self.comfort_requests_per_pod = float(self.config.get("comfort_requests_per_pod", 500.0))
        self.comfort_users_per_pod = float(self.config.get("comfort_users_per_pod", 1500.0))
        self.base_cpu_idle = float(self.config.get("base_cpu_idle", 15.0))
        self.base_mem_idle = float(self.config.get("base_mem_idle", 20.0))
        
        # Runtime states
        self.current_replicas = 5
        self.active_replicas = 5
        self.pending_scale_ups = []   # list of step indices when pod becomes active
        self.pending_scale_downs = [] # list of step indices when pod is removed
        
    def reset(self, initial_replicas: int = 5):
        """Resets cluster simulator to clean state."""
        self.current_replicas = int(np.clip(initial_replicas, self.min_replicas, self.max_replicas))
        self.active_replicas = self.current_replicas
        self.pending_scale_ups.clear()
        self.pending_scale_downs.clear()
        
    def scale_to(self, target_replicas: int, current_step: int) -> Tuple[int, str]:
        """
        Schedules replica scale ups or scale downs with provisioning delays.
        """
        # Clamp target replica to limits
        target = int(np.clip(target_replicas, self.min_replicas, self.max_replicas))
        diff = target - self.current_replicas
        
        if diff > 0:
            # Scale up: schedule pods to boot
            for _ in range(diff):
                activation_step = current_step + self.startup_delay
                self.pending_scale_ups.append(activation_step)
            self.current_replicas = target
            return diff, f"Provisioning +{diff} pod(s) (activation at step {current_step + self.startup_delay})"
            
        elif diff < 0:
            # Scale down: schedule pods to terminate
            num_to_kill = abs(diff)
            # Remove from active count immediately (simulates scaling down target)
            self.active_replicas = max(self.min_replicas, self.active_replicas - num_to_kill)
            
            for _ in range(num_to_kill):
                deactivation_step = current_step + self.shutdown_delay
                self.pending_scale_downs.append(deactivation_step)
            self.current_replicas = target
            return diff, f"Terminating -{num_to_kill} pod(s) (full shutdown at step {current_step + self.shutdown_delay})"
            
        return 0, "No scaling capacity change scheduled"
        
    def update_states(self, current_step: int):
        """
        Advances the provisioning queues, activating boot-completed pods 
        and fully removing shut-down pods.
        """
        # 1. Process active scale-ups
        still_booting = []
        for activation_step in self.pending_scale_ups:
            if current_step >= activation_step:
                self.active_replicas = min(self.max_replicas, self.active_replicas + 1)
            else:
                still_booting.append(activation_step)
        self.pending_scale_ups = still_booting
        
        # 2. Process active scale-downs
        still_terminating = []
        for deactivation_step in self.pending_scale_downs:
            if current_step < deactivation_step:
                still_terminating.append(deactivation_step)
        self.pending_scale_downs = still_terminating
        
        # Keep counts consistent
        self.active_replicas = int(np.clip(self.active_replicas, self.min_replicas, self.current_replicas))

    def calculate_utilization(self, workload: Dict[str, Any]) -> Tuple[float, float]:
        """
        Computes CPU and Memory utilization percentages based on workload metrics
        and current active pods.
        """
        requests = float(workload.get("requests", 0.0))
        users = float(workload.get("users", 0.0))
        
        # 1. CPU Utilization
        if self.active_replicas <= 0:
            cpu = 100.0
        else:
            capacity_comfort = float(self.active_replicas) * self.comfort_requests_per_pod
            # CPU utilization is linear to demand with idle bias
            cpu = self.base_cpu_idle + (requests / capacity_comfort) * 60.0
            
        # 2. Memory Utilization
        if self.active_replicas <= 0:
            mem = 100.0
        else:
            users_comfort = float(self.active_replicas) * self.comfort_users_per_pod
            mem = self.base_mem_idle + (users / users_comfort) * 55.0
            
        cpu = float(np.clip(cpu, 5.0, 100.0))
        mem = float(np.clip(mem, 5.0, 100.0))
        
        return cpu, mem
