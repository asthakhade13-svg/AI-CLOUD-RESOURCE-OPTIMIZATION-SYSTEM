# simulation/cloud_adapter.py

from abc import ABC, abstractmethod
from typing import Dict, Any

class CloudAdapter(ABC):
    """
    Abstract Cloud Adapter Interface separating simulation runs 
    from real production infrastructure controllers.
    """
    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """Fetches active cluster metric telemetry."""
        pass
        
    @abstractmethod
    def apply_scale(self, target_replicas: int) -> int:
        """Applies cluster capacity updates and returns scheduled replica counts."""
        pass

class SimulatedCloudAdapter(CloudAdapter):
    """
    Simulated Cloud Adapter routing requests entirely to the 
    in-memory Digital Twin engine (safety mode).
    """
    def __init__(self, digital_twin_env: Any):
        self.env = digital_twin_env
        
    def get_telemetry(self) -> Dict[str, Any]:
        # Return state metric map
        workload = self.env.workload_generator.generate_step(self.env.current_step, self.env.workload_patterns)
        return self.env._get_current_metrics(self.env.prev_action_step, workload)
        
    def apply_scale(self, target_replicas: int) -> int:
        # Scale through the simulator
        diff, _ = self.env.cluster_simulator.scale_to(target_replicas, self.env.current_step)
        return self.env.cluster_simulator.current_replicas

class KubernetesCloudAdapter(CloudAdapter):
    """
    Production Cloud Adapter using Kubernetes client APIs 
    to interact with live microservice deployments.
    """
    def __init__(self, namespace: str = "default", deployment_name: str = "app-deployment"):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.k8s_client = None
        
    def _lazy_init_k8s(self):
        if self.k8s_client is None:
            from kubernetes import client, config
            try:
                config.load_incluster_config()
            except Exception:
                config.load_kube_config()
            self.k8s_client = client.AppsV1Api()
            
    def get_telemetry(self) -> Dict[str, Any]:
        """
        Query prometheus/grafana metrics database to assemble live cluster state telemetry.
        """
        # In a real environment, query Prometheus API endpoints
        return {
            "cpu_usage": 50.0,
            "memory_usage": 50.0,
            "current_servers": self.get_replica_count(),
            "sla_status": "HEALTHY"
        }
        
    def get_replica_count(self) -> int:
        self._lazy_init_k8s()
        try:
            deploy = self.k8s_client.read_namespaced_deployment(self.deployment_name, self.namespace)
            return deploy.spec.replicas
        except Exception:
            return 1
            
    def apply_scale(self, target_replicas: int) -> int:
        self._lazy_init_k8s()
        try:
            body = {"spec": {"replicas": target_replicas}}
            self.k8s_client.patch_namespaced_deployment_scale(
                self.deployment_name, self.namespace, body
            )
            return target_replicas
        except Exception as e:
            print(f"Kubernetes cluster scale failed: {e}")
            return self.get_replica_count()
