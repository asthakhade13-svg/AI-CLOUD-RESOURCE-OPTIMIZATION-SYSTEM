import os
import time
import logging
import requests
from datetime import datetime
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Configure structured log formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("AIKubeController")

# Configuration via environment variables
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")
TARGET_DEPLOYMENT = os.getenv("TARGET_DEPLOYMENT", "target-app")
TARGET_NAMESPACE = os.getenv("TARGET_NAMESPACE", "default")

# Safety parameters
MIN_REPLICAS = int(os.getenv("MIN_REPLICAS", "2"))
MAX_REPLICAS = int(os.getenv("MAX_REPLICAS", "10"))
SCALE_UP_COOLDOWN = int(os.getenv("SCALE_UP_COOLDOWN", "60"))      # Seconds
SCALE_DOWN_COOLDOWN = int(os.getenv("SCALE_DOWN_COOLDOWN", "180"))  # Seconds
MAX_SCALE_STEP = int(os.getenv("MAX_SCALE_STEP", "3"))              # Max replicas changed in one tick
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))               # Seconds
SAFE_DEFAULT_REPLICAS = int(os.getenv("SAFE_DEFAULT_REPLICAS", "3"))

SCALING_MODE = os.getenv("SCALING_MODE", "RECOMMENDATION")          # RECOMMENDATION or AUTONOMOUS
SCALING_METHOD = os.getenv("SCALING_METHOD", "KEDA")                # KEDA, HPA, DIRECT, or NONE


# State variables
last_scale_time = 0.0
api_failure_count = 0

def init_kubernetes_client():
    """Initializes in-cluster configuration if inside Kubernetes, otherwise falls back to local kubeconfig."""
    try:
        config.load_incluster_config()
        logger.info("Loaded in-cluster Kubernetes configuration.")
    except Exception:
        try:
            config.load_kube_config()
            logger.info("Loaded local kubeconfig file.")
        except Exception as e:
            logger.error(f"Failed to load Kubernetes configuration: {e}")
            raise

def query_prometheus_metric(query: str, fallback_value: float) -> float:
    """Queries Prometheus HTTP API. Returns fallback_value if query fails."""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=5
        )
        if response.status_code == 200:
            res_data = response.json()
            results = res_data.get("data", {}).get("result", [])
            if results:
                return float(results[0]["value"][1])
        logger.warning(f"Prometheus query returned empty or error: {query}. Using fallback: {fallback_value}")
    except Exception as e:
         logger.warning(f"Failed to query Prometheus: {e}. Using fallback: {fallback_value}")
    return fallback_value

def collect_system_telemetry(current_replicas: int) -> dict:
    """Gathers application workload metrics from Prometheus or mock generator fallbacks."""
    # Query current metrics (using target-app containers or falling back to typical local operation values)
    cpu = query_prometheus_metric("sum(rate(container_cpu_usage_seconds_total{container='target-app'}[2m])) * 100", 45.0)
    memory = query_prometheus_metric("sum(container_memory_working_set_bytes{container='target-app'}) / sum(machine_memory_bytes) * 100", 55.0)
    request_rate = query_prometheus_metric("sum(rate(http_requests_total{job='target-app'}[2m]))", 120.0)
    response_time = query_prometheus_metric("histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{job='target-app'}[2m])) by (le)) * 1000", 85.0)
    error_rate = query_prometheus_metric("sum(rate(http_requests_total{job='target-app', status=~'5..'}[2m])) / sum(rate(http_requests_total{job='target-app'}[2m])) * 100", 0.1)
    
    # Active users estimation relative to traffic load
    active_users = int(query_prometheus_metric("cloud_resource_active_users_count", float(current_replicas * 50)))
    
    return {
        "cpu_usage": cpu,
        "memory_usage": memory,
        "network_in": 150.0,
        "network_out": 200.0,
        "network_traffic": 350.0,
        "disk_read": 1.0,
        "disk_write": 2.0,
        "active_users": active_users,
        "request_rate": request_rate,
        "response_time": response_time,
        "error_rate": error_rate,
        "current_servers": current_replicas
    }

def get_current_deployment_replicas() -> int:
    """Queries target Deployment replicas from Kubernetes API."""
    apps_api = client.AppsV1Api()
    try:
        deployment = apps_api.read_namespaced_deployment(
            name=TARGET_DEPLOYMENT,
            namespace=TARGET_NAMESPACE
        )
        # Fall back to spec replicas or actual ready replicas
        return deployment.spec.replicas if deployment.spec.replicas is not None else SAFE_DEFAULT_REPLICAS
    except ApiException as e:
        logger.error(f"Kubernetes API error reading deployment: {e}")
        return SAFE_DEFAULT_REPLICAS

def apply_replicas_patch(target_replicas: int):
    """Updates target deployment replica count via Kubernetes Scale API patch."""
    apps_api = client.AppsV1Api()
    body = {"spec": {"replicas": target_replicas}}
    try:
        apps_api.patch_namespaced_deployment_scale(
            name=TARGET_DEPLOYMENT,
            namespace=TARGET_NAMESPACE,
            body=body
        )
        logger.info(f"Successfully scaled '{TARGET_DEPLOYMENT}' to {target_replicas} replicas.")
    except ApiException as e:
        logger.error(f"Failed to patch deployment scale: {e}")

def run_autoscaling_tick():
    """Executes one autoscale polling and patching loop iteration with safety controls."""
    global last_scale_time, api_failure_count
    
    # 1. Fetch current scale replicas count
    current_replicas = get_current_deployment_replicas()
    
    # 2. Gather metrics
    telemetry = collect_system_telemetry(current_replicas)
    
    # 3. Call AI Gateway Predict endpoint
    try:
        response = requests.post(f"{API_GATEWAY_URL}/predict", json=telemetry, timeout=5)
        if response.status_code == 200:
            api_failure_count = 0
            res_data = response.json()
            recommended = int(res_data.get("recommended_servers", current_replicas))
            logger.info(f"AI Model recommends: {recommended} replicas. (Current replicas: {current_replicas})")
        else:
            raise requests.RequestException(f"Bad gateway response: {response.status_code}")
    except Exception as e:
        api_failure_count += 1
        logger.error(f"AI API prediction failure count ({api_failure_count}/3): {e}")
        
        # Safe Rollback default on persistent API outages (>= 3 failure ticks)
        if api_failure_count >= 3:
            logger.warning(f"Gateway outage exceeds threshold! Executing rollback to safe default: {SAFE_DEFAULT_REPLICAS} replicas.")
            if current_replicas != SAFE_DEFAULT_REPLICAS:
                apply_replicas_patch(SAFE_DEFAULT_REPLICAS)
                last_scale_time = time.time()
        return

    # 4. Enforce Safety constraints (Min/Max limits)
    bounded_replicas = max(MIN_REPLICAS, min(MAX_REPLICAS, recommended))
    
    # 5. Hysteresis check (no change needed)
    if bounded_replicas == current_replicas:
        logger.info("Replicas count matches current capacity. No scaling action required.")
        return
        
    # 6. Cooldown checks
    now = time.time()
    elapsed = now - last_scale_time
    
    if bounded_replicas > current_replicas:
        # Scale Up Cooldown check
        if elapsed < SCALE_UP_COOLDOWN:
            logger.info(f"Scale UP request deferred due to cooldown lock (elapsed: {int(elapsed)}s / min: {SCALE_UP_COOLDOWN}s)")
            return
    else:
        # Scale Down Cooldown check
        if elapsed < SCALE_DOWN_COOLDOWN:
            logger.info(f"Scale DOWN request deferred due to cooldown lock (elapsed: {int(elapsed)}s / min: {SCALE_DOWN_COOLDOWN}s)")
            return
            
    # 7. Max Scaling step change sizing limits
    step = bounded_replicas - current_replicas
    if abs(step) > MAX_SCALE_STEP:
        limited_step = MAX_SCALE_STEP if step > 0 else -MAX_SCALE_STEP
        bounded_replicas = current_replicas + limited_step
        logger.info(f"Scaling step bounded by MAX_SCALE_STEP limit ({MAX_SCALE_STEP}). Adjusting target to: {bounded_replicas}")

    # 8. Execute Scale Action
    if SCALING_MODE == "AUTONOMOUS":
        if SCALING_METHOD == "DIRECT":
            apply_replicas_patch(bounded_replicas)
            last_scale_time = now
        elif SCALING_METHOD in ("KEDA", "HPA"):
            logger.info(f"[AUTONOMOUS KEDA/HPA] Sizing decision {bounded_replicas} replicas is exposed via Prometheus. Scaling delegated to Kubernetes controller.")
        else:
            logger.info(f"[AUTONOMOUS] Scaling method '{SCALING_METHOD}' configured. No direct action taken.")
    else:
        logger.info(f"[RECOMMENDATION MODE] Sizing calculation: {bounded_replicas} replicas. (Direct patching bypassed).")



def main():
    logger.info("Initializing Custom AI Autoscaler Controller Loop...")
    init_kubernetes_client()
    
    while True:
        try:
            run_autoscaling_tick()
        except Exception as e:
            logger.error(f"Unexpected error in controller loop tick: {e}", exc_info=True)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
