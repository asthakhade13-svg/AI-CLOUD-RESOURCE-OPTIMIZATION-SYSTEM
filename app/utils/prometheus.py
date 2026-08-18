from prometheus_client import Gauge, Counter, Histogram

# Name prefixes
PREFIX = "cloud_resource"

# Telemetry Gauges
CPU_USAGE = Gauge(f"{PREFIX}_cpu_usage_ratio", "Current CPU utilization percentage (0-100)")
MEMORY_USAGE = Gauge(f"{PREFIX}_memory_usage_ratio", "Current memory utilization percentage (0-100)")
NETWORK_TRAFFIC = Gauge(f"{PREFIX}_network_traffic_mbps", "Current network traffic throughput in Mbps")
REQUEST_RATE = Gauge(f"{PREFIX}_request_rate_per_sec", "Current request rate in requests/sec")
RESPONSE_TIME = Gauge(f"{PREFIX}_response_time_ms", "Current average request latency in milliseconds")
ERROR_RATE = Gauge(f"{PREFIX}_error_rate_ratio", "Current error rate percentage (0-100)")
ACTIVE_USERS = Gauge(f"{PREFIX}_active_users_count", "Current count of active user sessions")

# Sizing Gauges
CURRENT_SERVERS = Gauge(f"{PREFIX}_current_servers_count", "Current server capacity count")
PREDICTED_SERVERS = Gauge(f"{PREFIX}_predicted_servers_count", "Raw server capacity count predicted by the ML model")
RECOMMENDED_SERVERS = Gauge(f"{PREFIX}_recommended_servers_count", "Final recommended server capacity count after optimization")

# Scaling Events Counter (labeled by action type)
SCALING_ACTIONS = Counter(
    f"{PREFIX}_scaling_actions_total", 
    "Total count of scaling actions executed by the controller",
    ["action"]
)

# Latency Histogram for ML pipeline executions
PREDICTION_LATENCY = Histogram(
    f"{PREFIX}_prediction_latency_seconds",
    "Time taken in seconds to run the end-to-end forecasting, capacity, optimization, and scaling decision pipeline"
)

def update_prometheus_metrics(telemetry_data: dict, prediction_data: dict, latency: float):
    """
    Utility helper to dynamically update all Gauges, Counters, and Histograms 
    with the values processed during the /predict lifecycle tick.
    """
    # 1. Update Telemetry
    CPU_USAGE.set(telemetry_data.get("cpu_usage", 0.0))
    MEMORY_USAGE.set(telemetry_data.get("memory_usage", 0.0))
    NETWORK_TRAFFIC.set(telemetry_data.get("network_traffic", 0.0))
    REQUEST_RATE.set(telemetry_data.get("request_rate", 0.0))
    RESPONSE_TIME.set(telemetry_data.get("response_time", 0.0))
    ERROR_RATE.set(telemetry_data.get("error_rate", 0.0))
    ACTIVE_USERS.set(telemetry_data.get("active_users", 0))
    
    # 2. Update Sizing
    CURRENT_SERVERS.set(telemetry_data.get("current_servers", 1))
    PREDICTED_SERVERS.set(prediction_data.get("predicted_servers", 1.0))
    RECOMMENDED_SERVERS.set(prediction_data.get("recommended_servers", 1))
    
    # 3. Increment scaling actions (if SCALE_UP or SCALE_DOWN)
    action = prediction_data.get("action", "NO_ACTION")
    SCALING_ACTIONS.labels(action=action).inc()
    
    # 4. Record ML processing latency
    PREDICTION_LATENCY.observe(latency)
