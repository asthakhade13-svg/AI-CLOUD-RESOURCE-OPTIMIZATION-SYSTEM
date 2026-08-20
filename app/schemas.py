from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# =====================================================================
# REQUEST SCHEMAS
# =====================================================================

class TelemetryPayload(BaseModel):
    # Telemetry metrics
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage (0-100)", json_schema_extra={"example": 78.0})
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage (0-100)", json_schema_extra={"example": 82.0})
    network_in: float = Field(150.0, ge=0, description="Network In throughput (Mbps)")
    network_out: float = Field(240.0, ge=0, description="Network Out throughput (Mbps)")
    network_traffic: Optional[float] = Field(None, ge=0, description="Total network traffic (Mbps)")
    disk_read: float = Field(50.0, ge=0, description="Disk Read IOPS")
    disk_write: float = Field(25.0, ge=0, description="Disk Write IOPS")
    active_users: int = Field(..., ge=0, description="Number of active user sessions", json_schema_extra={"example": 290})
    request_rate: float = Field(400.0, ge=0, description="Requests per second")
    response_time: float = Field(150.0, ge=0, description="Average response latency (ms)")
    error_rate: float = Field(0.0, ge=0, le=100, description="Error rate percentage (0-100)")
    current_servers: int = Field(..., ge=1, description="Current active servers count", json_schema_extra={"example": 5})
    server_cost: float = Field(0.60, ge=0, description="Current server hosting cost per hour ($/hour)")

class PredictRequest(BaseModel):
    # User's specific required input schema for /predict
    cpu_usage: float = Field(..., ge=0, le=100, description="CPU usage percentage (0-100)", json_schema_extra={"example": 78})
    memory_usage: float = Field(..., ge=0, le=100, description="Memory usage percentage (0-100)", json_schema_extra={"example": 82})
    network_traffic: float = Field(..., ge=0, description="Network traffic (Mbps)", json_schema_extra={"example": 390})
    active_users: int = Field(..., ge=0, description="Number of active users", json_schema_extra={"example": 290})
    current_servers: int = Field(..., ge=1, description="Current server capacity count", json_schema_extra={"example": 5})
    
    # Optional parameters for advanced configurations (backwards compatibility)
    network_in: float = Field(150.0, ge=0)
    network_out: float = Field(240.0, ge=0)
    disk_read: float = Field(80.0, ge=0)
    disk_write: float = Field(40.0, ge=0)
    request_rate: float = Field(625.0, ge=0)
    response_time: float = Field(185.0, ge=0)
    error_rate: float = Field(0.05, ge=0, le=100)
    server_cost: float = Field(0.60, ge=0)
    
    min_servers: int = Field(1, ge=1)
    max_servers: int = Field(20, ge=1)
    safety_margin: float = Field(0.10, ge=0.0, le=1.0)
    
    scale_up_cpu_threshold: float = Field(80.0, ge=0.0, le=100.0)
    scale_down_cpu_threshold: float = Field(35.0, ge=0.0, le=100.0)
    cooldown_periods: int = Field(3, ge=0)
    scale_up_confirmations: int = Field(3, ge=1)
    scale_down_confirmations: int = Field(6, ge=1)
    max_scale_up_step: int = Field(2, ge=1)
    max_scale_down_step: int = Field(1, ge=1)
    
    sla_penalty_weight: float = Field(5.0, ge=0.0)
    overprovisioning_weight: float = Field(0.5, ge=0.0)
    
    target_response_time: float = Field(200.0, ge=0.0)
    maximum_error_rate: float = Field(1.0, ge=0.0)
    minimum_availability: float = Field(99.0, ge=0.0)

class AutoscaleRequest(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100)
    current_servers: int = Field(..., ge=1)
    predicted_servers: float = Field(..., ge=0)
    recommended_servers: int = Field(..., ge=1)
    sla_status: str = Field("HEALTHY")
    anomaly_severity: str = Field("LOW")
    
    # Configurations
    min_servers: int = Field(1, ge=1)
    max_servers: int = Field(20, ge=1)
    scale_up_cpu_threshold: float = Field(80.0, ge=0.0, le=100.0)
    scale_down_cpu_threshold: float = Field(35.0, ge=0.0, le=100.0)
    cooldown_periods: int = Field(3, ge=0)
    scale_up_confirmations: int = Field(3, ge=1)
    scale_down_confirmations: int = Field(6, ge=1)
    max_scale_up_step: int = Field(2, ge=1)
    max_scale_down_step: int = Field(1, ge=1)

class OptimizeRequest(BaseModel):
    predicted_required_servers: float = Field(..., ge=0)
    current_servers: int = Field(..., ge=1)
    server_cost_per_hour: float = Field(..., ge=0)
    min_servers: int = Field(1, ge=1)
    max_servers: int = Field(20, ge=1)
    sla_penalty_weight: float = Field(5.0, ge=0.0)
    overprovisioning_weight: float = Field(0.5, ge=0.0)

# =====================================================================
# RESPONSE SCHEMAS
# =====================================================================

class PredictResponse(BaseModel):
    # Core requested output parameters
    predicted_servers: int = Field(..., description="Raw server count predicted by model.")
    recommended_servers: int = Field(..., description="Final capacity recommendation after checks.")
    action: str = Field(..., description="Recommended scaling action: SCALE_UP / SCALE_DOWN / NO_ACTION")
    current_servers: Optional[int] = None
    
    # Advanced metrics (optional fields to maintain complete API output details)
    scaling_action: Optional[str] = None
    reason: Optional[str] = None
    reasoning: Optional[str] = None
    cooldown_active: Optional[bool] = None
    sla_status: Optional[str] = None
    risk_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    anomaly_score: Optional[float] = None
    severity: Optional[str] = None
    affected_metrics: Optional[List[str]] = None
    recommendation: Optional[str] = None
    shap_explanation: Optional[str] = None
    shap_contributions: Optional[Dict[str, float]] = None
    hourly_cost: Optional[float] = None
    estimated_daily_cost: Optional[float] = None
    estimated_monthly_cost: Optional[float] = None
    estimated_savings: Optional[float] = None
    forecasts: Optional[dict] = None

class ForecastOutput(BaseModel):
    forecasts: Dict[str, Dict[str, float]] = Field(..., description="Telemetry forecasts for 5min, 10min, 15min.")

class AutoscaleOutput(BaseModel):
    current_servers: int
    predicted_servers: float
    recommended_servers: int
    action: str
    reason: str
    cooldown_active: bool

class AnomalyOutput(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    severity: str
    affected_metrics: List[str]
    recommendation: str
    reason: str

class OptimizeOutput(BaseModel):
    recommended_servers: int
    hourly_cost: float
    estimated_daily_cost: float
    estimated_monthly_cost: float
    estimated_savings: float
    sla_status: str
    optimization_reason: str

# =====================================================================
# REINFORCEMENT LEARNING SCHEMAS
# =====================================================================

class RLPredictRequest(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100)
    memory_usage: float = Field(..., ge=0, le=100)
    network_traffic: float = Field(..., ge=0)
    active_users: int = Field(..., ge=0)
    request_rate: float = Field(..., ge=0)
    response_time: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=100)
    current_servers: int = Field(..., ge=1)
    
    # RL additional observation features
    predicted_workload: float = Field(..., ge=0, le=100)
    predicted_required_servers: int = Field(..., ge=1)
    hourly_cost: float = Field(..., ge=0)
    sla_status: str = Field("HEALTHY")
    is_anomaly: bool = Field(False)
    prev_step: int = Field(0, ge=-2, le=2)
    hour: float = Field(12.0, ge=0, le=23)

class RLPredictResponse(BaseModel):
    current_replicas: int
    recommended_action: str
    recommended_replicas: int
    expected_reward: float
    risk_score: float
    reason: str

class RLEvaluateRequest(BaseModel):
    episodes: int = Field(5, ge=1, le=100)
    seed: int = Field(42, ge=0)

class RLEvaluateResponse(BaseModel):
    benchmark_results: List[Dict[str, Any]]

class RLStatusResponse(BaseModel):
    model_loaded: bool
    checkpoint_exists: bool
    state_dimension: int
    action_dimension: int
    active_mode: str

# =====================================================================
# DIGITAL TWIN SIMULATION SCHEMAS
# =====================================================================

class SimulationScenarioRequest(BaseModel):
    policy_name: str = Field("HPA", description="Scaling policy: STATIC / THRESHOLD / HPA / ML_PREDICTIVE / RL_PPO")
    initial_replicas: int = Field(5, ge=1, le=20)
    max_steps: int = Field(288, ge=10, le=1000)
    traffic_multiplier: float = Field(1.0, ge=0.1, le=10.0)
    users_multiplier: float = Field(1.0, ge=0.1, le=10.0)
    workload_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    failures: List[Dict[str, Any]] = Field(default_factory=list)

class SimulationScenarioResponse(BaseModel):
    summary: Dict[str, Any]
    history: List[Dict[str, Any]]

class SimulationCompareRequest(BaseModel):
    initial_replicas: int = Field(5, ge=1, le=20)
    max_steps: int = Field(288, ge=10, le=1000)
    traffic_multiplier: float = Field(1.0, ge=0.1, le=10.0)
    users_multiplier: float = Field(1.0, ge=0.1, le=10.0)
    workload_patterns: List[Dict[str, Any]] = Field(default_factory=list)
    failures: List[Dict[str, Any]] = Field(default_factory=list)

class SimulationCompareResponse(BaseModel):
    benchmark_results: List[Dict[str, Any]]


