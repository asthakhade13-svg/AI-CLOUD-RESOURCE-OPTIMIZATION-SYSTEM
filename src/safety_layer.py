# src/safety_layer.py

import os
import json
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime

AUDIT_LOG_PATH = "data/decision_audit_log.json"
POLICIES_PATH = "data/safety_policies.json"

DEFAULT_POLICIES = {
    "min_replicas": 2,
    "max_replicas": 15,
    "scaling_step_limit": 4,
    "cooldown_seconds": 60,
    "budget_limit_hourly": 2.5,
    "sla_latency_threshold_ms": 250.0,
    "emergency_restrictions": False,
    "protected_services": ["Authentication", "Database"]
}

# Global in-memory states
EMERGENCY_STOP = False
OPERATING_MODE = "SIMULATION"  # SIMULATION, APPROVAL, AUTONOMOUS
LAST_SCALING_TIMESTAMP = 0.0

def load_policies() -> dict:
    """Loads safety policy thresholds from file or returns defaults."""
    os.makedirs("data", exist_ok=True)
    if os.path.exists(POLICIES_PATH):
        try:
            with open(POLICIES_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return DEFAULT_POLICIES.copy()
    else:
        # Write defaults
        with open(POLICIES_PATH, "w") as f:
            json.dump(DEFAULT_POLICIES, f, indent=2)
        return DEFAULT_POLICIES.copy()

def save_policies(policies: dict):
    """Saves policy configurations back to disk."""
    with open(POLICIES_PATH, "w") as f:
        json.dump(policies, f, indent=2)

class RiskEngine:
    """
    Computes a classification risk score (LOW, MEDIUM, HIGH, CRITICAL)
    based on model uncertainties, telemetry alerts, and infrastructure states.
    """
    @staticmethod
    def calculate_risk(
        prediction_uncertainty: float,
        sla_risk_pct: float,
        anomaly_severity: str,
        workload_volatility: float,
        infrastructure_health: str,
        model_confidence: float,
        recent_scaling_freq: int
    ) -> Tuple[str, float]:
        # Scoring scale from 0.0 to 1.0
        score = 0.0
        
        # 1. Uncertainty and low model confidence penalty
        score += prediction_uncertainty * 0.15
        score += (1.0 - model_confidence) * 0.15
        
        # 2. SLA degradation risks
        score += (sla_risk_pct / 100.0) * 0.25
        
        # 3. Anomaly warnings
        if anomaly_severity == "CRITICAL":
            score += 0.25
        elif anomaly_severity == "MAJOR":
            score += 0.15
        elif anomaly_severity == "MINOR":
            score += 0.05
            
        # 4. Infrastructure failures
        if infrastructure_health == "unhealthy":
            score += 0.20
        elif infrastructure_health == "degraded":
            score += 0.10
            
        # 5. Thrashing frequency
        score += min(recent_scaling_freq * 0.05, 0.15)
        
        # Classification threshold mapping
        if score >= 0.75:
            return "CRITICAL", round(score, 2)
        elif score >= 0.50:
            return "HIGH", round(score, 2)
        elif score >= 0.25:
            return "MEDIUM", round(score, 2)
        else:
            return "LOW", round(score, 2)

class PolicyEngine:
    """Enforces safety guardrails before action execution."""
    @staticmethod
    def validate_action(
        current_replicas: int,
        target_replicas: int,
        policies: dict,
        cooldown_elapsed: bool,
        is_emergency: bool = False
    ) -> Tuple[bool, str]:
        if is_emergency:
            return False, "BLOCKED: Global Emergency Stop is active."
            
        if policies.get("emergency_restrictions", False):
            return False, "BLOCKED: Safety engine emergency restrictions are currently enabled."
            
        # 1. Minimum and Maximum limits
        if target_replicas < policies["min_replicas"]:
            return False, f"BLOCKED: Target configuration {target_replicas} violates min_replicas limit ({policies['min_replicas']})."
            
        if target_replicas > policies["max_replicas"]:
            return False, f"BLOCKED: Target configuration {target_replicas} violates max_replicas limit ({policies['max_replicas']})."
            
        # 2. Step limits
        step = abs(target_replicas - current_replicas)
        if step > policies["scaling_step_limit"]:
            return False, f"BLOCKED: Action step size ({step} replicas) exceeds scaling_step_limit guardrail ({policies['scaling_step_limit']})."
            
        # 3. Cooldown limits
        if step > 0 and not cooldown_elapsed:
            return False, "BLOCKED: Action throttled due to scaling cooldown window constraint."
            
        # 4. Budget constraints (Simulated price index cap)
        hourly_est = target_replicas * 0.12  # $0.12 per core hour estimate
        if hourly_est > policies["budget_limit_hourly"]:
            return False, f"BLOCKED: Hourly cluster cost estimate (${hourly_est:.2f}/hr) exceeds budget cap (${policies['budget_limit_hourly']:.2f}/hr)."
            
        return True, "APPROVED: All safety policy checks passed successfully."

class SafetyControlLayer:
    """Unified Orchestration interface."""
    def __init__(self):
        self.policies = load_policies()
        self.load_audit_logs()

    def load_audit_logs(self):
        self.audit_logs = []
        if os.path.exists(AUDIT_LOG_PATH):
            try:
                with open(AUDIT_LOG_PATH, "r") as f:
                    self.audit_logs = json.load(f)
            except Exception:
                self.audit_logs = []

    def save_audit_log(self, entry: dict):
        self.audit_logs.insert(0, entry)
        self.audit_logs = self.audit_logs[:50]  # Cap history size
        with open(AUDIT_LOG_PATH, "w") as f:
            json.dump(self.audit_logs, f, indent=2)

    def process_decision(
        self,
        current_replicas: int,
        predicted_traffic: float,
        current_cpu: float,
        current_latency: float,
        model_confidence: float = 0.92,
        anomaly_severity: str = "NONE",
        operator: str = "SYSTEM_AI"
    ) -> dict:
        global EMERGENCY_STOP, OPERATING_MODE, LAST_SCALING_TIMESTAMP
        
        # Rule-based calculation of required capacity (for ML/RL failure fallback validation)
        # Fallback check
        is_model_failure = model_confidence < 0.30 or predicted_traffic < 0
        if is_model_failure:
            # Rule fallback
            if current_cpu > 80.0:
                recommended_replicas = current_replicas + 2
            elif current_cpu < 30.0:
                recommended_replicas = max(2, current_replicas - 1)
            else:
                recommended_replicas = current_replicas
            action_reason = "Fall back to rule-based safety loop (Low ML model confidence threshold)."
        else:
            # Predict replicas based on traffic index
            recommended_replicas = int(round(predicted_traffic / 50.0))
            recommended_replicas = max(self.policies["min_replicas"], min(recommended_replicas, self.policies["max_replicas"]))
            action_reason = "Predictive capacity adjustment."
            
        # Determine recommended action direction
        if recommended_replicas > current_replicas:
            action = f"SCALE_UP_{recommended_replicas - current_replicas}"
        elif recommended_replicas < current_replicas:
            action = f"SCALE_DOWN_{current_replicas - recommended_replicas}"
        else:
            action = "MAINTAIN_CAPACITY"

        # Calculate risk score
        prediction_uncertainty = 0.05 if model_confidence > 0.90 else 0.25
        sla_risk = 75.0 if current_latency > self.policies["sla_latency_threshold_ms"] else 12.0
        volatility = 0.15 if predicted_traffic < 500 else 0.40
        recent_scaling_freq = 1
        
        risk_class, risk_val = RiskEngine.calculate_risk(
            prediction_uncertainty=prediction_uncertainty,
            sla_risk_pct=sla_risk,
            anomaly_severity=anomaly_severity,
            workload_volatility=volatility,
            infrastructure_health="healthy",
            model_confidence=model_confidence,
            recent_scaling_freq=recent_scaling_freq
        )
        
        # Policy verification
        cooldown_elapsed = (time.time() - LAST_SCALING_TIMESTAMP) > self.policies["cooldown_seconds"]
        is_action_approved, policy_reason = PolicyEngine.validate_action(
            current_replicas=current_replicas,
            target_replicas=recommended_replicas,
            policies=self.policies,
            cooldown_elapsed=cooldown_elapsed,
            is_emergency=EMERGENCY_STOP
        )
        
        # Generate human readable description
        traffic_pct = int(round(((predicted_traffic - 100) / 100.0) * 100)) if predicted_traffic > 100 else 0
        if recommended_replicas > current_replicas:
            explanation = (
                f"Scale from {current_replicas} to {recommended_replicas} replicas because traffic is "
                f"expected to increase by {traffic_pct}%, CPU is at {current_cpu:.0f}%, and the current configuration "
                f"is projected to violate the {self.policies['sla_latency_threshold_ms']:.0f} ms latency SLA."
            )
        elif recommended_replicas < current_replicas:
            explanation = (
                f"Scale down from {current_replicas} to {recommended_replicas} replicas because CPU is low "
                f"({current_cpu:.0f}%) and traffic demand is stable at {predicted_traffic:.0f} rps."
            )
        else:
            explanation = f"Maintain {current_replicas} replicas. Current capacity matches forecasted workload metrics."

        # Audit decision execution
        execution_status = "REJECTED"
        if is_action_approved:
            if OPERATING_MODE == "SIMULATION":
                execution_status = "SIMULATED_SUCCESS"
            elif OPERATING_MODE == "APPROVAL":
                execution_status = "PENDING_OPERATOR_APPROVAL"
            elif OPERATING_MODE == "AUTONOMOUS":
                execution_status = "EXECUTED_SUCCESS"
                LAST_SCALING_TIMESTAMP = time.time()
        else:
            execution_status = "BLOCKED_BY_POLICY"
            
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_version": "v1.2.0-champion",
            "input_summary": {
                "current_replicas": current_replicas,
                "predicted_traffic": predicted_traffic,
                "current_cpu": current_cpu,
                "current_latency": current_latency,
                "anomaly_severity": anomaly_severity
            },
            "prediction": {
                "recommended_servers": recommended_replicas,
                "confidence": model_confidence
            },
            "action": action,
            "risk_score": risk_class,
            "policy_result": "PASSED" if is_action_approved else "FAILED",
            "reason": explanation if is_action_approved else policy_reason,
            "operator": operator if OPERATING_MODE != "AUTONOMOUS" else "AUTONOMOUS_DAEMON",
            "execution_result": execution_status
        }
        
        self.save_audit_log(audit_entry)
        
        return {
            "decision": audit_entry,
            "risk_value": risk_val,
            "operating_mode": OPERATING_MODE,
            "emergency_stop": EMERGENCY_STOP,
            "is_approved": is_action_approved
        }

    def generate_what_if(self, traffic_increase_pct: float, current_replicas: int) -> dict:
        """
        Calculates predicted outcomes for What-If scenarios.
        """
        forecasted_traffic = 200.0 * (1.0 + (traffic_increase_pct / 100.0))
        predicted_replicas = max(self.policies["min_replicas"], min(int(round(forecasted_traffic / 50.0)), self.policies["max_replicas"]))
        
        cost_diff = (predicted_replicas - current_replicas) * 0.12
        sla_impact = "HEALTHY" if predicted_replicas >= int(round(forecasted_traffic / 50.0)) else "CRITICAL_LATENCY_VIOLATION"
        carbon_impact = (predicted_replicas - current_replicas) * 1.5  # g/hr
        
        risk = "LOW"
        if traffic_increase_pct >= 50.0:
            risk = "MEDIUM"
        if traffic_increase_pct >= 100.0:
            risk = "HIGH"
            
        action = "MAINTAIN"
        if predicted_replicas > current_replicas:
            action = f"SCALE_UP_{predicted_replicas - current_replicas}"
        elif predicted_replicas < current_replicas:
            action = f"SCALE_DOWN_{current_replicas - predicted_replicas}"
            
        return {
            "traffic_increase_pct": traffic_increase_pct,
            "current_configuration": {"replicas": current_replicas, "cost_hourly": current_replicas * 0.12},
            "predicted_configuration": {"replicas": predicted_replicas, "cost_hourly": predicted_replicas * 0.12},
            "cost_difference": cost_diff,
            "sla_impact": sla_impact,
            "carbon_impact_ghr": carbon_impact,
            "risk": risk,
            "recommended_action": action
        }
