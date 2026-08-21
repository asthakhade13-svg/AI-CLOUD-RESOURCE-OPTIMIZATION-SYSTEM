# src/multi_objective_optimizer.py

import math
import numpy as np
from typing import Dict, Any, List, Tuple

# Database profiles for cloud instances
INSTANCE_PROFILES = {
    "t3.medium": {"price": 0.0416, "cpus": 2, "mem": 4.0, "p_idle": 4.0, "p_max": 15.0, "capacity_req_sec": 300.0},
    "m5.large": {"price": 0.0960, "cpus": 2, "mem": 8.0, "p_idle": 8.0, "p_max": 30.0, "capacity_req_sec": 600.0},
    "m5.xlarge": {"price": 0.1920, "cpus": 4, "mem": 16.0, "p_idle": 12.0, "p_max": 60.0, "capacity_req_sec": 1200.0},
    "c5.large": {"price": 0.0850, "cpus": 2, "mem": 4.0, "p_idle": 6.0, "p_max": 25.0, "capacity_req_sec": 700.0},
    "c5.xlarge": {"price": 0.1700, "cpus": 4, "mem": 8.0, "p_idle": 10.0, "p_max": 50.0, "capacity_req_sec": 1400.0}
}

# Grid carbon intensity rates (gCO2eq/kWh)
REGIONS_CARBON = {
    "us-east-1": 380.0,     # N. Virginia (fossil-heavy mix)
    "eu-central-1": 330.0,  # Frankfurt
    "ap-east-1": 710.0,     # Hong Kong (coal-heavy grid)
    "us-west-2": 80.0,      # Oregon (hydro & green mix)
    "eu-west-6": 20.0       # Stockholm (extremely green grid)
}

def evaluate_config(
    replicas: int,
    instance_type: str,
    cpu_alloc: float,
    mem_alloc: float,
    region: str,
    workload: Dict[str, Any],
    current_config: Dict[str, Any],
    target_latency: float = 200.0
) -> Dict[str, Any]:
    """
    Computes performance, cost, energy, carbon, and stability metrics for 
    a single cloud sizing configuration candidate.
    """
    requests = float(workload.get("request_rate", 500.0))
    users = float(workload.get("active_users", 1000.0))
    
    inst = INSTANCE_PROFILES[instance_type]
    price = inst["price"]
    
    # 1. Capacity multiplier based on vertical CPU & memory allocation limits
    # e.g., allocating only 50% CPU bounds decreases max replica throughput capacity
    cpu_factor = cpu_alloc / inst["cpus"]
    mem_factor = mem_alloc / inst["mem"]
    capacity_factor = min(cpu_factor, mem_factor)
    
    pod_capacity = inst["capacity_req_sec"] * capacity_factor
    total_capacity = replicas * pod_capacity
    
    # 2. Resource Utilizations
    if total_capacity <= 0:
        cpu_util = 100.0
    else:
        cpu_util = (requests / total_capacity) * 100.0
        
    memory_comfort = replicas * (inst["mem"] * 400.0) # Comfort factor: 400 users per GB
    if memory_comfort <= 0:
        memory_util = 100.0
    else:
        memory_util = 20.0 + (users / memory_comfort) * 55.0
        
    cpu_util = max(5.0, min(100.0, cpu_util))
    memory_util = max(10.0, min(100.0, memory_util))
    
    # 3. Queuing response latency
    base_latency = 75.0 # ms
    if cpu_util <= 75.0:
        latency = base_latency + (cpu_util / 75.0) * 20.0
    else:
        exponent = (cpu_util - 75.0) / 8.0
        latency = base_latency + 20.0 + 15.0 * math.exp(exponent)
        
    latency = max(30.0, min(2000.0, latency))
    
    # 4. Error rates
    error_rate = 0.0
    if cpu_util > 88.0:
        error_rate += (cpu_util - 88.0) * 0.6
    if memory_util > 92.0:
        error_rate += (memory_util - 92.0) * 0.9
    error_rate = max(0.0, min(100.0, error_rate))
    
    # 5. SLA Violations status
    sla_violated = 1.0 if (latency > target_latency or error_rate > 1.0) else 0.0
    
    # 6. Energy Consumption (Watts)
    # Power = Idle power + Utilization power delta
    power = replicas * (inst["p_idle"] + (inst["p_max"] - inst["p_idle"]) * (cpu_util / 100.0))
    energy_kwh = (power / 1000.0) * 1.0 # 1 hour duration
    
    # 7. Carbon footprint (gCO2eq)
    carbon_intensity = REGIONS_CARBON[region]
    carbon_g = energy_kwh * carbon_intensity
    
    # 8. Infrastructure Cost ($/hr)
    cost = replicas * price
    
    # 9. Over/Under-provisioning metrics
    pred_req_servers = float(workload.get("predicted_required_servers", replicas))
    optimal_capacity = pred_req_servers * inst["capacity_req_sec"]
    overprovisioning = max(0.0, total_capacity - optimal_capacity)
    underprovisioning = max(0.0, optimal_capacity - total_capacity)
    
    # 10. Stability Thrashing Penalty (higher distance means higher penalty)
    instability = 0.0
    if replicas != current_config.get("replicas", replicas):
        instability += 1.0
    if instance_type != current_config.get("instance_type", instance_type):
        instability += 1.5
    if region != current_config.get("region", region):
        instability += 2.0
    if abs(cpu_alloc - current_config.get("cpu_alloc", cpu_alloc)) > 0.01:
        instability += 0.5
        
    return {
        "replicas": replicas,
        "instance_type": instance_type,
        "cpu_alloc": cpu_alloc,
        "mem_alloc": mem_alloc,
        "region": region,
        "cost": cost,
        "latency": latency,
        "sla_violated": sla_violated,
        "energy": energy_kwh,
        "carbon": carbon_g,
        "overprovisioning": overprovisioning,
        "underprovisioning": underprovisioning,
        "instability": instability,
        "cpu_util": cpu_util,
        "memory_util": memory_util,
        "error_rate": error_rate
    }

def is_dominated(candidate: Dict[str, Any], other: Dict[str, Any]) -> bool:
    """
    Checks if a candidate configuration is dominated by another configuration.
    
    Domination rule: Other is better or equal in ALL objectives, and strictly
    better in at least one.
    Objectives to minimize: cost, latency, sla_violated, energy, carbon, overprovisioning, underprovisioning, instability.
    """
    keys = ["cost", "latency", "sla_violated", "energy", "carbon", "overprovisioning", "underprovisioning", "instability"]
    
    better_in_at_least_one = False
    for k in keys:
        val_cand = candidate[k]
        val_other = other[k]
        
        if val_other > val_cand:
            # Other is worse in this objective, so other cannot dominate candidate
            return False
        if val_other < val_cand:
            better_in_at_least_one = True
            
    return better_in_at_least_one

def solve_pareto_grid_search(
    workload: Dict[str, Any],
    current_config: Dict[str, Any],
    constraints: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Runs a full discrete grid search across replicas, instance types, and regions
    to calculate metrics and extract the non-dominated Pareto Front set.
    """
    constraints = constraints or {}
    min_rep = int(constraints.get("min_replicas", 2))
    max_rep = int(constraints.get("max_replicas", 10))
    target_latency = float(constraints.get("target_latency", 200.0))
    max_error = float(constraints.get("max_error_rate", 1.0))
    
    # Grid options
    replica_range = range(min_rep, max_rep + 1)
    instance_types = list(INSTANCE_PROFILES.keys())
    regions = list(REGIONS_CARBON.keys())
    
    # Vertical options (CPU Cores & Memory GiB)
    cpu_options = [1.0, 2.0, 4.0]
    mem_options = [2.0, 4.0, 8.0, 16.0]
    
    all_candidates = []
    
    for r in replica_range:
        for it in instance_types:
            for reg in regions:
                inst_spec = INSTANCE_PROFILES[it]
                
                # Filter vertical configurations matching instance physical limits
                valid_cpus = [c for c in cpu_options if c <= inst_spec["cpus"]]
                valid_mems = [m for m in mem_options if m <= inst_spec["mem"]]
                
                for c in valid_cpus:
                    for m in valid_mems:
                        res = evaluate_config(
                            replicas=r,
                            instance_type=it,
                            cpu_alloc=c,
                            mem_alloc=m,
                            region=reg,
                            workload=workload,
                            current_config=current_config,
                            target_latency=target_latency
                        )
                        
                        # Apply hard constraint filtering
                        if res["error_rate"] <= max_error:
                            all_candidates.append(res)
                            
    # Extract Pareto-front (non-dominated filtering)
    pareto_front = []
    for cand in all_candidates:
        dominated = False
        for other in all_candidates:
            if is_dominated(cand, other):
                dominated = True
                break
        if not dominated:
            pareto_front.append(cand)
            
    # Fallback to all if pareto_front is empty (safety backup)
    return pareto_front if pareto_front else all_candidates

def select_optimal_config(
    pareto_front: List[Dict[str, Any]],
    weights: Dict[str, float]
) -> Tuple[Dict[str, Any], float]:
    """
    Selects the optimal configuration from the Pareto front using 
    normalized weighted objectives scoring.
    """
    if not pareto_front:
        return {}, 0.0
        
    keys = ["cost", "latency", "sla_violated", "energy", "carbon", "overprovisioning", "underprovisioning", "instability"]
    
    # Calculate min-max bounds in Pareto front to normalize scores to [0.0, 1.0]
    bounds = {}
    for k in keys:
        vals = [c[k] for c in pareto_front]
        min_v = min(vals)
        max_v = max(vals)
        bounds[k] = (min_v, max_v if max_v > min_v else min_v + 1e-5)
        
    best_score = float('inf')
    best_config = pareto_front[0]
    
    w_cost = float(weights.get("cost", 1.0))
    w_lat = float(weights.get("latency", 1.0))
    w_sla = float(weights.get("sla", 1.0))
    w_energy = float(weights.get("energy", 0.5))
    w_carb = float(weights.get("carbon", 1.0))
    w_over = float(weights.get("overprovisioning", 0.5))
    w_under = float(weights.get("underprovisioning", 0.8))
    w_stab = float(weights.get("stability", 0.3))
    
    for cand in pareto_front:
        # Normalize objectives
        norm_cost = (cand["cost"] - bounds["cost"][0]) / bounds["cost"][1]
        norm_lat = (cand["latency"] - bounds["latency"][0]) / bounds["latency"][1]
        norm_sla = cand["sla_violated"] # already 0 or 1
        norm_energy = (cand["energy"] - bounds["energy"][0]) / bounds["energy"][1]
        norm_carb = (cand["carbon"] - bounds["carbon"][0]) / bounds["carbon"][1]
        norm_over = (cand["overprovisioning"] - bounds["overprovisioning"][0]) / bounds["overprovisioning"][1]
        norm_under = (cand["underprovisioning"] - bounds["underprovisioning"][0]) / bounds["underprovisioning"][1]
        norm_stab = (cand["instability"] - bounds["instability"][0]) / bounds["instability"][1]
        
        score = (
            w_cost * norm_cost +
            w_lat * norm_lat +
            w_sla * norm_sla +
            w_energy * norm_energy +
            w_carb * norm_carb +
            w_over * norm_over +
            w_under * norm_under +
            w_stab * norm_stab
        )
        
        if score < best_score:
            best_score = score
            best_config = cand
            
    # Calculate optimization score on a 0-100 scale (100 is best, i.e., lowest penalty score)
    # score = 100 - (best_score / total_weights) * 100
    total_w = w_cost + w_lat + w_sla + w_energy + w_carb + w_over + w_under + w_stab
    opt_score = max(0.0, min(100.0, 100.0 * (1.0 - (best_score / total_w))))
    
    return best_config, opt_score

def run_multi_objective_optimization(
    workload: Dict[str, Any],
    current_config: Dict[str, Any],
    weights: Dict[str, float] = None,
    constraints: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Main Multi-Objective optimizer router execution.
    Runs Pareto front analysis and returns recommended config alongside
    scenario comparison configurations.
    """
    weights = weights or {}
    constraints = constraints or {}
    
    # 1. Compute Pareto Front
    pareto_front = solve_pareto_grid_search(workload, current_config, constraints)
    
    # 2. Extract scenarios
    # Balanced (configured weights)
    balanced_config, balanced_score = select_optimal_config(pareto_front, weights)
    
    # Cost-Optimal (cost = 1.0, others = 0)
    cost_opt, _ = select_optimal_config(pareto_front, {"cost": 1.0})
    
    # Performance-Optimal (latency = 1.0, sla = 1.0, others = 0)
    perf_opt, _ = select_optimal_config(pareto_front, {"latency": 1.0, "sla": 1.0})
    
    # Carbon-Optimal (carbon = 1.0, others = 0)
    carb_opt, _ = select_optimal_config(pareto_front, {"carbon": 1.0})
    
    # Current config evaluation
    current_eval = evaluate_config(
        replicas=current_config.get("replicas", 5),
        instance_type=current_config.get("instance_type", "m5.large"),
        cpu_alloc=current_config.get("cpu_alloc", 2.0),
        mem_alloc=current_config.get("mem_alloc", 8.0),
        region=current_config.get("region", "us-east-1"),
        workload=workload,
        current_config=current_config,
        target_latency=float(constraints.get("target_latency", 200.0))
    )
    
    # Format comparison list
    comparisons = [
        {"scenario": "Current Configuration", **current_eval},
        {"scenario": "Cost-Optimal Configuration", **cost_opt},
        {"scenario": "Performance-Optimal Configuration", **perf_opt},
        {"scenario": "Balanced Configuration", **balanced_config},
        {"scenario": "Carbon-Optimal Configuration", **carb_opt}
    ]
    
    # Setup recommended config dict matching output schema
    rec_conf = {
        "replicas": int(balanced_config["replicas"]),
        "instance_type": balanced_config["instance_type"],
        "cpu_allocation": float(balanced_config["cpu_alloc"]),
        "memory_allocation": float(balanced_config["mem_alloc"]),
        "region": balanced_config["region"]
    }
    
    # Sizing reason summary text
    reason = (
        f"Multi-Objective optimizer recommends {rec_conf['replicas']}x {rec_conf['instance_type']} "
        f"in {rec_conf['region']} (allocated {rec_conf['cpu_allocation']} cores / {rec_conf['memory_allocation']} GiB). "
        f"This setup achieves an optimization index score of {balanced_score:.1f}/100, "
        f"balancing hosting cost (${balanced_config['cost']:.4f}/hr), latency ({balanced_config['latency']:.1f}ms), "
        f"and carbon footprint ({balanced_config['carbon']:.1f} gCO2eq/hr)."
    )
    
    return {
        "recommended_configuration": rec_conf,
        "estimated_cost": float(round(balanced_config["cost"], 4)),
        "estimated_latency": float(round(balanced_config["latency"], 1)),
        "sla_status": "HEALTHY" if balanced_config["sla_violated"] == 0.0 else "VIOLATED",
        "energy_consumption": float(round(balanced_config["energy"], 4)),
        "carbon_emissions": float(round(balanced_config["carbon"], 2)),
        "overprovisioning": float(round(balanced_config["overprovisioning"], 1)),
        "optimization_score": float(round(balanced_score, 1)),
        "reason": reason,
        "comparisons": comparisons,
        "pareto_front": pareto_front
    }
