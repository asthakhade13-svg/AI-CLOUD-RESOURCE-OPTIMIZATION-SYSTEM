import math

def optimize_capacity_cost(
    predicted_required_servers: float,
    current_servers: int,
    server_cost_per_hour: float,
    min_servers: int = 1,
    max_servers: int = 20,
    sla_penalty_weight: float = 5.0,
    overprovisioning_weight: float = 0.5
) -> dict:
    """
    Evaluates multiple server configurations to find the count that minimizes 
    the total cost objective function:
    
    Total Cost = Infrastructure Cost + SLA Penalty + Overprovisioning Penalty
    
    Returns the optimized server count along with detailed cost comparisons, 
    estimated savings, and SLA status details.
    """
    # Enforce constraints
    min_servers = max(1, min_servers)
    max_servers = max(min_servers, max_servers)
    current_servers = max(1, current_servers)
    
    best_cost = float('inf')
    best_servers = min_servers
    
    # Evaluate all candidate server configurations
    for S in range(min_servers, max_servers + 1):
        infra_cost = S * server_cost_per_hour
        
        # SLA Penalty is incurred if the capacity is under-provisioned
        deficit = max(0.0, predicted_required_servers - S)
        sla_penalty = deficit * sla_penalty_weight * server_cost_per_hour
        
        # Overprovisioning Penalty is incurred if we have excess idle servers
        surplus = max(0.0, S - predicted_required_servers)
        over_penalty = surplus * overprovisioning_weight * server_cost_per_hour
        
        total_cost = infra_cost + sla_penalty + over_penalty
        
        # Select the configuration that minimizes cost
        if total_cost < best_cost:
            best_cost = total_cost
            best_servers = S
        elif total_cost == best_cost:
            # If cost is tied, prefer the configuration closer to the predicted required servers
            if abs(S - predicted_required_servers) < abs(best_servers - predicted_required_servers):
                best_servers = S
                
    # Calculate costs for the optimized server count
    opt_hourly_cost = best_servers * server_cost_per_hour
    opt_daily_cost = opt_hourly_cost * 24
    opt_monthly_cost = opt_daily_cost * 30
    
    # Calculate costs for the current server count
    curr_hourly_cost = current_servers * server_cost_per_hour
    curr_daily_cost = curr_hourly_cost * 24
    
    # Projections and Savings
    # Savings are positive if optimized servers < current servers (scaling down saves money)
    # Savings are negative if optimized servers > current servers (scaling up costs more)
    daily_savings = curr_daily_cost - opt_daily_cost
    monthly_savings = daily_savings * 30
    
    # Estimate the under-provisioning cost of the CURRENT setup if it violates SLA requirements
    curr_deficit = max(0.0, predicted_required_servers - current_servers)
    curr_sla_penalty_hourly = curr_deficit * sla_penalty_weight * server_cost_per_hour
    curr_sla_penalty_daily = curr_sla_penalty_hourly * 24
    
    sla_status = "SATISFIED" if best_servers >= predicted_required_servers else "VIOLATED"
    
    # Generate optimization summary reason
    if best_servers < current_servers:
        reason = (
            f"Optimized server capacity to {best_servers} from {current_servers}. "
            f"Workload demands allow a safe scale-down, saving an estimated ${daily_savings:.2f}/day "
            f"(${monthly_savings:.2f}/month) while satisfying SLA targets."
        )
    elif best_servers > current_servers:
        reason = (
            f"Optimized server capacity to {best_servers} from {current_servers}. "
            f"Scale-up required to prevent SLA violations. Current setup incurs a projected "
            f"SLA breach penalty of ${curr_sla_penalty_daily:.2f}/day. Investing in extra servers "
            f"minimizes total system risk."
        )
    else:
        reason = (
            f"Server count of {best_servers} is optimal. Changing capacity is not recommended. "
            f"This configuration minimizes over-provisioning cost while maintaining SLA targets."
        )
        
    return {
        "recommended_servers": int(best_servers),
        "hourly_cost": round(opt_hourly_cost, 4),
        "estimated_daily_cost": round(opt_daily_cost, 4),
        "estimated_monthly_cost": round(opt_monthly_cost, 4),
        "estimated_savings_hourly": round(curr_hourly_cost - opt_hourly_cost, 4),
        "estimated_savings_daily": round(daily_savings, 4),
        "estimated_savings_monthly": round(monthly_savings, 4),
        "underprovisioning_cost_daily": round(curr_sla_penalty_daily, 4),
        "sla_status": sla_status,
        "optimization_reason": reason
    }
