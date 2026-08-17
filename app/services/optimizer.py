from src.optimizer import optimize_capacity_cost

def optimize_cost(
    predicted_required_servers: float,
    current_servers: int,
    server_cost_per_hour: float,
    min_servers: int = 1,
    max_servers: int = 20,
    sla_penalty_weight: float = 5.0,
    overprovisioning_weight: float = 0.5
) -> dict:
    """Wraps cost-aware optimization calculations."""
    return optimize_capacity_cost(
        predicted_required_servers=predicted_required_servers,
        current_servers=current_servers,
        server_cost_per_hour=server_cost_per_hour,
        min_servers=min_servers,
        max_servers=max_servers,
        sla_penalty_weight=sla_penalty_weight,
        overprovisioning_weight=overprovisioning_weight
    )
