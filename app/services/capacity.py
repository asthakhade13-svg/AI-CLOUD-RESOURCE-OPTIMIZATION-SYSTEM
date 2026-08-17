from app.models.manager import model_manager
from src.capacity import calculate_required_servers, estimate_prediction_uncertainty as ep_uncertainty

def evaluate_capacity(prediction: float, current_servers: int, min_servers: int = 1, max_servers: int = 20, safety_margin: float = 0.10) -> dict:
    """Wraps capacity calculations."""
    return calculate_required_servers(
        prediction=prediction,
        current_servers=current_servers,
        min_servers=min_servers,
        max_servers=max_servers,
        safety_margin=safety_margin
    )

def estimate_uncertainty(scaled_record) -> dict:
    """Wraps prediction uncertainty calculations using model_manager cached model."""
    return ep_uncertainty(model_manager.capacity_model, scaled_record)
