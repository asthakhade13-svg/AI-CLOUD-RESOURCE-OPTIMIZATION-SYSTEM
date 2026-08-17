from src.controller import AutoscalingController

# Global stateful controller instance
_autoscaler = None

def get_autoscaler(current_servers: int = 5) -> AutoscalingController:
    """Retrieves or initializes the global stateful AutoscalingController."""
    global _autoscaler
    if _autoscaler is None:
        _autoscaler = AutoscalingController(current_servers=current_servers)
    return _autoscaler
