# rl/actions.py

from typing import Dict

# Action index mapping definitions
ACTIONS_MAP: Dict[int, str] = {
    0: "SCALE_DOWN_2",
    1: "SCALE_DOWN_1",
    2: "NO_ACTION",
    3: "SCALE_UP_1",
    4: "SCALE_UP_2"
}

ACTION_STEPS: Dict[str, int] = {
    "SCALE_DOWN_2": -2,
    "SCALE_DOWN_1": -1,
    "NO_ACTION": 0,
    "SCALE_UP_1": 1,
    "SCALE_UP_2": 2
}

def idx_to_action(idx: int) -> str:
    """Converts a discrete action index to its string name representation."""
    return ACTIONS_MAP.get(idx, "NO_ACTION")

def action_to_step(action_name: str) -> int:
    """Converts a scaling action name to its corresponding replica count step change."""
    return ACTION_STEPS.get(action_name, 0)

def idx_to_step(idx: int) -> int:
    """Converts a discrete action index directly to its replica count step change."""
    action_name = idx_to_action(idx)
    return action_to_step(action_name)

def get_action_count() -> int:
    """Returns the total number of discrete actions in the space."""
    return len(ACTIONS_MAP)
