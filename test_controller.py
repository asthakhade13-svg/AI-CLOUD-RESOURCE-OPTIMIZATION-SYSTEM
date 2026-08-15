import pytest
from src.controller import AutoscalingController

def test_controller_cooldown():
    # Instantiate with current_servers=5, cooldown_periods=3
    controller = AutoscalingController(
        current_servers=5,
        scale_up_cpu_threshold=80.0,
        cooldown_periods=3,
        scale_up_confirmations=1
    )
    
    # 1. First decision: scaling is ready (seeded as ready)
    # CPU: 85% (exceeds 80%), recommended: 7 (requires scale up)
    # scale_up_confirmations is 1, so it should trigger immediately
    decision1 = controller.make_scaling_decision(cpu_usage=85.0, predicted_servers=7.0, recommended_servers=7)
    assert decision1["action"] == "SCALE_UP"
    assert decision1["recommended_servers"] == 7
    assert decision1["cooldown_active"] is False
    assert controller.current_server_count == 7
    assert controller.ticks_since_last_scaling == 0
    
    # 2. Second decision: tick 1 since scale action. Lock in cooldown!
    decision2 = controller.make_scaling_decision(cpu_usage=85.0, predicted_servers=8.0, recommended_servers=8)
    assert decision2["action"] == "NO_ACTION"
    assert decision2["recommended_servers"] == 7
    assert decision2["cooldown_active"] is True
    assert controller.current_server_count == 7
    
    # 3. Third decision: tick 2 since scale action. Lock in cooldown!
    decision3 = controller.make_scaling_decision(cpu_usage=85.0, predicted_servers=8.0, recommended_servers=8)
    assert decision3["action"] == "NO_ACTION"
    assert decision3["recommended_servers"] == 7
    assert decision3["cooldown_active"] is True
    
    # 4. Fourth decision: tick 3 since scale action. Lock in cooldown!
    decision4 = controller.make_scaling_decision(cpu_usage=85.0, predicted_servers=8.0, recommended_servers=8)
    assert decision4["action"] == "NO_ACTION"
    assert decision4["recommended_servers"] == 7
    assert decision4["cooldown_active"] is True
    
    # 5. Fifth decision: tick 4. Cooldown has elapsed (4 > 3). Action should be allowed!
    decision5 = controller.make_scaling_decision(cpu_usage=85.0, predicted_servers=8.0, recommended_servers=8)
    assert decision5["action"] == "SCALE_UP"
    assert decision5["recommended_servers"] == 8
    assert decision5["cooldown_active"] is False

def test_controller_consecutive_confirmations():
    # Instantiate with current_servers=5, scale_up_confirmations=3
    controller = AutoscalingController(
        current_servers=5,
        scale_up_cpu_threshold=80.0,
        cooldown_periods=0,  # No cooldown for easy isolation
        scale_up_confirmations=3
    )
    
    # Tick 1: CPU=85%, recommended=6. High load but no confirmation yet.
    d1 = controller.make_scaling_decision(cpu_usage=85.0, predicted_servers=6.0, recommended_servers=6)
    assert d1["action"] == "NO_ACTION"
    assert controller.scale_up_consecutive_ticks == 1
    
    # Tick 2: CPU=85%, recommended=6. Confirm count = 2.
    d2 = controller.make_scaling_decision(cpu_usage=85.0, predicted_servers=6.0, recommended_servers=6)
    assert d2["action"] == "NO_ACTION"
    assert controller.scale_up_consecutive_ticks == 2
    
    # Tick 3: CPU=85%, recommended=6. Confirm count = 3. Trigger scale up!
    d3 = controller.make_scaling_decision(cpu_usage=85.0, predicted_servers=6.0, recommended_servers=6)
    assert d3["action"] == "SCALE_UP"
    assert d3["recommended_servers"] == 6
    assert controller.scale_up_consecutive_ticks == 0

def test_controller_scale_steps_limit():
    # current_servers = 5, recommended = 10, max_scale_up_step = 2
    controller = AutoscalingController(
        current_servers=5,
        scale_up_cpu_threshold=80.0,
        cooldown_periods=0,
        scale_up_confirmations=1,
        max_scale_up_step=2
    )
    decision = controller.make_scaling_decision(cpu_usage=85.0, predicted_servers=10.0, recommended_servers=10)
    assert decision["action"] == "SCALE_UP"
    # Should only scale up by 2 (from 5 to 7)
    assert decision["recommended_servers"] == 7
    assert controller.current_server_count == 7
