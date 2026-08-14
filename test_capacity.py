import pytest
import numpy as np
from src.capacity import calculate_required_servers, estimate_prediction_uncertainty
from sklearn.ensemble import RandomForestRegressor

def test_calculate_required_servers_ceil_logic():
    # Raw prediction: 5.4, Safety margin: 10% (0.10)
    # Projected: 5.4 * 1.10 = 5.94
    # Recommended: ceil(5.94) = 6
    capacity = calculate_required_servers(
        prediction=5.4,
        current_servers=4,
        min_servers=1,
        max_servers=20,
        safety_margin=0.10
    )
    assert capacity["predicted_servers"] == 5.4
    assert capacity["recommended_servers"] == 6
    assert capacity["safety_margin"] == 0.10
    assert capacity["safety_buffer"] == pytest.approx(0.54)

def test_calculate_required_servers_min_clamp():
    # Raw prediction: 0.1, Safety margin: 10%
    # Projected: 0.11 -> ceil = 1
    # If min_servers = 3, clamped recommended = 3
    capacity = calculate_required_servers(
        prediction=0.1,
        current_servers=4,
        min_servers=3,
        max_servers=10,
        safety_margin=0.10
    )
    assert capacity["recommended_servers"] == 3

def test_calculate_required_servers_max_clamp():
    # Raw prediction: 12.0, Safety margin: 20%
    # Projected: 12.0 * 1.20 = 14.4 -> ceil = 15
    # If max_servers = 10, clamped recommended = 10
    capacity = calculate_required_servers(
        prediction=12.0,
        current_servers=4,
        min_servers=1,
        max_servers=10,
        safety_margin=0.20
    )
    assert capacity["recommended_servers"] == 10

def test_estimate_prediction_uncertainty_mock():
    # Create simple mock model with estimators
    class DummyTree:
        def __init__(self, val):
            self.val = val
        def predict(self, X):
            return [self.val]
            
    class DummyForest:
        def __init__(self):
            self.estimators_ = [DummyTree(4.0), DummyTree(5.0), DummyTree(6.0)]
            
    forest = DummyForest()
    uncertainty = estimate_prediction_uncertainty(forest, None)
    
    assert uncertainty["estimators_evaluated"] == 3
    # Std dev of [4.0, 5.0, 6.0] is np.std([4,5,6]) = sqrt(((4-5)**2 + (5-5)**2 + (6-5)**2)/3) = sqrt(2/3) = 0.8165
    assert uncertainty["uncertainty_std"] == pytest.approx(0.81649658)
    assert uncertainty["confidence_interval_lower"] == pytest.approx(4.05)  # 2.5 percentile of [4,5,6]
    assert uncertainty["confidence_interval_upper"] == pytest.approx(5.95)  # 97.5 percentile
