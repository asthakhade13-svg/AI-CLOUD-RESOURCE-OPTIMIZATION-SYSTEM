import pytest
from app.services.currency import CurrencyService, FALLBACK_RATES

def test_currency_fallback_rates():
    service = CurrencyService()
    data = service.get_exchange_data()
    
    assert "rates" in data
    assert "timestamp" in data
    assert "source" in data
    assert "status" in data
    
    # Check that USD is base
    assert data["rates"]["USD"] == 1.0
    
    # Check that INR and other major currencies are present
    for curr in ["INR", "EUR", "GBP", "JPY", "AUD", "CAD", "SGD"]:
        assert curr in data["rates"]
        assert data["rates"][curr] > 0

def test_currency_conversion():
    service = CurrencyService()
    rates = FALLBACK_RATES
    
    # Test USD to INR
    inr_amount = service.convert(100.0, "USD", "INR", rates)
    assert inr_amount == pytest.approx(100.0 * rates["INR"])
    
    # Test USD to JPY
    jpy_amount = service.convert(5.5, "USD", "JPY", rates)
    assert jpy_amount == pytest.approx(5.5 * rates["JPY"])
    
    # Test converting back and forth
    usd_again = service.convert(inr_amount, "INR", "USD", rates)
    assert usd_again == pytest.approx(100.0)
    
    # Test JPY to EUR
    eur_amount = service.convert(1000.0, "JPY", "EUR", rates)
    usd_middle = 1000.0 / rates["JPY"]
    expected_eur = usd_middle * rates["EUR"]
    assert eur_amount == pytest.approx(expected_eur)

def test_currency_edge_values():
    service = CurrencyService()
    rates = FALLBACK_RATES
    
    # Test 0 conversion
    assert service.convert(0.0, "USD", "INR", rates) == 0.0
    
    # Test 1
    assert service.convert(1.0, "USD", "INR", rates) == rates["INR"]
    
    # Test 1,000,000
    assert service.convert(1000000.0, "USD", "INR", rates) == pytest.approx(1000000.0 * rates["INR"])
    
    # Test negative value
    assert service.convert(-10.0, "USD", "INR", rates) == pytest.approx(-10.0 * rates["INR"])
    
    # Test conversion error for invalid currency
    with pytest.raises(ValueError):
        service.convert(100.0, "USD", "INVALID", rates)

def test_cache_mechanism(tmp_path):
    # Use a temporary cache file to test loading/writing cache
    temp_cache = str(tmp_path / "temp_rates.json")
    service = CurrencyService(cache_file=temp_cache)
    
    # Initial query should write cache
    data1 = service.get_exchange_data()
    assert data1["status"] in ["live", "fresh", "fallback"]
    
    # Query again within cache duration - should load from cache ("fresh" or "stale")
    data2 = service.get_exchange_data()
    assert data2["status"] == "fresh"
    assert data2["rates"] == data1["rates"]
