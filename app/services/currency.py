import os
import json
import time
import requests
from typing import Dict, Any, Optional
from app.utils.logging import logger

# Cache file path
CACHE_FILE = os.path.join("data", "cached_rates.json")
CACHE_DURATION_SECS = 3600  # 1 hour config

# Default fallback rates (USD as base)
FALLBACK_RATES = {
    "USD": 1.0,
    "INR": 84.30,
    "EUR": 0.92,
    "GBP": 0.78,
    "JPY": 154.50,
    "AUD": 1.52,
    "CAD": 1.39,
    "SGD": 1.34
}

class ExchangeRateProvider:
    def get_rates(self) -> Optional[Dict[str, float]]:
        raise NotImplementedError

class APIProvider(ExchangeRateProvider):
    def __init__(self, url: str = "https://open.er-api.com/v6/latest/USD", timeout: float = 5.0):
        self.url = url
        self.timeout = timeout

    def get_rates(self) -> Optional[Dict[str, float]]:
        try:
            logger.info(f"Fetching live exchange rates from API: {self.url}")
            response = requests.get(self.url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if "rates" in data:
                    rates = data["rates"]
                    # Filter and clean only the ones we support
                    filtered_rates = {}
                    for currency in FALLBACK_RATES.keys():
                        if currency in rates:
                            filtered_rates[currency] = float(rates[currency])
                    return filtered_rates
            logger.warn(f"API Provider returned status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching from API Provider: {e}")
        return None

class FallbackProvider(ExchangeRateProvider):
    def get_rates(self) -> Dict[str, float]:
        logger.info("Using hardcoded fallback exchange rates")
        return FALLBACK_RATES.copy()

class CurrencyService:
    def __init__(self, api_url: str = "https://open.er-api.com/v6/latest/USD", cache_file: str = CACHE_FILE):
        self.api_provider = APIProvider(api_url)
        self.fallback_provider = FallbackProvider()
        self.cache_file = cache_file

    def _read_cache(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file, "r") as f:
                data = json.load(f)
                # Verify structure
                if "rates" in data and "timestamp" in data:
                    return data
        except Exception as e:
            logger.error(f"Error reading cache file {self.cache_file}: {e}")
        return None

    def _write_cache(self, rates: Dict[str, float], source: str):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            data = {
                "rates": rates,
                "timestamp": time.time(),
                "source": source
            }
            with open(self.cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing cache file: {e}")

    def get_exchange_data(self) -> Dict[str, Any]:
        now = time.time()
        
        # 1. Try to read cache first to see if it is still fresh
        cache = self._read_cache()
        if cache:
            age = now - cache["timestamp"]
            if age < CACHE_DURATION_SECS:
                logger.info(f"Returning cached exchange rates (age: {int(age)}s)")
                return {
                    "rates": cache["rates"],
                    "timestamp": cache["timestamp"],
                    "source": cache.get("source", "Live API"),
                    "status": "fresh"
                }

        # 2. Try API provider (Live rates)
        live_rates = self.api_provider.get_rates()
        if live_rates:
            self._write_cache(live_rates, "Live API")
            return {
                "rates": live_rates,
                "timestamp": now,
                "source": "Live API",
                "status": "live"
            }

        # 3. Fallback to cached last-known rate (even if stale)
        if cache:
            logger.warn("Live API failed. Using last-known stale cached rates.")
            return {
                "rates": cache["rates"],
                "timestamp": cache["timestamp"],
                "source": cache.get("source", "Live API"),
                "status": "stale"
            }

        # 4. Fallback to configured fallback rates
        fallback_rates = self.fallback_provider.get_rates()
        self._write_cache(fallback_rates, "Fallback Rates")
        return {
            "rates": fallback_rates,
            "timestamp": now,
            "source": "Fallback Rates",
            "status": "fallback"
        }

    def convert(self, amount: float, from_curr: str, to_curr: str, rates: Dict[str, float]) -> float:
        """Converts an amount from one currency to another using the supplied rates dict."""
        if from_curr == to_curr:
            return amount
        
        # Ensure base rates are present
        if from_curr not in rates or to_curr not in rates:
            raise ValueError(f"Exchange rate not available for {from_curr} or {to_curr}")
            
        # USD is the base currency (1.0). Convert from_curr to USD, then USD to to_curr
        amount_in_usd = amount / rates[from_curr]
        return amount_in_usd * rates[to_curr]
