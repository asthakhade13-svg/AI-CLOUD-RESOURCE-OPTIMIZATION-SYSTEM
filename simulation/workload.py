# simulation/workload.py

import numpy as np
import pandas as pd
import random
import os
from typing import Dict, Any, List, Optional

class WorkloadGenerator:
    """
    Generates complex, configurable, multi-pattern workload profiles (active users,
    request rates, and network throughput) for the Cloud Digital Twin.
    """
    def __init__(self, csv_cache: Optional[Dict[str, pd.DataFrame]] = None):
        self.csv_cache = csv_cache or {}

    def generate_step(self, step: int, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates and sums up all active workload patterns for a given timestep.
        
        Each pattern dict in patterns:
        {
            "type": "constant" | "linear" | "sinusoidal" | "diurnal" | "spike" | "drop" | "random" | "csv",
            "params": { ... }
        }
        """
        users = 0.0
        requests_ratio = 4.2  # baseline requests per active user session
        network_ratio = 0.6   # baseline Mbps per request/sec
        
        for pat in patterns:
            p_type = pat.get("type", "constant").lower()
            params = pat.get("params", {})
            
            if p_type == "constant":
                users += float(params.get("users", 100.0))
                
            elif p_type == "linear":
                slope = float(params.get("slope", 1.0))
                base = float(params.get("base", 100.0))
                users += base + slope * step
                
            elif p_type == "sinusoidal":
                amplitude = float(params.get("amplitude", 500.0))
                period = float(params.get("period", 288.0))  # 288 steps = 24h
                phase = float(params.get("phase", 0.0))
                bias = float(params.get("bias", 1000.0))
                users += bias + amplitude * np.sin(2 * np.pi * step / period + phase)
                
            elif p_type == "diurnal":
                # Double peak: morning peak (e.g. step 100) and evening peak (e.g. step 200)
                bias = float(params.get("bias", 1000.0))
                peak1_center = float(params.get("peak1_center", 108.0)) # ~9 AM
                peak2_center = float(params.get("peak2_center", 216.0)) # ~6 PM
                width = float(params.get("width", 24.0)) # peak std dev
                amp1 = float(params.get("amp1", 1500.0))
                amp2 = float(params.get("amp2", 2000.0))
                
                p1 = amp1 * np.exp(-((step - peak1_center) ** 2) / (2 * (width ** 2)))
                p2 = amp2 * np.exp(-((step - peak2_center) ** 2) / (2 * (width ** 2)))
                users += bias + p1 + p2
                
            elif p_type == "spike":
                start = int(params.get("start", 120))
                duration = int(params.get("duration", 20))
                amplitude = float(params.get("amplitude", 3000.0))
                if start <= step < (start + duration):
                    # Simulates gaussian rising/falling edge or raw step
                    users += amplitude
                    
            elif p_type == "drop":
                start = int(params.get("start", 150))
                duration = int(params.get("duration", 30))
                amplitude = float(params.get("amplitude", 800.0))
                if start <= step < (start + duration):
                    users -= min(users, amplitude)
                    
            elif p_type == "random":
                mean = float(params.get("mean", 0.0))
                std = float(params.get("std", 50.0))
                users += random.gauss(mean, std)
                
            elif p_type == "csv":
                file_path = params.get("file_path", "")
                col_name = params.get("column", "active_users")
                
                if file_path and os.path.exists(file_path):
                    if file_path not in self.csv_cache:
                        try:
                            self.csv_cache[file_path] = pd.read_csv(file_path)
                        except Exception:
                            pass
                    
                    df = self.csv_cache.get(file_path)
                    if df is not None and col_name in df.columns:
                        idx = step % len(df)
                        users += float(df.iloc[idx][col_name])
                        
            # Apply dynamic conversion ratios overrides
            requests_ratio = float(params.get("requests_per_user", requests_ratio))
            network_ratio = float(params.get("network_mbps_per_request", network_ratio))
            
        # Ensure active users count is non-negative
        users = max(0.0, users)
        requests = users * requests_ratio
        network = requests * network_ratio
        
        return {
            "users": int(np.round(users)),
            "requests": float(requests),
            "network": float(network)
        }
