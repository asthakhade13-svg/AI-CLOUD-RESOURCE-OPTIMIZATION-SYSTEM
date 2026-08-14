import pandas as pd
import numpy as np
import os

def generate_synthetic_workload(days=30, output_path="data/synthetic_workload.csv", seed=42):
    """
    Generates a highly realistic but explicitly SYNTHETIC cloud workload dataset 
    with a 15-column schema for cloud resource auto-scaling optimization modeling.
    """
    print("==================================================================")
    print("   WARNING: GENERATING SYNTHETIC BENCHMARKING WORKLOAD DATASET")
    print("   This data is synthetically modeled and does NOT represent")
    print("   live/real-world production measurements from a specific system.")
    print("==================================================================")
    
    np.random.seed(seed)
    
    # 1. Timestamps (5-Minute Intervals)
    periods = days * 24 * 12
    time_index = pd.date_range(start="2026-08-01 00:00:00", periods=periods, freq="5min")
    
    # Initialize metric containers
    timestamps = []
    active_users_list = []
    cpu_usage_list = []
    memory_usage_list = []
    net_in_list = []
    net_out_list = []
    disk_read_list = []
    disk_write_list = []
    req_rate_list = []
    resp_time_list = []
    err_rate_list = []
    req_servers_list = []
    
    for i, dt in enumerate(time_index):
        hour = dt.hour
        day_of_week = dt.dayofweek # 0=Monday, 6=Sunday
        
        # --- Active Users (Diurnal Pattern with Weekend drop) ---
        # Dual peaks: lunch peak (13:00 - 14:00) and evening peak (20:00 - 21:00)
        diurnal_factor = (
            0.4 * np.sin(2 * np.pi * (hour - 6) / 24) + 
            0.6 * np.sin(4 * np.pi * (hour - 10) / 24)
        )
        # Scale range: base load of 80 users up to ~450 users
        base_users = 250 + 200 * diurnal_factor
        
        # Weekend load is typically lower (e.g. B2B software reduction by 40%)
        if day_of_week >= 5:
            base_users *= 0.6
            
        users = int(max(10, base_users + np.random.normal(0, 15)))
        active_users_list.append(users)
        
        # --- Request Rate (correlated with active users) ---
        # Average of 2.5 requests per active user, with noise
        req_rate = max(1.0, float(users * 2.5 + np.random.normal(0, 10)))
        req_rate_list.append(round(req_rate, 2))
        
        # --- CPU Usage (correlated with request rate, memory, and spikes) ---
        # Base CPU of 15% + request rate load factor
        cpu = 15.0 + (req_rate / 1500) * 60.0 + np.random.normal(0, 5)
        # Random heavy background tasks or compute-intensive jobs (5% chance)
        if np.random.rand() < 0.05:
            cpu += np.random.uniform(15, 30)
        cpu = min(100.0, max(0.0, cpu))
        cpu_usage_list.append(round(cpu, 2))
        
        # --- Memory Usage (smoother than CPU, correlated with active users) ---
        mem = 20.0 + (users / 500) * 55.0 + np.random.normal(0, 3)
        mem = min(100.0, max(0.0, mem))
        memory_usage_list.append(round(mem, 2))
        
        # --- Network Telemetry ---
        # Network In (downloads/ingress) - correlated with request rate
        net_in = max(1.0, float(req_rate * 0.15 + np.random.normal(0, 5)))
        net_in_list.append(round(net_in, 2))
        
        # Network Out (uploads/egress) - usually higher for media/data distribution
        net_out = max(1.0, float(req_rate * 0.45 + np.random.normal(0, 15)))
        net_out_list.append(round(net_out, 2))
        
        # --- Disk Read / Write (IOPS) ---
        disk_read = max(0.0, float(req_rate * 0.8 + np.random.normal(0, 12)))
        disk_read_list.append(round(disk_read, 2))
        
        disk_write = max(0.0, float(req_rate * 0.5 + np.random.normal(0, 8)))
        # Occasional database backup dump (2% chance)
        if np.random.rand() < 0.02:
            disk_write += np.random.uniform(100, 300)
        disk_write_list.append(round(disk_write, 2))
        
        # --- Response Time (ms) ---
        # Increases when CPU/Memory is saturated or active users are high (queue delay)
        resource_saturation = (cpu / 100.0) ** 2 + (mem / 100.0) ** 2
        base_resp_time = 150.0 + (users / 500) * 100.0 + resource_saturation * 400.0
        resp_time = max(10.0, base_resp_time + np.random.normal(0, 20))
        resp_time_list.append(round(resp_time, 2))
        
        # --- Error Rate (%) ---
        # Spikes when response times are high (timeouts) or when CPU is pegged
        err_prob = 0.1
        if cpu > 85.0:
            err_prob += (cpu - 85.0) * 0.3 # fast spike in errors
        if resp_time > 500.0:
            err_prob += (resp_time - 500.0) * 0.02
        err_rate = max(0.0, min(100.0, err_prob + np.random.normal(0, 0.5)))
        err_rate_list.append(round(err_rate, 4))
        
        # --- Required Servers (optimal scale count) ---
        # Rule-based calculation of how many servers are needed
        # Factor 1: Active users (1 server handles ~85 users)
        srv_users = np.ceil(users / 85.0)
        # Factor 2: CPU utilization (target 65% CPU limit)
        srv_cpu = np.ceil(cpu / 65.0)
        # Factor 3: Memory utilization (target 70% Memory limit)
        srv_mem = np.ceil(mem / 70.0)
        
        req_srv = int(max(1, srv_users, srv_cpu, srv_mem))
        req_servers_list.append(req_srv)
        
        timestamps.append(dt.strftime("%Y-%m-%d %H:%M:%S"))

    # Convert required servers to an array to model scaling latency and current servers
    req_servers_arr = np.array(req_servers_list)
    current_servers_list = []
    
    # Current servers lags behind required servers due to provisioning latency (autoscaling delay)
    current_servers_list.append(req_servers_arr[0])
    for i in range(1, len(req_servers_arr)):
        prev_srv = current_servers_list[-1]
        target_srv = req_servers_arr[i]
        
        if prev_srv == target_srv:
            current_servers_list.append(prev_srv)
        else:
            # 70% chance to adjust server count towards required count per hour
            if np.random.rand() < 0.70:
                adjustment = 1 if target_srv > prev_srv else -1
                current_servers_list.append(prev_srv + adjustment)
            else:
                current_servers_list.append(prev_srv)
                
    current_servers_arr = np.array(current_servers_list)
    
    # --- Server Cost ---
    # Cost formula: $0.12/hour per server + minor scaling costs for traffic
    # e.g., Base rate + variable pricing for disk and network usage
    base_cost_per_server = 0.12 # USD per hour
    server_cost_list = []
    for i in range(periods):
        srv_cnt = current_servers_arr[i]
        net_traffic = net_in_list[i] + net_out_list[i]
        # Active servers cost money, plus minor factor for processing high network traffic
        cost = srv_cnt * base_cost_per_server + (net_traffic / 1000) * 0.02
        server_cost_list.append(round(cost, 4))
        
    # Combine into DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_usage": cpu_usage_list,
        "memory_usage": memory_usage_list,
        "network_in": net_in_list,
        "network_out": net_out_list,
        "network_traffic": [round(in_ + out_, 2) for in_, out_ in zip(net_in_list, net_out_list)],
        "disk_read": disk_read_list,
        "disk_write": disk_write_list,
        "active_users": active_users_list,
        "request_rate": req_rate_list,
        "response_time": resp_time_list,
        "error_rate": err_rate_list,
        "current_servers": current_servers_list,
        "server_cost": server_cost_list,
        "required_servers": req_servers_list
    })
    
    # Introduce explicit anomalies/missing values for cleaning pipeline validation
    # Let's add 5 rows with missing CPU/Memory values (NaN)
    nan_indices = np.random.choice(periods, size=5, replace=False)
    df.loc[nan_indices, "cpu_usage"] = np.nan
    df.loc[nan_indices[:3], "memory_usage"] = np.nan
    
    # Let's add 3 duplicate rows
    dup_indices = np.random.choice(periods, size=3, replace=False)
    dup_rows = df.loc[dup_indices]
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Let's inject a few extreme outliers (e.g. CPU = 999.0 or Network = -500.0)
    outlier_idx_1 = np.random.randint(0, len(df))
    outlier_idx_2 = np.random.randint(0, len(df))
    df.loc[outlier_idx_1, "cpu_usage"] = 888.8  # Outlier CPU
    df.loc[outlier_idx_2, "network_in"] = -100.0  # Outlier negative traffic
    
    # Save the synthetic dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset successfully generated and saved to {output_path}")
    print(f"Total Rows (with duplicates and NaNs): {len(df)}")
    return df

if __name__ == "__main__":
    generate_synthetic_workload()
