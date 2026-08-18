CREATE TABLE IF NOT EXISTS telemetry_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cpu_usage FLOAT NOT NULL,
    memory_usage FLOAT NOT NULL,
    network_traffic FLOAT NOT NULL,
    active_users INT NOT NULL,
    current_servers INT NOT NULL,
    recommended_servers INT NOT NULL,
    action VARCHAR(50) NOT NULL,
    sla_status VARCHAR(50) NOT NULL,
    anomaly_severity VARCHAR(50) NOT NULL,
    optimization_reason TEXT
);
