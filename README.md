# AI-Powered Predictive Cloud Resource Optimization & Auto-Scaling System

A B.Tech level project implementing a production-quality machine learning data pipeline, metric validation engine, visualization suites, and REST API backend. It predicts the optimal cloud server allocation requirements and triggers auto-scaling alerts based on system telemetry.

---

## 📂 Project Directory Structure

```text
ai-cloud-resource/
│
├── data/                    # Raw & preprocessed datasets, and plots
│   ├── synthetic_workload.csv   # Raw generated synthetic workload (with anomalies)
│   ├── cleaned_workload.csv     # Preprocessed, validated, and anomaly-cleaned dataset
│   └── plots/               # Automatically generated dashboard visualizations
│
├── src/                     # Reusable source code modules
│   ├── __init__.py          # Python package initializer
│   ├── generator.py         # Synthetic workload data generator (labeled clearly)
│   ├── validation.py        # Static schema and data boundary validation
│   ├── pipeline.py          # Duplication removal, imputation, outlier handling, & scaling
│   ├── visualization.py     # Generates 6 workload profile plots over time
│   └── train_models.py      # Random Forest vs XGBoost training, evaluation, & selection
│
├── artifacts/               # Serialized binary machine learning assets
│   ├── cloud_resource_optimization_model.pkl   # The trained XGBoost model (best performer)
│   └── scaler.pkl           # StandardScaler fitted on clean training features
│
├── main.py                  # FastAPI Backend serving predictions
├── test_main.py             # Unit tests checking API endpoints
├── test_pipeline.py         # Unit tests checking validations & preprocessing pipeline
├── requirements.txt         # Project package dependencies
├── Dockerfile               # Containerization configuration
└── README.md                # Project documentation & instructions
```

---

## 🛠️ Tech Stack & Requirements
* **Framework:** FastAPI, Uvicorn (REST API Backend)
* **ML Engines:** Scikit-Learn, XGBoost, Pandas, NumPy, Joblib
* **Visualizations:** Matplotlib
* **Unit Testing:** Pytest, HTTPX

---

## 🚀 Step-by-Step Execution Guide

### 1. Installation
Install all backend and data science dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate the Workload Dataset
Generate the synthetic raw workload dataset. Note that this dataset contains duplicates, missing values, and outlier anomalies to simulate real-world logging discrepancies:
```bash
python -m src.generator
```
* Generates `data/synthetic_workload.csv`.

### 3. Run Preprocessing Pipeline
Clean missing values (forward/backward fill), drop duplicate records, clip outliers via IQR thresholds, validate data schema limits, and fit/serialize the feature standardizer (`scaler.pkl`):
```bash
python -m src.pipeline
```
* Outputs `data/cleaned_workload.csv` and saves standardizer to `artifacts/scaler.pkl`.

### 4. Generate Visualizations Dashboard
Produce the 6 key telemetry plots comparing CPU, Memory, Active Users, Network throughput, Server Counts, and SLA response times over a 1-week timeline:
```bash
python -m src.visualization
```
* Saves plots inside `data/plots/` as PNG files.

### 5. Train & Select Best Model
Train both `RandomForestRegressor` and `XGBRegressor` on the cleaned metrics. Evaluates performance (MAE, RMSE, and $R^2$) and automatically serializes the best performing model:
```bash
python -m src.train_models
```
* Saves the model to `artifacts/cloud_resource_optimization_model.pkl`.

### 6. Run the Test Suite
Run automated unit tests to verify the pipeline, schema validation, API logic, and model inference:
```bash
python -m pytest
```

### 7. Launch the API Server
Start the FastAPI REST backend to serve real-time predictions:
```bash
uvicorn main:app --reload
```
* Interactive API Documentation is available at: `http://127.0.0.1:8000/docs`

---

## 📋 Sample CSV Schema (15-Column Layout)

The pipeline parses CSV logs conforming to this layout. Here is a sample representation of a single row in the dataset:

| Column Name | Sample Value | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | `2026-08-01 14:00:00` | String (`YYYY-MM-DD HH:MM:SS`) | Timestamp of the metrics recording. |
| `cpu_usage` | `68.45` | Float ($0.0 - 100.0$) | Percentage of system CPU usage. |
| `memory_usage` | `72.12` | Float ($0.0 - 100.0$) | Percentage of system RAM usage. |
| `network_in` | `112.50` | Float ($\ge 0$) | Network downloads/ingress rate in Mbps. |
| `network_out` | `320.15` | Float ($\ge 0$) | Network uploads/egress rate in Mbps. |
| `network_traffic`| `432.65` | Float ($\ge 0$) | Combined network throughput ($network\_in + network\_out$). |
| `disk_read` | `125.40` | Float ($\ge 0$) | Disk Read IOPS. |
| `disk_write` | `78.20` | Float ($\ge 0$) | Disk Write IOPS. |
| `active_users` | `285` | Integer ($\ge 0$) | Number of active user sessions. |
| `request_rate` | `712.50` | Float ($\ge 0$) | Server requests processed per second (RPS). |
| `response_time` | `195.40` | Float ($\ge 0$) | Average HTTP response time in milliseconds. |
| `error_rate` | `0.0234` | Float ($0.0 - 100.0$) | Percentage of requests resulting in failures. |
| `current_servers`| `4` | Integer ($\ge 1$) | Current count of active web/worker VMs. |
| `server_cost` | `0.4886` | Float ($\ge 0$) | Operational hosting cost per hour (in USD). |
| `required_servers`| `5` | Integer ($\ge 1$) | **Target Label:** The optimal count of servers needed. |

---

## 🔌 Instructions for Replacing Synthetic Data with Real Cloud Metrics

To transition this project from synthetic data to live, real-world cloud resources, follow these steps:

### A. Metric Sources from Production Cloud Providers
You can extract the 15 schema metrics from standard monitoring agents:

1. **AWS EC2 & CloudWatch:**
   * `cpu_usage` $\rightarrow$ `AWS/EC2` $\rightarrow$ `CPUUtilization`
   * `network_in`/`network_out` $\rightarrow$ `AWS/EC2` $\rightarrow$ `NetworkIn` / `NetworkOut`
   * `disk_read`/`disk_write` $\rightarrow$ `AWS/EC2` $\rightarrow$ `DiskReadOps` / `DiskWriteOps`
   * `response_time`/`request_rate`/`error_rate` $\rightarrow$ `AWS/ApplicationELB` $\rightarrow$ `TargetResponseTime` / `RequestCount` / `HTTPCode_Target_5XX_Count`
   * `current_servers` $\rightarrow$ `AWS/AutoScaling` $\rightarrow$ `GroupMaxSize` (or query EC2 describe instance count)
   * `server_cost` $\rightarrow$ Fetch pricing via the AWS Price List API based on instance type (e.g. `t3.medium` cost/hr).

2. **Prometheus & Kubernetes (for containerized workloads):**
   * `cpu_usage` $\rightarrow$ `sum(rate(container_cpu_usage_seconds_total[5m])) by (pod) * 100`
   * `memory_usage` $\rightarrow$ `(sum(container_memory_working_set_bytes) by (pod) / sum(machine_memory_bytes)) * 100`
   * `network_in` $\rightarrow$ `sum(rate(container_network_receive_bytes_total[5m]))`
   * `active_users` $\rightarrow$ Query application session state database or Nginx log analytics.

### B. Ingesting Real-world Metrics into the Pipeline
1. **Export as CSV:**
   * Write a Python Cron script or use Prometheus web UI to query Prometheus HTTP API endpoints. Export the output metrics into a pandas DataFrame and save it as `cloud_historical_data.csv` following the column names defined in the schema.
   * Place the exported file in the `data/` directory and run the preprocessing pipeline (`python src/pipeline.py`) to handle missing server logs, drop duplicates, standardize the telemetry scales, and retrain the models.

2. **Real-time Metric Ingestion (API Hook):**
   * Setup your monitoring dashboard (e.g. Prometheus Webhook, AWS SNS, or cron script) to fire POST requests containing live infrastructure metrics to the `/predict` API endpoint every 5 or 10 minutes.
   * The FastAPI server will automatically parse, clean, and scale the real-time record on the fly before predicting the required auto-scaling action.
