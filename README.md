# AI Cloud Resource Optimization Backend

A lightweight, high-performance REST API built using **FastAPI** to serve predictions from the Random Forest Regressor model. The API recommends real-time server scaling actions based on current resource utilization and user demand metrics.

---

## 🛠️ Tech Stack & Requirements
* **Framework:** FastAPI
* **Server:** Uvicorn
* **ML Libraries:** Scikit-Learn, Pandas, NumPy, Joblib

---

## 🚀 How to Run the Server

1. **Install Dependencies:**
   Ensure you have all the required libraries installed:
   ```bash
   pip install fastapi uvicorn scikit-learn pandas numpy joblib
   ```

2. **Start the API Server:**
   Run the following command in the project root directory:
   ```bash
   uvicorn main:app --reload
   ```
   * The server will run at `http://127.0.0.1:8000`
   * `--reload` enables automatic restart on code changes (ideal for local development).

---

## 📖 API Endpoints & Usage

Once the server is running, you can access the automatic interactive API documentation at:
* **Swagger UI:** `http://127.0.0.1:8000/docs`
* **Redoc:** `http://127.0.0.1:8000/redoc`

### 1. Root / Welcome
* **Method:** `GET`
* **Path:** `/`
* **Response:**
  ```json
  {
    "message": "Welcome to the AI Cloud Resource Optimization API",
    "docs_url": "/docs",
    "health_check_url": "/health",
    "model_loaded": true
  }
  ```

### 2. API Health Check
* **Method:** `GET`
* **Path:** `/health`
* **Response:**
  ```json
  {
    "status": "healthy",
    "model_file": "cloud_resource_optimization_model.pkl",
    "model_type": "RandomForestRegressor"
  }
  ```

### 3. Predict Scaling Requirement
* **Method:** `POST`
* **Path:** `/predict`
* **Request Body:**
  ```json
  {
    "cpu_usage": 85.0,
    "memory_usage": 78.5,
    "network_traffic": 450.0,
    "active_users": 310,
    "current_servers": 3
  }
  ```
* **Response Body:**
  ```json
  {
    "predicted_required_servers": 6,
    "raw_prediction": 5.82,
    "current_servers": 3,
    "scaling_action": "SCALE UP",
    "reasoning": "Current load requires 6 servers. Scale up by adding 3 server(s)."
  }
  ```

### 4. Feature Importance
* **Method:** `GET`
* **Path:** `/features`
* **Response:**
  ```json
  {
    "features": [
      "cpu_usage",
      "memory_usage",
      "network_traffic",
      "active_users",
      "current_servers"
    ],
    "importances": {
      "cpu_usage": 0.4851,
      "active_users": 0.3122,
      "memory_usage": 0.1205,
      "network_traffic": 0.0532,
      "current_servers": 0.0290
    }
  }
  ```
