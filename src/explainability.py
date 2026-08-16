import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent GUI window blocking
import matplotlib.pyplot as plt
import shap

CLEANED_DATA_PATH = "data/cleaned_workload.csv"
FEATURES_LIST_PATH = "artifacts/features_list.pkl"
MODEL_PATH = "artifacts/cloud_resource_optimization_model.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

def generate_global_shap_plots(model, X_test_scaled, feature_names):
    """
    Computes global SHAP values and saves summary and bar plots.
    """
    print("Computing global SHAP values for visualization...")
    plots_dir = "data/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Take a sample of 200 observations for fast generation
    sample_size = min(200, len(X_test_scaled))
    X_sample = X_test_scaled[:sample_size]
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_sample)
    
    # For RandomForestRegressor, shap_values has shape (sample_size, features)
    
    # 1. SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.title("SHAP Global Feature Importance Summary", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary_plot.png"), dpi=150)
    plt.close()
    print("-> Saved shap_summary_plot.png")
    
    # 2. SHAP Bar Plot (Summary bar plot)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar Plot)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_bar_plot.png"), dpi=150)
    plt.close()
    print("-> Saved shap_bar_plot.png")

def explain_prediction_shap(explainer, X_scaled_record, feature_names: list, recommended_servers: int) -> dict:
    """
    Computes local feature contributions for a single prediction and generates
    a clean human-readable explanation sentence.
    """
    # Calculate SHAP values for the single record
    # shap_values shape will be (1, features)
    shap_vals = explainer.shap_values(X_scaled_record)
    
    if len(shap_vals.shape) == 2:
        shap_vector = shap_vals[0]
    else:
        shap_vector = shap_vals
        
    # Create feature contributions dictionary
    raw_contributions = {feat: float(val) for feat, val in zip(feature_names, shap_vector)}
    
    # Group contributions into core cloud metrics categories
    categories = {
        "CPU utilization": 0.0,
        "Memory utilization": 0.0,
        "Network traffic": 0.0,
        "Active users": 0.0,
        "Request workload rate": 0.0,
        "Response latency": 0.0,
        "Current active servers": 0.0,
        "Engineered temporal features": 0.0
    }
    
    for feat, val in raw_contributions.items():
        feat_lower = feat.lower()
        if "cpu" in feat_lower:
            categories["CPU utilization"] += val
        elif "memory" in feat_lower:
            categories["Memory utilization"] += val
        elif "traffic" in feat_lower or "network" in feat_lower or "net_" in feat_lower:
            categories["Network traffic"] += val
        elif "user" in feat_lower:
            categories["Active users"] += val
        elif "req" in feat_lower:
            categories["Request workload rate"] += val
        elif "resp" in feat_lower or "latency" in feat_lower:
            categories["Response latency"] += val
        elif "servers" in feat_lower:
            categories["Current active servers"] += val
        else:
            categories["Engineered temporal features"] += val
            
    # Sort categories by absolute contribution to find the top drivers
    sorted_categories = sorted(categories.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Identify positive and negative contributors
    positive_drivers = [cat for cat, val in sorted_categories if val > 0.01]
    negative_drivers = [cat for cat, val in sorted_categories if val < -0.01]
    
    # Map key categories to concise terms for human reading
    concise_names = {
        "CPU utilization": "CPU utilization",
        "Memory utilization": "memory usage",
        "Network traffic": "network traffic throughput",
        "Active users": "active-user demand",
        "Request workload rate": "request rate intensity",
        "Response latency": "application response time",
        "Current active servers": "current active capacity",
        "Engineered temporal features": "seasonal workload patterns"
    }
    
    # 6. Generate Human-Readable Sentence
    if positive_drivers:
        drivers_friendly = [concise_names[cat] for cat in positive_drivers[:2]]
        if len(drivers_friendly) == 2:
            reason = f"The system recommends {recommended_servers} servers primarily because {drivers_friendly[0]} and {drivers_friendly[1]} are increasing."
        else:
            reason = f"The system recommends {recommended_servers} servers primarily because {drivers_friendly[0]} is high."
    else:
        # Scale down or stable capacity scenario
        reason = f"The system recommends {recommended_servers} servers because resource demand (like active users and CPU usage) is stable/decreasing."
        
    return {
        "shap_explanation": reason,
        "category_contributions": {cat: round(val, 5) for cat, val in categories.items()},
        "top_feature_contributions": dict(sorted(raw_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
    }

if __name__ == "__main__":
    # Generate global plots as standalone script execution
    if os.path.exists(MODEL_PATH) and os.path.exists(CLEANED_DATA_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURES_LIST_PATH)
        
        df = pd.read_csv(CLEANED_DATA_PATH)
        X_raw = df[features]
        X_scaled = scaler.transform(X_raw)
        
        generate_global_shap_plots(model, X_scaled, features)
    else:
        print("Required training assets not found. Run pipeline and training scripts first.")
