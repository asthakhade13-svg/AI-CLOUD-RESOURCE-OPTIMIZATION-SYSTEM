import pandas as pd
import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None

CLEANED_DATA_PATH = "data/cleaned_workload.csv"
SCALER_PATH = "artifacts/scaler.pkl"
FEATURES_LIST_PATH = "artifacts/features_list.pkl"
MODEL_PATH = "artifacts/cloud_resource_optimization_model.pkl"
TARGET = "required_servers"

def print_feature_importance_analysis(model, feature_names):
    """
    Extracts and prints the top 15 features by importance for the selected best model,
    if the model supports feature importances.
    """
    print("\n======================= FEATURE IMPORTANCE ANALYSIS =======================")
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15)
        
        print(f"{'Rank':<4} | {'Feature Name':<30} | {'MDI Importance':<14}")
        print("-" * 55)
        for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{rank:<4} | {row['Feature']:<30} | {row['Importance']:<14.4f}")
    elif hasattr(model, "coef_"):
        # For linear model, use coefficients as proxy for importance
        importances = np.abs(model.coef_)
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(15)
        
        print(f"{'Rank':<4} | {'Feature Name':<30} | {'Absolute Coefficient':<20}")
        print("-" * 60)
        for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{rank:<4} | {row['Feature']:<30} | {row['Importance']:<20.4f}")
    else:
        print("Selected model does not directly support feature importance or coefficient extraction.")
        return

    print("\n* Key Insights on Feature Utility for Predictive Scaling:")
    print("1. Lag Features (e.g. active_users_lag_1, cpu_usage_lag_5):")
    print("   -> Captures autocorrelation. A high CPU usage in the immediate past (t-1) strongly indicates high resource demands at time t.")
    print("2. Rolling Features (e.g. cpu_usage_moving_average_30, active_users_moving_average_5):")
    print("   -> Smooths out transient workload spikes. Moving averages allow the model to understand the longer-term trend rather than micro-bursts.")
    print("3. Trend Features (e.g. cpu_usage_growth_rate):")
    print("   -> Represents acceleration/velocity of workloads. A positive growth rate means server scaling must be proactive before CPU hits limits.")
    print("4. Cyclical Time Features (e.g. sin_hour, cos_day_of_week):")
    print("   -> Allows the model to predict resource scale based on daily/weekly cycles (e.g. lunch hour peaks, night drops).")
    print("===========================================================================")

def generate_capacity_plots(best_model, X_test, y_test, y_preds, best_name):
    """Generates evaluation plots for the selected best capacity model."""
    plots_dir = "data/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Actual vs Predicted required servers (slice of 100 observations)
    plt.figure(figsize=(12, 5))
    plt.step(np.arange(100), y_test[:100], where="post", color="black", linewidth=2.5, label="Actual Servers")
    plt.step(np.arange(100), y_preds[:100], where="post", color="#1f77b4", linestyle="--", alpha=0.9, label=f"Predicted ({best_name})")
    plt.title(f"Capacity Prediction: Actual vs. Predicted Required Servers ({best_name})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Sample Timeline (5-Min Steps)", fontsize=11)
    plt.ylabel("Server Quantity", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "capacity_actual_vs_predicted.png"), dpi=150)
    plt.close()
    print("-> Saved capacity_actual_vs_predicted.png")
    
    # 2. Residual error analysis
    residuals = y_test - y_preds
    plt.figure(figsize=(10, 5))
    plt.hist(residuals, bins=30, color="#1f77b4", alpha=0.7, edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    plt.title(f"Capacity Predictor Error Residual Distribution ({best_name})", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Prediction Error (Actual - Predicted Server Count)", fontsize=11)
    plt.ylabel("Frequency Count", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "capacity_prediction_residuals.png"), dpi=150)
    plt.close()
    print("-> Saved capacity_prediction_residuals.png")

def train_and_compare_models():
    """
    Loads preprocessed dataset, splits chronologically, scales features, trains 6 candidate algorithms,
    ranks them on validation MAE, and automatically saves the best performing model to disk.
    """
    print("\nLoading dataset and features list for training...")
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(f"Cleaned dataset not found at: {CLEANED_DATA_PATH}. Run pipeline first.")
    if not os.path.exists(FEATURES_LIST_PATH):
        raise FileNotFoundError(f"Features list not found at: {FEATURES_LIST_PATH}. Run pipeline first.")
        
    df = pd.read_csv(CLEANED_DATA_PATH)
    features = joblib.load(FEATURES_LIST_PATH)
    
    X = df[features]
    y = df[TARGET]
    
    # Chronological Split (Train: 70%, Val: 15%, Test: 15%)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train_raw = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val_raw = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test_raw = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    print(f"Time-aware split sizes -> Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")
    
    # Scale Features using standardizer
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Fitted scaler not found at: {SCALER_PATH}")
        
    scaler = joblib.load(SCALER_PATH)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Define candidate models with fixed seeds for reproducibility
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42, n_jobs=-1)
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(n_estimators=100, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1)
    
    results_list = []
    trained_models = {}
    
    print("\nTraining and evaluating candidate capacity models...")
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start_time
        
        # Predict on validation and test sets
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        
        # Evaluate
        val_mae = mean_absolute_error(y_val, val_preds)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        val_r2 = r2_score(y_val, val_preds)
        
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)
        
        trained_models[name] = {
            "model_obj": model,
            "test_preds": test_preds
        }
        
        results_list.append({
            "Model": name,
            "Val MAE": val_mae,
            "Val RMSE": val_rmse,
            "Val R2": val_r2,
            "Test MAE": test_mae,
            "Test RMSE": test_rmse,
            "Test R2": test_r2,
            "Train Time (s)": duration
        })
        
    # Construct comparison dataframe and sort by validation MAE (lower is better)
    comparison_df = pd.DataFrame(results_list).sort_values(by="Val MAE", ascending=True).reset_index(drop=True)
    
    print("\n======================= AUTOMATED MODEL COMPARISON PIPELINE =======================")
    print(comparison_df.to_string(index=False))
    print("===================================================================================")
    
    # 7. Select and save the best model
    best_row = comparison_df.iloc[0]
    best_name = best_row["Model"]
    best_model_obj = trained_models[best_name]["model_obj"]
    best_test_preds = trained_models[best_name]["test_preds"]
    
    print(f"\nWinner selected automatically: {best_name} (Val MAE={best_row['Val MAE']:.4f})")
    
    # Save best model to disk
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(best_model_obj, MODEL_PATH)
    print(f"Saved best performing capacity model to: {MODEL_PATH}")
    
    # 8. Generate evaluation plots
    generate_capacity_plots(best_model_obj, X_test, y_test.values, best_test_preds, best_name)
    
    # 9. Output feature importances
    print_feature_importance_analysis(best_model_obj, features)
    
    return best_model_obj

if __name__ == "__main__":
    train_and_compare_models()
