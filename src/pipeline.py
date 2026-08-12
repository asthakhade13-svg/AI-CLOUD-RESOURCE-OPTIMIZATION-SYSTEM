import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from src.validation import validate_dataset, DatasetValidationError
from src.features import apply_feature_engineering_pipeline

# Paths for storing pipeline output and artifacts
CLEANED_DATA_PATH = "data/cleaned_workload.csv"
SCALER_PATH = "artifacts/scaler.pkl"

# Base input columns
BASE_FEATURES = [
    "cpu_usage", "memory_usage", "network_in", "network_out", 
    "network_traffic", "disk_read", "disk_write", "active_users", 
    "request_rate", "response_time", "error_rate", "current_servers", 
    "server_cost"
]

# We will dynamically populate ALL_MODEL_FEATURES after running the pipeline, 
# but we can list the expected structure for verification.
TARGET = "required_servers"

def run_preprocessing_pipeline(raw_csv_path: str, output_csv_path: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """
    Ingestion & cleaning pipeline. Cleans data, generates advanced features, 
    standards features using StandardScaler, and dumps the scaler to disk.
    """
    print(f"\nStarting ingestion pipeline for: {raw_csv_path}")
    
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_csv_path}")
        
    df = pd.read_csv(raw_csv_path)
    
    # 1. Deduplication
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Timestamp sorting
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    
    # 3. Missing Value Imputation
    df[BASE_FEATURES] = df[BASE_FEATURES].ffill().bfill()
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].ffill().bfill()
        
    # 4. Outlier mitigation
    for col in BASE_FEATURES:
        if col in ["cpu_usage", "memory_usage", "error_rate"]:
            df[col] = df[col].clip(0.0, 100.0)
        else:
            df[col] = df[col].clip(lower=0.0)
            
    # 5. Schema Validation on clean base data
    validate_dataset(df, raise_exception=True)
    
    # 6. Apply Feature Engineering
    print("-> Applying advanced feature engineering...")
    df_engineered = apply_feature_engineering_pipeline(df, is_training=True)
    
    # Extract the exact feature list (excluding target, timestamp, and metadata like intermediate time variables)
    exclude_cols = [TARGET, "timestamp"]
    model_features = [col for col in df_engineered.columns if col not in exclude_cols]
    
    # Save the feature list to disk so the API knows the exact column order
    joblib.dump(model_features, "artifacts/features_list.pkl")
    print(f"-> Saved model features list (Total {len(model_features)} features) to artifacts/features_list.pkl")
    
    # 7. Fit and Save Scaler
    print("-> Standardizing/Scaling all engineered features...")
    scaler = StandardScaler()
    scaler.fit(df_engineered[model_features])
    
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"-> Saved fitted StandardScaler object to {SCALER_PATH}")
    
    # Save the cleaned dataset to disk
    df_engineered.to_csv(output_csv_path, index=False)
    print(f"-> Cleaned & engineered dataset saved to {output_csv_path}. Shape: {df_engineered.shape}")
    
    return df_engineered

def preprocess_single_record(history_df: pd.DataFrame, scaler: StandardScaler = None) -> np.ndarray:
    """
    Preprocesses a single live metric record by applying feature engineering 
    across the historical context buffer and returning the scaled vector for the latest record.
    """
    # 1. Clean history base values (impute, clip)
    df = history_df.copy()
    df[BASE_FEATURES] = df[BASE_FEATURES].ffill().bfill()
    for col in BASE_FEATURES:
        if col in ["cpu_usage", "memory_usage", "error_rate"]:
            df[col] = df[col].clip(0.0, 100.0)
        else:
            df[col] = df[col].clip(lower=0.0)
            
    # 2. Run feature engineering (is_training=False preserves all records, filling NaNs)
    df_engineered = apply_feature_engineering_pipeline(df, is_training=False)
    
    # 3. Load feature names order
    features_list_path = "artifacts/features_list.pkl"
    if os.path.exists(features_list_path):
        model_features = joblib.load(features_list_path)
    else:
        raise FileNotFoundError(f"Feature list file not found at: {features_list_path}")
        
    # 4. Extract the latest record (which now contains complete lag/rolling values computed from history)
    latest_record = df_engineered.iloc[[-1]][model_features]
    
    # 5. Scale using standardizer
    if scaler is None:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            raise FileNotFoundError(f"Scaler file not found at: {SCALER_PATH}")
            
    scaled_vector = scaler.transform(latest_record)
    return scaled_vector

if __name__ == "__main__":
    from src.generator import generate_synthetic_workload
    generate_synthetic_workload()
    run_preprocessing_pipeline("data/synthetic_workload.csv")
