import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from src.validation import validate_dataset, DatasetValidationError

# Paths for storing pipeline output and artifacts
CLEANED_DATA_PATH = "data/cleaned_workload.csv"
SCALER_PATH = "artifacts/scaler.pkl"

# Features used for ML modeling (all numeric input telemetry)
FEATURES = [
    "cpu_usage", "memory_usage", "network_in", "network_out", 
    "network_traffic", "disk_read", "disk_write", "active_users", 
    "request_rate", "response_time", "error_rate", "current_servers", 
    "server_cost"
]
TARGET = "required_servers"

def convert_and_sort_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes the timestamp column to datetime and sorts chronologically."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    return df

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values.
    For time-series cloud metrics, forward-fill (ffill) followed by 
    backward-fill (bfill) is standard to maintain sequence continuity.
    """
    df = df.copy()
    # Apply ffill and bfill on numeric features
    df[FEATURES] = df[FEATURES].ffill().bfill()
    # Check if target also has NaNs and handle it
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].ffill().bfill()
    return df

def detect_and_handle_outliers(df: pd.DataFrame, iqr_multiplier=3.0) -> pd.DataFrame:
    """
    Detects outliers in features using the IQR (Interquartile Range) method.
    Clips extreme outliers to the upper/lower boundary thresholds rather than 
    deleting data, preserving the integrity of the timeline.
    """
    df = df.copy()
    for col in FEATURES:
        # Define ranges according to physical/business rules first
        if col in ["cpu_usage", "memory_usage", "error_rate"]:
            df[col] = df[col].clip(0.0, 100.0)
        elif col in ["network_in", "network_out", "network_traffic", "disk_read", "disk_write", "active_users", "request_rate", "response_time", "server_cost"]:
            df[col] = df[col].clip(lower=0.0)
            
        # Apply statistical IQR thresholding for remaining extreme outliers
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)
        
        # Clip to IQR boundaries
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
    return df

def run_preprocessing_pipeline(raw_csv_path: str, output_csv_path: str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """
    Inbound batch preprocessing pipeline. Reads raw metrics, executes duplicate,
    missing value, validation, and outlier steps, and prepares data for scaling.
    """
    print(f"\nStarting ingestion pipeline for: {raw_csv_path}")
    
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_csv_path}")
        
    df = pd.read_csv(raw_csv_path)
    
    # 1. Deduplication
    initial_len = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dedup_len = len(df)
    if initial_len != dedup_len:
        print(f"-> Removed {initial_len - dedup_len} duplicate records.")
        
    # 2. Convert and sort timestamps
    df = convert_and_sort_timestamps(df)
    
    # 3. Impute missing values
    df = clean_missing_values(df)
    
    # 4. Range and Outlier mitigation
    df = detect_and_handle_outliers(df)
    
    # 5. Schema Validation (Run validation checks on preprocessed dataset)
    # The validation ensures that after preprocessing, data is fully ready for modeling.
    validate_dataset(df, raise_exception=True)
    
    # 6. Fit and Save Scaler
    print("-> Standardizing/Scaling numeric features...")
    scaler = StandardScaler()
    scaler.fit(df[FEATURES])
    
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"-> Saved fitted StandardScaler object to {SCALER_PATH}")
    
    # Save the cleaned dataset to disk
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"-> Cleaned dataset saved to {output_csv_path}")
    
    return df

def preprocess_single_record(record: dict, scaler: StandardScaler = None) -> np.ndarray:
    """
    Preprocesses and scales a single live telemetry record for real-time inference.
    """
    # 1. Input Validation using business rules
    errors = []
    for col in FEATURES:
        if col not in record:
            errors.append(f"Missing required metric: '{col}'")
            continue
        try:
            val = float(record[col])
            # Basic validation checks
            if col in ["cpu_usage", "memory_usage"] and (val < 0 or val > 100):
                errors.append(f"'{col}' must be between 0 and 100. Received: {val}")
            elif val < 0:
                errors.append(f"'{col}' must be non-negative. Received: {val}")
        except (ValueError, TypeError):
            errors.append(f"'{col}' must be numeric. Received: {record[col]}")
            
    if errors:
        raise DatasetValidationError("Real-time metric validation failed:\n- " + "\n- ".join(errors))
        
    # 2. Structure as DataFrame to match fit shapes
    record_df = pd.DataFrame([{col: float(record[col]) for col in FEATURES}])
    
    # 3. Clip any bounds
    for col in FEATURES:
        if col in ["cpu_usage", "memory_usage", "error_rate"]:
            record_df[col] = record_df[col].clip(0.0, 100.0)
        else:
            record_df[col] = record_df[col].clip(lower=0.0)
            
    # 4. Scale inputs
    if scaler is None:
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            raise FileNotFoundError(f"Fitted scaler not found at: {SCALER_PATH}. Run batch pipeline first.")
            
    scaled_features = scaler.transform(record_df[FEATURES])
    return scaled_features

if __name__ == "__main__":
    # Test batch pipeline
    # (Ensure to generate raw data first)
    from src.generator import generate_synthetic_workload
    generate_synthetic_workload()
    run_preprocessing_pipeline("data/synthetic_workload.csv")
