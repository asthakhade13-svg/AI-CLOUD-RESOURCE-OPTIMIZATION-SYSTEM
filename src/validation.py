import pandas as pd
import numpy as np

class DatasetValidationError(ValueError):
    """Exception raised when a dataset violates schema or business logic checks."""
    pass

REQUIRED_SCHEMA = {
    "timestamp": "object_or_datetime",
    "cpu_usage": "numeric",
    "memory_usage": "numeric",
    "network_in": "numeric",
    "network_out": "numeric",
    "network_traffic": "numeric",
    "disk_read": "numeric",
    "disk_write": "numeric",
    "active_users": "numeric",
    "request_rate": "numeric",
    "response_time": "numeric",
    "error_rate": "numeric",
    "current_servers": "numeric",
    "server_cost": "numeric",
    "required_servers": "numeric"
}

def validate_schema(df: pd.DataFrame) -> list:
    """
    Validates that the input DataFrame matches the expected column schema and types.
    Returns a list of error messages (empty if the schema is fully valid).
    """
    errors = []
    
    # 1. Column existence checks
    for col, expected_type in REQUIRED_SCHEMA.items():
        if col not in df.columns:
            errors.append(f"Missing required column: '{col}'")
            continue
            
        # 2. Basic Datatype checks
        col_type = df[col].dtype
        if expected_type == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column '{col}' must be numeric, but has type '{col_type}'")
        elif expected_type == "object_or_datetime":
            if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col])):
                errors.append(f"Column '{col}' must be string or datetime, but has type '{col_type}'")
                
    return errors

def validate_data_ranges(df: pd.DataFrame) -> list:
    """
    Validates that columns contain values within realistic business-logic boundaries.
    Ignores rows with null values (which are caught/cleaned by the pipeline later).
    """
    errors = []
    
    # helper for range checks
    def check_bounds(col_name, min_val=None, max_val=None):
        if col_name not in df.columns:
            return
        
        non_null_data = df[col_name].dropna()
        if min_val is not None:
            out_of_bounds_min = non_null_data[non_null_data < min_val]
            if len(out_of_bounds_min) > 0:
                errors.append(
                    f"Column '{col_name}' has {len(out_of_bounds_min)} value(s) below lower bound ({min_val}). "
                    f"Min found: {out_of_bounds_min.min()}"
                )
                
        if max_val is not None:
            out_of_bounds_max = non_null_data[non_null_data > max_val]
            if len(out_of_bounds_max) > 0:
                errors.append(
                    f"Column '{col_name}' has {len(out_of_bounds_max)} value(s) above upper bound ({max_val}). "
                    f"Max found: {out_of_bounds_max.max()}"
                )

    # Apply boundary checks
    check_bounds("cpu_usage", min_val=0.0, max_val=100.0)
    check_bounds("memory_usage", min_val=0.0, max_val=100.0)
    check_bounds("network_in", min_val=0.0)
    check_bounds("network_out", min_val=0.0)
    check_bounds("network_traffic", min_val=0.0)
    check_bounds("disk_read", min_val=0.0)
    check_bounds("disk_write", min_val=0.0)
    check_bounds("active_users", min_val=0.0)
    check_bounds("request_rate", min_val=0.0)
    check_bounds("response_time", min_val=0.0)
    check_bounds("error_rate", min_val=0.0, max_val=100.0)
    check_bounds("current_servers", min_val=1.0)
    check_bounds("server_cost", min_val=0.0)
    check_bounds("required_servers", min_val=1.0)
    
    return errors

def validate_dataset(df: pd.DataFrame, raise_exception: bool = False) -> bool:
    """
    Runs the full validation suite (schema + ranges) on a DataFrame.
    """
    errors = []
    
    schema_errors = validate_schema(df)
    errors.extend(schema_errors)
    
    # Only check bounds if the basic schema is correct
    if not schema_errors:
        range_errors = validate_data_ranges(df)
        errors.extend(range_errors)
        
    if errors:
        error_msg = "Dataset Validation Failed with the following errors:\n- " + "\n- ".join(errors)
        if raise_exception:
            raise DatasetValidationError(error_msg)
        else:
            print(f"[VALIDATION WARNING] {error_msg}")
            return False
            
    print("[VALIDATION SUCCESS] Dataset structure and values are fully valid!")
    return True

if __name__ == "__main__":
    # Test on a dummy DataFrame
    test_df = pd.DataFrame({"cpu_usage": [120.0, 50.0], "timestamp": ["2026-08-01", "2026-08-02"]})
    validate_dataset(test_df)
