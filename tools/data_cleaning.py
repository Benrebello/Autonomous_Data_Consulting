# tools/data_cleaning.py
"""Data cleaning and preparation tools."""

import pandas as pd
import numpy as np
from typing import List


def clean_data(df: pd.DataFrame, column: str, strategy: str = 'mean') -> pd.DataFrame:
    """Fills null values in a column based on a strategy.
    
    Args:
        df: Input DataFrame
        column: Column name to clean
        strategy: Strategy for filling nulls ('mean', 'median', 'mode')
        
    Returns:
        DataFrame with cleaned column
    """
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    else:  # mode
        fill_value = df[column].mode()[0]
    # Avoid chained assignment with inplace=True to prevent FutureWarning in pandas 3.0+
    df[column] = df[column].fillna(fill_value)
    return df


def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Column labels to consider for identifying duplicates
        
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')


def fill_missing_with_median(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Fill missing values with median for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of column names to fill
        
    Returns:
        DataFrame with filled values
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            median_val = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(median_val)
    return df_copy


def validate_and_correct_data_types(df: pd.DataFrame):
    """Validate and correct data types automatically.

    Args:
        df: Input DataFrame

    Returns:
        Tuple (DataFrame with corrected types, report dict)

    The report dict maps each column name to one of:
        - 'numeric'   -> column converted to numeric
        - 'datetime'  -> column converted to datetime
        - 'unchanged' -> no conversion applied
    """
    df_copy = df.copy()
    report: dict[str, str] = {}

    for col in df_copy.columns:
        original_is_numeric = pd.api.types.is_numeric_dtype(df_copy[col])
        original_is_datetime = pd.api.types.is_datetime64_any_dtype(df_copy[col])

        # Attempt numeric conversion first
        converted = df_copy[col]
        try:
            converted_num = pd.to_numeric(converted, errors='coerce')
            # Only apply conversion if at least some values were successfully converted
            if converted_num.notna().any():
                df_copy[col] = converted_num
            else:
                df_copy[col] = converted
        except Exception:
            df_copy[col] = converted
        became_numeric = (not original_is_numeric) and pd.api.types.is_numeric_dtype(df_copy[col])

        # Attempt datetime conversion if name suggests time/date
        became_datetime = False
        if any(keyword in col.lower() for keyword in ['date', 'time', 'timestamp']):
            try:
                before_dt = pd.api.types.is_datetime64_any_dtype(df_copy[col])
                converted_dt = pd.to_datetime(df_copy[col], errors='coerce')
                # Only apply conversion if at least some values were successfully converted
                if converted_dt.notna().any():
                    df_copy[col] = converted_dt
                    after_dt = pd.api.types.is_datetime64_any_dtype(df_copy[col])
                    became_datetime = (not original_is_datetime) and after_dt and not before_dt
            except Exception:
                pass

        if became_numeric:
            report[col] = 'numeric'
        elif became_datetime:
            report[col] = 'datetime'
        else:
            report[col] = 'unchanged'

    return df_copy, report


def validate_and_clean_dataframe(df: pd.DataFrame, 
                                  remove_duplicates_flag: bool = True,
                                  fill_numeric_nulls: bool = True) -> dict:
    """Comprehensive data validation and cleaning.
    
    Args:
        df: Input DataFrame
        remove_duplicates_flag: Whether to remove duplicate rows
        fill_numeric_nulls: Whether to fill numeric null values with median
        
    Returns:
        Dictionary with 'dataframe' (cleaned DataFrame) and 'report' (dict with validation details)
    """
    df_clean = df.copy()
    corrections_applied = []
    warnings = []
    
    # Track initial state
    initial_rows = len(df_clean)
    initial_nulls = df_clean.isnull().sum().sum()
    
    # Remove duplicates
    if remove_duplicates_flag:
        duplicates_count = df_clean.duplicated().sum()
        if duplicates_count > 0:
            df_clean = df_clean.drop_duplicates()
            corrections_applied.append(f"Removed {duplicates_count} duplicate rows")
    
    # Fill numeric nulls
    if fill_numeric_nulls:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            null_count = df_clean[col].isnull().sum()
            if null_count > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
                corrections_applied.append(f"Filled {null_count} null values in '{col}' with median ({median_val:.2f})")
    
    # Check for remaining nulls in non-numeric columns
    remaining_nulls = df_clean.isnull().sum()
    for col, null_count in remaining_nulls.items():
        if null_count > 0:
            warnings.append(f"Column '{col}' still has {null_count} null values")
    
    # Build report
    report = {
        'corrections_applied': corrections_applied,
        'warnings': warnings,
        'initial_rows': initial_rows,
        'final_rows': len(df_clean),
        'initial_nulls': initial_nulls,
        'final_nulls': df_clean.isnull().sum().sum()
    }
    
    return {
        'dataframe': df_clean,
        'report': report
    }


def smart_type_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Intelligently infer and convert column types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with inferred types
    """
    df_copy = df.copy()
    
    for col in df_copy.columns:
        # Skip if already numeric or datetime
        if pd.api.types.is_numeric_dtype(df_copy[col]) or pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            continue
        
        # Try numeric conversion
        try:
            converted = pd.to_numeric(df_copy[col], errors='coerce')
            if converted.notna().sum() / len(df_copy) > 0.8:  # 80% success rate
                df_copy[col] = converted
                continue
        except Exception:
            pass
        
        # Try datetime conversion
        try:
            converted = pd.to_datetime(df_copy[col], errors='coerce')
            if converted.notna().sum() / len(df_copy) > 0.8:
                df_copy[col] = converted
                continue
        except Exception:
            pass
        
        # Convert to category if low cardinality
        if df_copy[col].nunique() / len(df_copy) < 0.05:  # Less than 5% unique
            df_copy[col] = df_copy[col].astype('category')
    
    return df_copy
