# tools/outlier_detection.py
"""Outlier detection and removal tools."""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any


@st.cache_data(show_spinner=False)
def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """Detect outliers using IQR or (modified) Z-score.
    
    - iqr: classic Tukey fences (1.5 * IQR)
    - zscore: Modified Z-Score based on median and MAD (threshold 3.5)
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        method: Detection method ('iqr' or 'zscore')
        
    Returns:
        DataFrame with 'outlier' column added
    """
    df = df.copy()
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        # Ensure native Python bools (dtype object) for identity checks in tests
        df['outlier'] = pd.Series([bool(v) for v in mask.tolist()], dtype=object)
    elif method == 'zscore':
        # Modified Z-Score using median and MAD (more robust for small samples)
        median = df[column].median()
        abs_dev = (df[column] - median).abs()
        MAD = abs_dev.median()
        if MAD == 0:
            # Fallback to standard deviation if MAD is zero
            mean = df[column].mean()
            std = df[column].std(ddof=0) if df[column].std(ddof=0) != 0 else 1.0
            z = (df[column] - mean) / std
            df['z_score'] = z
            mask = z.abs() > 3
        else:
            mod_z = 0.6745 * (df[column] - median) / MAD
            df['z_score'] = mod_z
            mask = mod_z.abs() > 3.5
        df['outlier'] = pd.Series([bool(v) for v in mask.tolist()], dtype=object)
    return df


def get_outliers_summary(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Summarize outliers using IQR method.
    
    Args:
        df: Input DataFrame
        column: Column to analyze
        
    Returns:
        Dictionary with outlier statistics
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return {
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / len(df) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def detect_and_remove_outliers(df: pd.DataFrame, column: str, 
                                method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """Detect and remove outliers from DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column to check
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for zscore)
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if method == 'iqr':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    
    elif method == 'zscore':
        mean = df_clean[column].mean()
        std = df_clean[column].std()
        z_scores = np.abs((df_clean[column] - mean) / std)
        df_clean = df_clean[z_scores <= threshold]
    
    return df_clean
