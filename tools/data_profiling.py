# tools/data_profiling.py
"""Data profiling and exploratory data analysis tools."""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any
from scipy import stats


@st.cache_data(show_spinner=False)
def descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate descriptive statistics for all numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with statistics, types, null counts, and shape
    """
    stats_dict = df.describe().to_dict()
    types = df.dtypes.to_dict()
    null_counts = df.isnull().sum().to_dict()
    return {
        'stats': stats_dict,
        'types': types,
        'null_counts': null_counts,
        'shape': df.shape
    }


def get_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Return the data types of each column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping column names to data types
    """
    return df.dtypes.astype(str).to_dict()


def get_central_tendency(df: pd.DataFrame) -> Dict[str, Any]:
    """Return central tendency measures: mean and median for numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with mean and median for each numeric column
    """
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}
    if not numeric_df.empty:
        result['mean'] = numeric_df.mean().to_dict()
        result['median'] = numeric_df.median().to_dict()
    return result


def get_variability(df: pd.DataFrame) -> Dict[str, Any]:
    """Return variability measures: std and variance for numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with standard deviation and variance
    """
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}
    if not numeric_df.empty:
        result['std'] = numeric_df.std().to_dict()
        result['var'] = numeric_df.var().to_dict()
    return result


def get_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Return min and max for numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with min and max values
    """
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}
    if not numeric_df.empty:
        result['min'] = numeric_df.min().to_dict()
        result['max'] = numeric_df.max().to_dict()
    return result


def calculate_min_max_per_variable(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate min and max for each numeric variable (column).
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with min/max for each column
    """
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}
    if not numeric_df.empty:
        for col in numeric_df.columns:
            result[col] = {
                'min': numeric_df[col].min(),
                'max': numeric_df[col].max()
            }
    return result


def get_value_counts(df: pd.DataFrame, column: str) -> Dict[Any, int]:
    """Return value counts for a specific column.
    
    Args:
        df: Input DataFrame
        column: Column name
        
    Returns:
        Dictionary with value counts
    """
    return df[column].value_counts().to_dict()


def get_frequent_values(df: pd.DataFrame, column: str, top_n: int = 10) -> Dict[Any, int]:
    """Return the most frequent values in a column.
    
    Args:
        df: Input DataFrame
        column: Column name
        top_n: Number of top values to return
        
    Returns:
        Dictionary with top N frequent values
    """
    return df[column].value_counts().head(top_n).to_dict()


def check_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for duplicate rows.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with duplicate count and percentage
    """
    duplicates = df.duplicated().sum()
    return {"duplicate_rows": duplicates, "percentage": duplicates / len(df) * 100}


def data_profiling(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data profiling with quality score.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with comprehensive profiling information
    """
    profile = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
    }
    
    # Calculate quality score (0-100)
    missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    quality_score = max(0, 100 - missing_pct)
    profile['quality_score'] = quality_score
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        profile['numeric_summary'] = {
            'count': len(numeric_cols),
            'columns': list(numeric_cols)
        }
    
    # Categorical columns summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        profile['categorical_summary'] = {
            'count': len(cat_cols),
            'columns': list(cat_cols)
        }
    
    return profile


def missing_data_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing data patterns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with missing data analysis
    """
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100)
    
    # Columns with missing data
    cols_with_missing = missing_counts[missing_counts > 0].to_dict()
    
    # Categorize severity
    severe = missing_pct[missing_pct > 50].index.tolist()
    moderate = missing_pct[(missing_pct > 20) & (missing_pct <= 50)].index.tolist()
    mild = missing_pct[(missing_pct > 0) & (missing_pct <= 20)].index.tolist()
    
    return {
        'total_missing': df.isnull().sum().sum(),
        'total_cells': df.shape[0] * df.shape[1],
        'overall_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'columns_with_missing': cols_with_missing,
        'severity': {
            'severe (>50%)': severe,
            'moderate (20-50%)': moderate,
            'mild (<20%)': mild
        },
        'recommendations': {
            'severe': 'Consider removing these columns' if severe else 'None',
            'moderate': 'Use advanced imputation' if moderate else 'None',
            'mild': 'Simple imputation (mean/median/mode)' if mild else 'None'
        }
    }


def cardinality_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze cardinality of columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with cardinality information
    """
    cardinality = {}
    
    for col in df.columns:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)
        
        # Categorize
        if unique_ratio == 1.0:
            category = "unique_identifier"
        elif unique_ratio > 0.95:
            category = "high_cardinality"
        elif unique_ratio < 0.05:
            category = "low_cardinality"
        else:
            category = "medium_cardinality"
        
        cardinality[col] = {
            'unique_count': unique_count,
            'unique_ratio': unique_ratio,
            'category': category
        }
    
    return cardinality


def calculate_skewness_kurtosis(df: pd.DataFrame, columns: list = None) -> Dict[str, Dict[str, float]]:
    """Calculate skewness and kurtosis for numeric columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to analyze (None = all numeric)
        
    Returns:
        Dictionary with skewness and kurtosis for each column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    result = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            result[col] = {
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
    
    return result


def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect various data quality issues.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with detected issues
    """
    issues = {
        'missing_data': df.isnull().sum().sum() > 0,
        'duplicates': df.duplicated().sum() > 0,
        'issues_found': []
    }
    
    # Check for missing data
    if issues['missing_data']:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        issues['issues_found'].append(f"Missing data: {missing_pct:.2f}% of cells")
    
    # Check for duplicates
    if issues['duplicates']:
        dup_count = df.duplicated().sum()
        issues['issues_found'].append(f"Duplicate rows: {dup_count} ({dup_count/len(df)*100:.2f}%)")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        issues['issues_found'].append(f"Constant columns: {', '.join(constant_cols)}")
    
    # Check for high cardinality in small datasets
    if len(df) < 1000:
        high_card_cols = [col for col in df.columns if df[col].nunique() / len(df) > 0.9]
        if high_card_cols:
            issues['issues_found'].append(f"High cardinality columns: {', '.join(high_card_cols)}")
    
    issues['quality_score'] = max(0, 100 - len(issues['issues_found']) * 10)
    
    return issues


def get_exploratory_analysis(df: pd.DataFrame) -> str:
    """Generate a light EDA text summary combining stats and correlations.

    Returns a short string to be displayed in UI/tests without heavy processing.
    """
    try:
        stats = descriptive_stats(df)
        from tools.correlation_analysis import correlation_matrix
        corr = correlation_matrix(df)
        nrows, ncols = stats.get('shape', (len(df), len(df.columns)))
        text = [
            f"Shape: {nrows} x {ncols}",
            f"Numeric columns: {len(df.select_dtypes(include=['number']).columns)}",
            "Correlation analysis: OK" if isinstance(corr, dict) else "Correlation analysis: N/A",
        ]
        return "\n".join(text)
    except Exception:
        return "EDA summary unavailable"
