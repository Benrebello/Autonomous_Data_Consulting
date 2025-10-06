# tools/math_operations.py
"""Mathematical operations on DataFrame columns.

Provides column-wise arithmetic operations, calculus operations,
and mathematical transformations.
"""

import pandas as pd
import numpy as np
from typing import Optional


def add_columns(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> pd.DataFrame:
    """Add two columns element-wise.
    
    Args:
        df: Input DataFrame
        col1: First column name
        col2: Second column name
        new_col: Name for result column
    
    Returns:
        DataFrame with new column
    """
    df_result = df.copy()
    df_result[new_col] = df_result[col1] + df_result[col2]
    return df_result


def subtract_columns(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> pd.DataFrame:
    """Subtract col2 from col1 element-wise.
    
    Args:
        df: Input DataFrame
        col1: First column name
        col2: Second column name
        new_col: Name for result column
    
    Returns:
        DataFrame with new column
    """
    df_result = df.copy()
    df_result[new_col] = df_result[col1] - df_result[col2]
    return df_result


def multiply_columns(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> pd.DataFrame:
    """Multiply two columns element-wise.
    
    Args:
        df: Input DataFrame
        col1: First column name
        col2: Second column name
        new_col: Name for result column
    
    Returns:
        DataFrame with new column
    """
    df_result = df.copy()
    df_result[new_col] = df_result[col1] * df_result[col2]
    return df_result


def divide_columns(df: pd.DataFrame, col1: str, col2: str, new_col: str) -> pd.DataFrame:
    """Divide col1 by col2 element-wise.
    
    Args:
        df: Input DataFrame
        col1: Numerator column name
        col2: Denominator column name
        new_col: Name for result column
    
    Returns:
        DataFrame with new column
    """
    df_result = df.copy()
    df_result[new_col] = df_result[col1] / df_result[col2]
    return df_result


def apply_math_function(df: pd.DataFrame, column: str, func_name: str, new_col: Optional[str] = None) -> pd.DataFrame:
    """Apply a mathematical function to a column.
    
    Supported functions: log, exp, sqrt, sin, cos, tan, abs
    
    Args:
        df: Input DataFrame
        column: Column to transform
        func_name: Name of function to apply
        new_col: Optional name for result column (if None, modifies in place)
    
    Returns:
        DataFrame with transformed column
    """
    func_map = {
        'log': np.log,
        'exp': np.exp,
        'sqrt': np.sqrt,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'abs': np.abs
    }
    
    df_result = df.copy()
    if func_name in func_map:
        target_col = new_col if new_col else column
        df_result[target_col] = func_map[func_name](df_result[column])
    
    return df_result


def compute_numerical_derivative(df: pd.DataFrame, column: str, new_col: str, dx: float = 1.0) -> pd.DataFrame:
    """Compute numerical derivative using numpy.gradient.
    
    Args:
        df: Input DataFrame
        column: Column to differentiate
        new_col: Name for derivative column
        dx: Spacing between points
    
    Returns:
        DataFrame with derivative column
    """
    df_result = df.copy()
    df_result[new_col] = np.gradient(df_result[column].values, dx)
    return df_result


def compute_numerical_integral(df: pd.DataFrame, column: str, new_col: str) -> pd.DataFrame:
    """Compute numerical integral using cumulative sum.
    
    Args:
        df: Input DataFrame
        column: Column to integrate
        new_col: Name for integral column
    
    Returns:
        DataFrame with integral column
    """
    df_result = df.copy()
    values = df_result[column].values
    # Use index spacing if available, otherwise assume dx=1
    dx = df_result.index[1] - df_result.index[0] if len(df_result) > 1 else 1
    integral = np.cumsum(values) * dx
    df_result[new_col] = integral
    return df_result
