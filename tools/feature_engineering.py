# tools/feature_engineering.py
"""Feature engineering utilities."""

import pandas as pd
from typing import List


def create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
    """Create polynomial features up to specified degree.
    
    Args:
        df: Input DataFrame
        columns: Columns to transform
        degree: Polynomial degree
    
    Returns:
        DataFrame with original + polynomial features
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(df[columns])
    feature_names = poly.get_feature_names_out(columns)
    df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
    
    return pd.concat([df, df_poly], axis=1)


def create_interaction_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Create interaction features between specified columns."""
    df_result = df.copy()
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_i, col_j = columns[i], columns[j]
            name = f"{col_i}_x_{col_j}"
            df_result[name] = df_result[col_i] * df_result[col_j]
    return df_result


def create_rolling_features(df: pd.DataFrame, column: str, windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
    """Create rolling statistics features for time series."""
    df_result = df.copy()
    for w in windows:
        df_result[f"{column}_roll_mean_{w}"] = df_result[column].rolling(window=w, min_periods=1).mean()
        df_result[f"{column}_roll_std_{w}"] = df_result[column].rolling(window=w, min_periods=1).std()
        df_result[f"{column}_roll_min_{w}"] = df_result[column].rolling(window=w, min_periods=1).min()
        df_result[f"{column}_roll_max_{w}"] = df_result[column].rolling(window=w, min_periods=1).max()
    return df_result


def create_lag_features(df: pd.DataFrame, column: str, lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
    """Create lag features for time series."""
    df_result = df.copy()
    for lag in lags:
        df_result[f"{column}_lag_{lag}"] = df_result[column].shift(lag)
    return df_result


def create_binning(df: pd.DataFrame, column: str, bins: int = 5, strategy: str = 'quantile') -> pd.DataFrame:
    """Create binned categorical feature from numeric column.

    Args:
        df: Input DataFrame
        column: Column to bin
        bins: Number of bins
        strategy: 'quantile' or 'uniform'

    Returns:
        DataFrame with new binned column
    """
    df_result = df.copy()
    if column not in df_result.columns:
        return df_result
    
    try:
        if strategy == 'quantile':
            df_result[f'{column}_binned'] = pd.qcut(df_result[column], q=bins, labels=False, duplicates='drop')
        else:  # uniform
            df_result[f'{column}_binned'] = pd.cut(df_result[column], bins=bins, labels=False)
    except Exception:
        # Fallback if binning fails
        df_result[f'{column}_binned'] = 0
    
    return df_result
