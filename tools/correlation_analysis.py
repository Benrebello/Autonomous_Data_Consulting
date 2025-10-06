# tools/correlation_analysis.py
"""Correlation analysis and multicollinearity detection tools."""

import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
from typing import Dict, Any


@st.cache_data(show_spinner=False)
def correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate correlation matrix with statistical significance and interpretation.
    
    Returns correlation matrix with p-values and interpretation guidelines.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with correlation matrix, p-values, and interpretations
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return {
            'error': 'Need at least 2 numeric columns for correlation analysis',
            'available_columns': list(numeric_df.columns)
        }
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Calculate p-values for each correlation
    n = len(numeric_df)
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                            columns=corr_matrix.columns, 
                            index=corr_matrix.index)
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i != j:
                r = corr_matrix.iloc[i, j]
                # Calculate t-statistic and p-value
                t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2) if abs(r) < 1 else np.inf
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                p_values.iloc[i, j] = p_val
            else:
                p_values.iloc[i, j] = 0.0
    
    # Identify significant correlations
    significant_correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle to avoid duplicates
                r = corr_matrix.iloc[i, j]
                p = p_values.iloc[i, j]
                
                # Interpret strength
                abs_r = abs(r)
                if abs_r < 0.1:
                    strength = "negligible"
                elif abs_r < 0.3:
                    strength = "weak"
                elif abs_r < 0.5:
                    strength = "moderate"
                elif abs_r < 0.7:
                    strength = "strong"
                else:
                    strength = "very strong"
                
                direction = "positive" if r > 0 else "negative"
                significant = p < 0.05
                
                significant_correlations.append({
                    'variable1': col1,
                    'variable2': col2,
                    'correlation': float(r),
                    'p_value': float(p),
                    'significant': significant,
                    'strength': strength,
                    'direction': direction,
                    'interpretation': f"{strength} {direction} correlation (r={r:.3f}, p={p:.4f})"
                })
    
    # Sort by absolute correlation value
    significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'p_values': p_values.to_dict(),
        'significant_correlations': significant_correlations,
        'sample_size': n,
        'interpretation_guide': {
            'correlation_strength': {
                '0.0-0.1': 'negligible',
                '0.1-0.3': 'weak',
                '0.3-0.5': 'moderate',
                '0.5-0.7': 'strong',
                '0.7-1.0': 'very strong'
            },
            'significance': 'p < 0.05 indicates statistically significant correlation',
            'warning': 'Correlation does not imply causation. Always consider context and other factors.'
        }
    }


def get_variable_relations(df: pd.DataFrame, x_column: str, y_column: str) -> Dict[str, Any]:
    """Get relation between two variables: correlation if numeric, crosstab if categorical.
    
    Args:
        df: Input DataFrame
        x_column: First column
        y_column: Second column
        
    Returns:
        Dictionary with relationship information
    """
    if x_column in df.columns and y_column in df.columns:
        if df[x_column].dtype in ['int64', 'float64'] and df[y_column].dtype in ['int64', 'float64']:
            corr = df[[x_column, y_column]].corr().iloc[0, 1]
            return {'correlation': corr}
        else:
            crosstab = pd.crosstab(df[x_column], df[y_column])
            return {'crosstab': crosstab.to_dict()}
    return {}


def get_influential_variables(df: pd.DataFrame, target_column: str) -> Dict[str, float]:
    """Get correlation of all numeric variables with a numeric target.
    
    Returns empty dict if target is missing or non-numeric.
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        
    Returns:
        Dictionary with correlations
    """
    if target_column not in df.columns:
        return {}
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if target_column not in numeric_df.columns:
        return {}
    correlations = numeric_df.corr()[target_column].to_dict()
    return correlations


def multicollinearity_detection(df: pd.DataFrame, columns: list = None) -> Dict[str, Any]:
    """Detect multicollinearity using VIF (Variance Inflation Factor).
    
    Args:
        df: Input DataFrame
        columns: List of columns to check (None = all numeric)
        
    Returns:
        Dictionary with VIF scores and interpretation
    """
    from sklearn.linear_model import LinearRegression
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Need at least 2 columns
    if len(columns) < 2:
        return {"error": "Need at least 2 columns for VIF calculation"}
    
    vif_data = {}
    
    for i, col in enumerate(columns):
        # Use other columns as predictors
        X = df[[c for c in columns if c != col]].dropna()
        y = df[col].loc[X.index]
        
        if len(X) < 2:
            continue
        
        # Fit linear regression
        try:
            model = LinearRegression()
            model.fit(X, y)
            r_squared = model.score(X, y)
            
            # Calculate VIF
            if r_squared < 0.9999:  # Avoid division by zero
                vif = 1 / (1 - r_squared)
            else:
                vif = float('inf')
            
            # Interpret VIF
            if vif < 5:
                interpretation = "Low multicollinearity (OK)"
            elif vif < 10:
                interpretation = "Moderate multicollinearity (investigate)"
            else:
                interpretation = "High multicollinearity (consider removing or using PCA)"
            
            vif_data[col] = {
                'vif': vif,
                'r_squared': r_squared,
                'interpretation': interpretation
            }
        except Exception:
            vif_data[col] = {
                'vif': None,
                'error': 'Could not calculate VIF'
            }
    
    return {
        'vif_scores': vif_data,
        'interpretation_guide': {
            'VIF < 5': 'Low multicollinearity',
            '5 ≤ VIF < 10': 'Moderate multicollinearity',
            'VIF ≥ 10': 'High multicollinearity - consider removing variable'
        }
    }
