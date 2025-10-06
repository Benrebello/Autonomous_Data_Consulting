# tools/geometry.py
"""Geometric calculations and distance metrics.

Provides functions for:
- Euclidean distance
- Haversine distance (geographic)
- Polygon area calculation
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def euclidean_distance(df: pd.DataFrame, x1: str, y1: str, x2: str, y2: str, new_col: str) -> pd.DataFrame:
    """Calculate Euclidean distance between two points in 2D space.
    
    Args:
        df: Input DataFrame
        x1: First point x-coordinate column
        y1: First point y-coordinate column
        x2: Second point x-coordinate column
        y2: Second point y-coordinate column
        new_col: Name for distance column
    
    Returns:
        DataFrame with distance column added
    """
    df_result = df.copy()
    df_result[new_col] = np.sqrt((df_result[x1] - df_result[x2])**2 + (df_result[y1] - df_result[y2])**2)
    return df_result


def haversine_distance_df(df: pd.DataFrame, lat1_col: str, lon1_col: str, 
                          lat2_col: str, lon2_col: str, new_col: str) -> pd.DataFrame:
    """Calculate Haversine distance between geographic coordinates.
    
    Computes great-circle distance between two points on Earth's surface.
    
    Args:
        df: Input DataFrame
        lat1_col: First point latitude column (degrees)
        lon1_col: First point longitude column (degrees)
        lat2_col: Second point latitude column (degrees)
        lon2_col: Second point longitude column (degrees)
        new_col: Name for distance column (km)
    
    Returns:
        DataFrame with distance column in kilometers
    """
    df_result = df.copy()
    
    R = 6371  # Earth radius in km
    
    # Convert to radians
    dlat = np.radians(df_result[lat2_col] - df_result[lat1_col])
    dlon = np.radians(df_result[lon2_col] - df_result[lon1_col])
    lat1_rad = np.radians(df_result[lat1_col])
    lat2_rad = np.radians(df_result[lat2_col])
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    df_result[new_col] = R * c
    return df_result


def polygon_area(points: List[Tuple[float, float]]) -> float:
    """Calculate area of a polygon using the Shoelace formula.
    
    Args:
        points: List of (x, y) coordinate tuples defining polygon vertices
    
    Returns:
        Area of polygon
    """
    if len(points) < 3:
        return 0.0
    
    points_array = np.array(points)
    x = points_array[:, 0]
    y = points_array[:, 1]
    
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return float(area)
