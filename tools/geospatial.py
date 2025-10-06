# tools/geospatial.py
"""Geospatial plotting utilities."""

import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from typing import Optional


def plot_geospatial_map(df: pd.DataFrame, lat_column: str, lon_column: str) -> Optional[BytesIO]:
    """Plot simple geospatial map if lat/lon available.
    
    Args:
        df: Input DataFrame
        lat_column: Latitude column name
        lon_column: Longitude column name
        
    Returns:
        BytesIO buffer with plot or None if columns not found
    """
    if lat_column in df.columns and lon_column in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[lon_column], df[lat_column], alpha=0.5)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geospatial Scatter Plot')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    return None
