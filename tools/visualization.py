# tools/visualization.py
"""Visualization and charting tools."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import BytesIO
from typing import Optional


def plot_histogram(df: pd.DataFrame, column: str) -> str:
    """Generate histogram for a column and store bytes in memory.
    
    Args:
        df: Input DataFrame
        column: Column name to plot
        
    Returns:
        Success message
    """
    plt.figure()
    sns.histplot(data=df, x=column)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    st.session_state.charts.append({'bytes': buf.getvalue(), 'caption': f'Histograma: {column}'})
    return f"Histogram for {column} generated."


def plot_boxplot(df: pd.DataFrame, column: str) -> str:
    """Generate boxplot to detect outliers and store bytes in memory.
    
    Args:
        df: Input DataFrame
        column: Column name to plot
        
    Returns:
        Success message
    """
    plt.figure()
    sns.boxplot(data=df, y=column)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    st.session_state.charts.append({'bytes': buf.getvalue(), 'caption': f'Boxplot: {column}'})
    return f"Boxplot for {column} generated."


def plot_scatter(df: pd.DataFrame, x_column: str, y_column: str) -> str:
    """Generate scatter plot and store bytes in memory.
    
    Args:
        df: Input DataFrame
        x_column: Column for x-axis
        y_column: Column for y-axis
        
    Returns:
        Success message
    """
    plt.figure()
    sns.scatterplot(data=df, x=x_column, y=y_column)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    st.session_state.charts.append({'bytes': buf.getvalue(), 'caption': f'Scatter: {x_column} x {y_column}'})
    return f"Scatter plot between {x_column} and {y_column} generated."


def generate_chart(df: pd.DataFrame, chart_type: str, x_column: str, y_column: Optional[str] = None) -> str:
    """Generate a chart and store bytes in memory.
    
    Args:
        df: Input DataFrame
        chart_type: Type of chart ('bar', 'hist', 'scatter', 'box')
        x_column: Column for x-axis
        y_column: Column for y-axis (optional, depends on chart type)
        
    Returns:
        Success message
    """
    plt.figure()
    if chart_type == 'bar':
        sns.barplot(data=df, x=x_column, y=y_column)
    elif chart_type == 'hist':
        sns.histplot(data=df, x=x_column)
    elif chart_type == 'scatter':
        sns.scatterplot(data=df, x=x_column, y=y_column)
    elif chart_type == 'box':
        sns.boxplot(data=df, y=x_column)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    st.session_state.charts.append({'bytes': buf.getvalue(), 'caption': f"{chart_type} chart: {x_column}{' vs ' + y_column if y_column else ''}"})
    return "Chart generated successfully."


def plot_heatmap(df: pd.DataFrame, columns: Optional[list] = None) -> BytesIO:
    """Plot correlation heatmap for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to include (None = all numeric)
        
    Returns:
        BytesIO buffer with plot
    """
    import numpy as np
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def plot_line_chart(df: pd.DataFrame, x_column: str, y_column: str) -> BytesIO:
    """Plot line chart for time series.
    
    Args:
        df: Input DataFrame
        x_column: Column for x-axis (typically time)
        y_column: Column for y-axis (values)
        
    Returns:
        BytesIO buffer with plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column], df[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Line Chart: {y_column} over {x_column}')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def plot_violin_plot(df: pd.DataFrame, x_column: str, y_column: str) -> BytesIO:
    """Plot violin plot for distribution comparison.
    
    Args:
        df: Input DataFrame
        x_column: Categorical column for grouping
        y_column: Numeric column for distribution
        
    Returns:
        BytesIO buffer with plot
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df[x_column], y=df[y_column])
    plt.title(f'Violin Plot: {y_column} by {x_column}')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


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


def generate_wordcloud(df: pd.DataFrame, text_column: str) -> Optional[BytesIO]:
    """Generate wordcloud from text column.
    
    Args:
        df: Input DataFrame
        text_column: Column containing text data
        
    Returns:
        BytesIO buffer with plot or None if wordcloud not available
    """
    try:
        from wordcloud import WordCloud
    except Exception:
        # Graceful fallback when optional dependency is missing
        return None
    
    text = ' '.join(df[text_column].dropna().astype(str))
    wc = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf
