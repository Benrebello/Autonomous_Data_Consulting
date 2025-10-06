# tools/clustering.py
"""Clustering tools."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any
import streamlit as st


@st.cache_data(show_spinner=False)
def run_kmeans_clustering(df: pd.DataFrame, columns: list, n_clusters: int) -> pd.DataFrame:
    """Run K-Means clustering without mutating the input DataFrame.
    
    Returns a copied DataFrame with a new 'cluster' column aligned to the
    original indices of the selected data.
    """
    data_to_cluster = df[columns].dropna()
    scaled_data = StandardScaler().fit_transform(data_to_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    df_copy = df.copy()
    df_copy.loc[data_to_cluster.index, 'cluster'] = clusters
    return df_copy


def get_clusters_summary(df: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
    """Perform K-Means and return cluster centers summary and labels.
    
    Args:
        df: Input DataFrame
        n_clusters: Number of clusters
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {}
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(numeric_df)
    return {
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'labels': kmeans.labels_.tolist()
    }
