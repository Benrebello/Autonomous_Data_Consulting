import pandas as pd
from tools import run_kmeans_clustering, get_clusters_summary


def test_run_kmeans_clustering_basic():
    df = pd.DataFrame({
        'x': [1, 2, 3, 10, 11, 12],
        'y': [1, 2, 3, 10, 11, 12]
    })
    out = run_kmeans_clustering(df, columns=['x', 'y'], n_clusters=2)
    assert 'cluster' in out.columns
    assert out['cluster'].notna().sum() == len(df)


def test_get_clusters_summary_basic():
    df = pd.DataFrame({
        'a': [1, 2, 3, 10, 11, 12],
        'b': [2, 3, 4, 11, 12, 13]
    })
    res = get_clusters_summary(df, n_clusters=2)
    assert 'cluster_centers' in res
    assert 'labels' in res
