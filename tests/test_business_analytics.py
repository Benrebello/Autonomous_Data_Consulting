import pandas as pd
from tools import rfm_analysis, calculate_growth_rate, ab_test_analysis

def test_calculate_growth_rate():
    df = pd.DataFrame({
        'date': [1, 2, 3, 4],
        'value': [100, 110, 99, 150]
    })
    out = calculate_growth_rate(df, 'value', 'date')
    assert 'growth_rate' in out.columns
    assert out['growth_rate'].iloc[0] != out['growth_rate'].iloc[1]  # has pct_change values


def test_rfm_analysis_basic():
    df = pd.DataFrame({
        'customer': ['A', 'A', 'B', 'C'],
        'date': ['2024-01-01', '2024-02-01', '2024-01-10', '2024-01-15'],
        'value': [100, 150, 80, 50]
    })
    out = rfm_analysis(df, 'customer', 'date', 'value')
    assert 'RFM' in out.columns
    assert 'segment' in out.columns


def test_ab_test_analysis_basic():
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'metric': [10, 12, 9, 11]
    })
    res = ab_test_analysis(df, 'group', 'metric')
    assert 'p_value' in res and 'cohens_d' in res
