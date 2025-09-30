import pandas as pd
import tools

def test_detect_outliers_iqr():
    df = pd.DataFrame({'x': [1, 2, 3, 1000]})
    res = tools.detect_outliers(df.copy(), 'x', method='iqr')
    # Should mark 1000 as outlier
    assert res['outlier'].iloc[-1] is True


def test_detect_outliers_zscore():
    df = pd.DataFrame({'x': [1, 2, 3, 1000]})
    res = tools.detect_outliers(df.copy(), 'x', method='zscore')
    # Should mark 1000 as outlier
    assert res['outlier'].iloc[-1] is True
