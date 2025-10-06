import pandas as pd
from tools import create_polynomial_features, create_lag_features, create_rolling_features


def test_create_polynomial_features_basic():
    df = pd.DataFrame({'x': [1, 2, 3]})
    out = create_polynomial_features(df, columns=['x'], degree=2)
    assert 'x^2' in out.columns or 'x^2' in ''.join(out.columns)


def test_create_lag_and_rolling_features_basic():
    df = pd.DataFrame({'val': [10, 20, 30, 40]})
    out = create_lag_features(df, column='val', lags=[1])
    assert 'val_lag_1' in out.columns
    out2 = create_rolling_features(df, column='val', windows=[2])
    assert 'val_roll_mean_2' in out2.columns
