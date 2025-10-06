import pandas as pd
from tools import decompose_time_series, forecast_arima

def test_decompose_time_series_basic():
    # Create a simple seasonal-ish series
    df = pd.DataFrame({'val': [1,2,3,4,5,6,7,8,9,10,11,12]})
    res = decompose_time_series(df, 'val', period=4)
    assert 'trend' in res and 'seasonal' in res and 'residual' in res
    assert isinstance(res['trend'], list)


def test_forecast_arima_basic():
    df = pd.DataFrame({'val': [i for i in range(1, 25)]})
    out = forecast_arima(df, 'val', order=(1,1,1), steps=3)
    assert 'forecast' in out and isinstance(out['forecast'], list)
