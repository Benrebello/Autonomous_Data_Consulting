import pandas as pd
import tools

def test_add_time_features_from_seconds():
    df = pd.DataFrame({
        'time': [0, 3600, 90000]  # 0s, 1h, 25h
    })
    df2 = tools.add_time_features_from_seconds(df, 'time')
    assert 'time_datetime' in df2.columns
    assert 'time_hour' in df2.columns
    assert 'time_dayofweek' in df2.columns
    assert 'time_month' in df2.columns
    # Original column must remain unchanged
    assert (df2['time'] == df['time']).all()
    # Hours should map 0,1, and 1 (since 25h wraps next day)
    assert df2['time_hour'].tolist() == [0, 1, 1]
