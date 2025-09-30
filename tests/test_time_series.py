import pandas as pd
import numpy as np
import tools

def test_decompose_time_series():
    # Create a simple seasonal series
    n = 48
    t = np.arange(n)
    # seasonal pattern with period 12
    series = 10 + 2 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(42).normal(0, 0.1, n)
    df = pd.DataFrame({'y': series})

    result = tools.decompose_time_series(df, 'y', period=12)

    assert set(result.keys()) == {'trend', 'seasonal', 'residual'}
    # trend/seasonal/residual should be lists of length n (some ends may be NaN -> they will be None after tolist)
    assert len(result['trend']) == n
    assert len(result['seasonal']) == n
    assert len(result['residual']) == n
