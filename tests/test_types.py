import pandas as pd
import numpy as np
import tools

def test_validate_and_correct_data_types():
    df = pd.DataFrame({
        'num_str': ['1', '2', '3'],
        'date_str': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'num_float': [1.5, 2.5, 3.5],
    })

    corrected, report = tools.validate_and_correct_data_types(df)

    # num_str should become numeric
    assert np.issubdtype(corrected['num_str'].dtype, np.number)
    # date_str should become datetime
    assert np.issubdtype(corrected['date_str'].dtype, np.datetime64)
    # num_float should remain numeric
    assert np.issubdtype(corrected['num_float'].dtype, np.number)
    # sanity of report keys
    assert set(report.keys()) == {'num_str', 'date_str', 'num_float'}
