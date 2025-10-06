import pandas as pd
from tools import export_to_excel, export_analysis_results


def test_export_to_excel_bytes():
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': ['x', 'y', 'z']
    })
    out = export_to_excel(df, filename="test.xlsx", sheet_name="Data")
    assert isinstance(out, (bytes, bytearray))
    assert len(out) > 0


def test_export_analysis_results_bytes():
    dfs = {
        'sheet1': pd.DataFrame({'v': [10, 20]}),
        'metrics': {'acc': 0.9, 'f1': 0.8},
    }
    out = export_analysis_results(dfs, filename="analysis.xlsx")
    assert isinstance(out, (bytes, bytearray))
    assert len(out) > 0
