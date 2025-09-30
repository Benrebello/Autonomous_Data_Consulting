import types
import pandas as pd
import numpy as np
import streamlit as st
import pytest

from app import AnalysisPipeline
import tools


class DummyLLM:
    def invoke(self, prompt):
        class R:
            content = "{}"
        return R()
    def stream(self, prompt):
        # Yield a couple of chunks for streaming compatibility
        yield types.SimpleNamespace(content="Resumo inicial. ")
        yield types.SimpleNamespace(content="Conclusões finais.")


def build_test_data():
    # Create a reasonably rich dataframe to satisfy defaults
    n = 120
    df = pd.DataFrame({
        'a': np.linspace(0, 10, n),
        'b': np.linspace(5, 15, n) + np.random.normal(0, 0.1, n),
        'class': np.random.randint(0, 2, size=n),
        'category': np.random.choice(['x', 'y', 'z'], size=n),
        'text': np.random.choice(['hello world', 'data science is fun', 'streamlit charts'], size=n),
        'latitude': -23.5 + np.random.normal(0, 0.01, n),
        'longitude': -46.6 + np.random.normal(0, 0.01, n),
        'time': np.arange(n) * 3600,  # seconds
        'event': np.random.randint(0, 2, size=n),
        'status': np.random.randint(0, 2, size=n),
    })
    df2 = df.copy()
    # Slightly perturb second df
    df2['b'] = df2['b'] * 1.05
    return {'df1': df, 'df2': df2, 'default': df}


def test_all_tools_in_mapping_execute_without_exception():
    # Ensure a clean session state for streamlit charts
    st.session_state.clear()
    data = build_test_data()
    dfs = {
        'default': data['default'],
        'df1': data['df1'],
        'df2': data['df2'],
    }
    pipe = AnalysisPipeline(DummyLLM(), dfs, rpm_limit=1000)

    tool_mapping = pipe.tool_mapping
    assert isinstance(tool_mapping, dict) and len(tool_mapping) > 0

    failures = {}
    for name, func in tool_mapping.items():
        # Generate default inputs for each tool
        inputs = pipe._fill_default_inputs_for_task(name, {})
        # Special-case adjustments
        if name == 'compare_datasets':
            inputs['df1'] = dfs['df1']
            inputs['df2'] = dfs['df2']
        if name == 'evaluate_model':
            # Provide a simple model
            from sklearn.linear_model import LogisticRegression
            X = dfs['default'].select_dtypes(include=['number']).drop(columns=['class'], errors='ignore')
            y = dfs['default'].get('class')
            inputs['model'] = LogisticRegression(max_iter=200)
            inputs['X'] = X
            inputs['y'] = y
        if name == 'plot_geospatial_map':
            # Ensure columns exist (they do), else skip gracefully
            pass
        if name == 'perform_survival_analysis':
            # lifelines may not be installed in some environments; the function returns error dict in that case
            pass
        if name == 'perform_named_entity_recognition':
            # spaCy model might be missing; function returns an error dict; that's acceptable
            pass
        if name == 'generate_wordcloud':
            # Wordcloud may fail on empty text; ensure text_column is populated
            inputs['text_column'] = 'text'

        # Resolve context refs and execute function
        resolved = pipe._resolve_inputs(inputs)
        try:
            result = func(**resolved)
        except Exception as e:
            failures[name] = str(e)
            continue

        # Validate result types in a lenient but meaningful way
        ok = False
        if result is None:
            ok = name in {'plot_geospatial_map'}  # may return None if columns not present
        elif isinstance(result, (pd.DataFrame, pd.Series, dict, list, str, bytes)):
            ok = True
        elif isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], pd.DataFrame):
            # e.g., (corrected_df, report) from validate_and_correct_data_types
            ok = True
        else:
            # Some tools return buffer-like objects
            try:
                import io
                ok = isinstance(result, io.BytesIO) or hasattr(result, 'read') or hasattr(result, 'getvalue')
            except Exception:
                ok = False

        if not ok:
            failures[name] = f"Unexpected result type: {type(result)}"

    # Helpful log for debugging which tools failed in CI output
    if failures:
        print("Failures detail:", failures)
    assert not failures, f"Some tools failed or returned unexpected results: {failures}"
