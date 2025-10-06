# tools/time_series.py
"""Time series analysis tools."""

import pandas as pd
from typing import Dict, Any, List, Optional
import streamlit as st


def decompose_time_series(df: pd.DataFrame, column: str, period: int = 12) -> Dict[str, List[Optional[float]]]:
    """Decompose time series into trend, seasonal, and residual components."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    series = df[column]
    decomposition = seasonal_decompose(series, model='additive', period=period)
    return {
        'trend': decomposition.trend.tolist(),
        'seasonal': decomposition.seasonal.tolist(),
        'residual': decomposition.resid.tolist()
    }


@st.cache_data(show_spinner=False)
def forecast_arima(df: pd.DataFrame, column: str, order=(1, 1, 1), steps: int = 10) -> Dict[str, Any]:
    """Forecast time series using ARIMA."""
    from statsmodels.tsa.arima.model import ARIMA
    series = df[column]
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return {
        'forecast': forecast.tolist(),
        'aic': model_fit.aic
    }


def add_time_features_from_seconds(df: pd.DataFrame, time_column: str, origin: str = "2000-01-01") -> pd.DataFrame:
    """Create datetime-based features from a numeric seconds column without mutating semantics.

    This function assumes `time_column` contains elapsed seconds (e.g., since the first observation),
    and generates additional features (datetime, date, hour, day, weekday, month, quarter, year).
    """
    df_result = df.copy()
    # Convert seconds to datetime and extract useful features
    base_datetime = pd.to_datetime(origin)
    dt_series = base_datetime + pd.to_timedelta(df_result[time_column].astype(float), unit='s')

    df_result[f'{time_column}_datetime'] = dt_series
    df_result[f'{time_column}_date'] = dt_series.dt.date
    df_result[f'{time_column}_hour'] = dt_series.dt.hour
    df_result[f'{time_column}_day'] = dt_series.dt.day
    df_result[f'{time_column}_weekday'] = dt_series.dt.weekday
    # Alias commonly used in tests: dayofweek (same as weekday)
    df_result[f'{time_column}_dayofweek'] = dt_series.dt.dayofweek
    df_result[f'{time_column}_month'] = dt_series.dt.month
    df_result[f'{time_column}_quarter'] = dt_series.dt.quarter
    df_result[f'{time_column}_year'] = dt_series.dt.year

    return df_result


def get_temporal_patterns(df: pd.DataFrame, time_column: str, value_column: str):
    """Simple temporal pattern: correlation between time and value.

    If time column exists, attempt to compute correlation between
    encoded time (as ordinal) and value column.
    """
    if time_column in df.columns and value_column in df.columns:
        s = df[time_column]
        # If datetime, use ordinal; else try numeric casting
        try:
            if pd.api.types.is_datetime64_any_dtype(s):
                x = s.map(lambda d: d.toordinal())
            else:
                x = pd.to_numeric(s, errors='coerce')
            y = pd.to_numeric(df[value_column], errors='coerce')
            cor = pd.concat([x, y], axis=1).dropna().corr().iloc[0, 1]
            return {'correlation_time_value': float(cor) if pd.notna(cor) else None}
        except Exception:
            return {'correlation_time_value': None}
    return {}
