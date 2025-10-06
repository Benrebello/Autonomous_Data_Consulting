# tools/advanced_analytics.py
"""Advanced analytics tools: risk, sensitivity, simulation, causal inference."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable


def forecast_time_series_arima(df: pd.DataFrame, column: str, periods: int = 10) -> Dict[str, Any]:
    """Forecast time series using ARIMA (light wrapper).
    
    Args:
        df: Input DataFrame
        column: Column to forecast
        periods: Number of periods ahead
    
    Returns:
        Dictionary with forecast values
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        series = pd.to_numeric(df[column], errors='coerce').dropna()
        if len(series) < 10:
            return {'error': 'Insufficient data for ARIMA'}
        model = ARIMA(series, order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=periods)
        return {'forecast': forecast.tolist()}
    except Exception as e:
        return {'error': str(e)}


def risk_assessment(df: pd.DataFrame, risk_factors: List[str], weights: Optional[List[float]] = None) -> Dict[str, Any]:
    """Simple risk score based on weighted sum of risk factors.
    
    Args:
        df: Input DataFrame
        risk_factors: List of column names representing risk factors
        weights: Optional weights for each factor (defaults to equal)
    
    Returns:
        Dictionary with risk scores per row
    """
    if not risk_factors:
        return {'error': 'No risk factors provided'}
    
    available = [c for c in risk_factors if c in df.columns]
    if not available:
        return {'error': 'None of the risk factors exist in DataFrame'}
    
    if weights is None:
        weights = [1.0] * len(available)
    else:
        weights = weights[:len(available)]
    
    risk_df = df[available].fillna(0)
    for col in risk_df.columns:
        risk_df[col] = pd.to_numeric(risk_df[col], errors='coerce').fillna(0)
    
    risk_score = sum(risk_df[col] * w for col, w in zip(available, weights))
    return {
        'risk_scores': risk_score.tolist(),
        'mean_risk': float(risk_score.mean()),
        'max_risk': float(risk_score.max())
    }


def sensitivity_analysis(base_value: float, variable_changes: Dict[str, List[float]], 
                        impact_function: Optional[Callable] = None) -> Dict[str, Any]:
    """Perform sensitivity analysis by varying input variables.
    
    Args:
        base_value: Base output value
        variable_changes: Dict mapping variable names to list of change percentages
        impact_function: Optional function to compute impact (defaults to linear)
    
    Returns:
        Dictionary with sensitivity results per variable
    """
    if impact_function is None:
        impact_function = lambda base, pct: base * (1 + pct)
    
    results = {}
    for var, changes in variable_changes.items():
        impacts = [impact_function(base_value, pct) for pct in changes]
        results[var] = {
            'changes': changes,
            'impacts': impacts,
            'sensitivity': max(impacts) - min(impacts) if impacts else 0
        }
    return results


def monte_carlo_simulation(variables: Dict[str, Dict[str, float]], n_simulations: int = 1000, 
                           output_function: Optional[Callable] = None) -> Dict[str, Any]:
    """Run Monte Carlo simulation with random variable sampling.
    
    Args:
        variables: Dict mapping var names to {'mean': x, 'std': y}
        n_simulations: Number of simulation runs
        output_function: Function to compute output from sampled variables
    
    Returns:
        Dictionary with simulation results
    """
    if output_function is None:
        output_function = lambda **kwargs: sum(kwargs.values())
    
    samples = {}
    for var, params in variables.items():
        mean = params.get('mean', 0)
        std = params.get('std', 1)
        samples[var] = np.random.normal(mean, std, n_simulations)
    
    outputs = []
    for i in range(n_simulations):
        sample_vals = {var: samples[var][i] for var in variables}
        outputs.append(output_function(**sample_vals))
    
    return {
        'mean': float(np.mean(outputs)),
        'std': float(np.std(outputs)),
        'percentiles': {
            '5%': float(np.percentile(outputs, 5)),
            '50%': float(np.percentile(outputs, 50)),
            '95%': float(np.percentile(outputs, 95))
        }
    }


def perform_causal_inference(df: pd.DataFrame, treatment: str, outcome: str, 
                             controls: Optional[List[str]] = None) -> Dict[str, Any]:
    """Simple causal inference using linear regression with controls.
    
    Args:
        df: Input DataFrame
        treatment: Treatment variable column
        outcome: Outcome variable column
        controls: Optional control variables
    
    Returns:
        Dictionary with treatment effect estimate
    """
    from sklearn.linear_model import LinearRegression
    
    if treatment not in df.columns or outcome not in df.columns:
        return {'error': 'Treatment or outcome column not found'}
    
    X_cols = [treatment]
    if controls:
        X_cols.extend([c for c in controls if c in df.columns])
    
    data = df[X_cols + [outcome]].dropna()
    if len(data) < 10:
        return {'error': 'Insufficient data for causal inference'}
    
    X = data[X_cols]
    y = data[outcome]
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Safely extract scalar coefficient - treatment is always first column
    try:
        treatment_effect = float(model.coef_[0])
    except (TypeError, ValueError):
        treatment_effect = 0.0
    
    return {
        'treatment_effect': treatment_effect,
        'interpretation': f'One unit increase in {treatment} is associated with {treatment_effect:.3f} change in {outcome}'
    }


def perform_named_entity_recognition(df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
    """Placeholder for NER (requires spacy or similar).
    
    Returns error message indicating external dependency needed.
    """
    return {
        'error': 'NER requires spacy or similar NLP library',
        'suggestion': 'Install spacy and download model: python -m spacy download en_core_web_sm'
    }


def text_summarization(text: str, max_sentences: int = 3) -> str:
    """Simple extractive summarization by taking first N sentences.
    
    Args:
        text: Input text
        max_sentences: Maximum sentences to include
    
    Returns:
        Summarized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Simple sentence split
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return '. '.join(sentences[:max_sentences]) + ('.' if sentences else '')
