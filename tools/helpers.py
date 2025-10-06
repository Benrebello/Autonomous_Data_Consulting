# tools/helpers.py
"""Internal helper functions for data analysis tools.

These are private utility functions used by other tools modules.
Not intended for direct use by agents.
"""

import pandas as pd
from typing import Dict, Any


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
    
    Returns:
        String interpretation (negligible, small, medium, large)
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def ab_test_recommendation(p_value: float, cohens_d: float, mean_a: float, mean_b: float) -> str:
    """Provide recommendation based on A/B test results.
    
    Args:
        p_value: Statistical significance p-value
        cohens_d: Cohen's d effect size
        mean_a: Mean of group A
        mean_b: Mean of group B
    
    Returns:
        Recommendation string
    """
    if p_value >= 0.05:
        return "No significant difference detected. Consider running test longer or with larger sample."
    
    winner = "B" if mean_b > mean_a else "A"
    effect = interpret_effect_size(cohens_d)
    
    return f"Group {winner} performs significantly better with {effect} effect size. Consider implementing."


def classify_rfm_segment(r_score: int, f_score: int, m_score: int) -> str:
    """Classify RFM segment based on scores.
    
    Args:
        r_score: Recency score (1-5)
        f_score: Frequency score (1-5)
        m_score: Monetary score (1-5)
    
    Returns:
        Segment label
    """
    if r_score >= 4 and f_score >= 4 and m_score >= 4:
        return "Champions"
    if r_score >= 4 and f_score >= 3:
        return "Loyal Customers"
    if r_score <= 2 and f_score >= 4 and m_score >= 4:
        return "At Risk"
    return "Potential Loyalists"


def get_imputation_recommendation(df: pd.DataFrame, column: str) -> str:
    """Recommend imputation strategy based on data characteristics.
    
    Args:
        df: Input DataFrame
        column: Column to analyze
    
    Returns:
        Recommendation string
    """
    missing_pct = df[column].isna().sum() / len(df) * 100
    
    if missing_pct > 50:
        return "Consider dropping column (>50% missing)"
    elif missing_pct > 20:
        return "Use advanced imputation (KNN, iterative)"
    elif pd.api.types.is_numeric_dtype(df[column]):
        skew = df[column].skew()
        if abs(skew) > 1:
            return "Use median (skewed distribution)"
        else:
            return "Use mean (normal distribution)"
    else:
        return "Use mode or create 'missing' category"


def interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient strength.
    
    Args:
        r: Correlation coefficient (-1 to 1)
    
    Returns:
        Interpretation string
    """
    abs_r = abs(r)
    if abs_r < 0.3:
        return "weak"
    elif abs_r < 0.7:
        return "moderate"
    else:
        return "strong"


def interpret_distribution(skewness: float, kurtosis: float) -> str:
    """Interpret distribution shape from skewness and kurtosis.
    
    Args:
        skewness: Skewness value
        kurtosis: Kurtosis value
    
    Returns:
        Distribution interpretation
    """
    skew_interp = "symmetric" if abs(skewness) < 0.5 else ("right-skewed" if skewness > 0 else "left-skewed")
    kurt_interp = "normal" if abs(kurtosis) < 1 else ("heavy-tailed" if kurtosis > 0 else "light-tailed")
    return f"{skew_interp}, {kurt_interp}"


def interpret_vif(vif: float) -> str:
    """Interpret Variance Inflation Factor.
    
    Args:
        vif: VIF value
    
    Returns:
        Interpretation string
    """
    if vif < 5:
        return "low multicollinearity"
    elif vif < 10:
        return "moderate multicollinearity"
    else:
        return "high multicollinearity (consider removing)"
