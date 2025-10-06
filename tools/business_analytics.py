# tools/business_analytics.py
"""Business analytics utilities (RFM, growth rate, A/B)."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def calculate_growth_rate(df: pd.DataFrame, value_column: str, time_column: str) -> pd.DataFrame:
    """Calculate growth rate over time.
    
    Args:
        df: Input DataFrame
        value_column: Column containing values
        time_column: Column with time ordering (sortable)
    
    Returns:
        DataFrame with a new `growth_rate` column (pct_change)
    """
    df_sorted = df.sort_values(time_column)
    df_sorted['growth_rate'] = df_sorted[value_column].pct_change()
    return df_sorted


def rfm_analysis(
    df: pd.DataFrame, 
    customer_col: str, 
    date_col: str, 
    value_col: str, 
    reference_date: Optional[str] = None
) -> pd.DataFrame:
    """Perform RFM (Recency, Frequency, Monetary) analysis.
    
    Args:
        df: Input DataFrame
        customer_col: Customer identifier
        date_col: Transaction date column
        value_col: Monetary value column
        reference_date: Optional reference date (string)
    
    Returns:
        DataFrame with R, F, M scores and segment labels
    """
    if reference_date is None:
        reference_ts = pd.to_datetime(df[date_col]).max()
    else:
        reference_ts = pd.to_datetime(reference_date)
    
    df_local = df.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])
    rfm = df_local.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (reference_ts - x.max()).days),
        Frequency=(customer_col, 'count'),
        Monetary=(value_col, 'sum')
    )
    
    # Score 1..5 (higher is better except recency)
    try:
        rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=False, duplicates='drop') + 1
        rfm['R_score'] = 6 - rfm['R_score']  # Invert: lower recency is better
    except Exception:
        rfm['R_score'] = 3
    try:
        rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=False, duplicates='drop') + 1
    except Exception:
        rfm['F_score'] = 3
    try:
        rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=False, duplicates='drop') + 1
    except Exception:
        rfm['M_score'] = 3
    rfm['RFM'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    
    def segment(row):
        try:
            r = int(row['R_score']) if pd.notna(row['R_score']) else 3
            f = int(row['F_score']) if pd.notna(row['F_score']) else 3
            m = int(row['M_score']) if pd.notna(row['M_score']) else 3
        except (ValueError, TypeError):
            return "Unknown"
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and f >= 3:
            return "Loyal Customers"
        if r <= 2 and f >= 4 and m >= 4:
            return "At Risk"
        return "Potential Loyalists"
    
    rfm['segment'] = rfm.apply(segment, axis=1)
    return rfm.reset_index()


def ab_test_analysis(df: pd.DataFrame, group_col: str, metric_col: str) -> Dict[str, Any]:
    """Analyze A/B test results with statistical significance and effect size.
    
    Args:
        df: Input DataFrame
        group_col: Group identifier (A/B)
        metric_col: Metric to compare
    
    Returns:
        Dictionary with t-test results and Cohen's d
    """
    from scipy import stats
    groups = df[group_col].unique()
    if len(groups) != 2:
        return {"error": "A/B test requires exactly two groups"}
    
    group_a = df[df[group_col] == groups[0]][metric_col].dropna()
    group_b = df[df[group_col] == groups[1]][metric_col].dropna()
    
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    pooled_std = np.sqrt(((len(group_a)-1)*group_a.std()**2 + (len(group_b)-1)*group_b.std()**2) / (len(group_a)+len(group_b)-2))
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_std if pooled_std > 0 else 0.0
    
    def effect_label(d):
        ad = abs(d)
        if ad < 0.2: return "negligible"
        if ad < 0.5: return "small"
        if ad < 0.8: return "medium"
        return "large"
    
    return {
        'group_a': {'name': groups[0], 'mean': group_a.mean(), 'std': group_a.std(), 'size': len(group_a)},
        'group_b': {'name': groups[1], 'mean': group_b.mean(), 'std': group_b.std(), 'size': len(group_b)},
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': effect_label(cohens_d)
    }


def perform_abc_analysis(df: pd.DataFrame, value_column: str) -> Dict[str, Any]:
    """Perform a simple ABC analysis based on cumulative percentage of value.

    A: top ~80% of cumulative value
    B: next ~15%
    C: remaining
    """
    if value_column not in df.columns:
        return {'error': f'{value_column} not in DataFrame'}
    value_col = value_column
    s = pd.to_numeric(df[value_col], errors='coerce').fillna(0).sort_values(ascending=False)
    total = s.sum()
    if total <= 0:
        return {'A': 0, 'B': 0, 'C': len(s), 'total_items': len(s), 'total_value': float(total)}
    cum_pct = s.cumsum() / total
    categories = pd.Series(index=s.index, dtype='object')
    categories[cum_pct <= 0.80] = 'A'
    categories[(cum_pct > 0.80) & (cum_pct <= 0.95)] = 'B'
    categories[cum_pct > 0.95] = 'C'
    counts = categories.value_counts().to_dict()
    return {
        'A': int(counts.get('A', 0)),
        'B': int(counts.get('B', 0)),
        'C': int(counts.get('C', 0)),
        'total_items': int(len(s)),
        'total_value': float(total)
    }


def cohort_analysis(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    value_col: Optional[str] = None,
    period: str = 'M'
) -> Dict[str, Any]:
    """Perform cohort analysis to track customer behavior over time.
    
    Args:
        df: Input DataFrame
        customer_col: Column with customer identifiers
        date_col: Column with transaction dates
        value_col: Optional column with monetary values (for revenue cohorts)
        period: Time period for cohorts ('M' for month, 'W' for week, 'Q' for quarter)
        
    Returns:
        Dictionary with cohort matrix, retention rates, and insights
    """
    df_local = df.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])
    
    # Determine cohort (first purchase period)
    df_local['cohort'] = df_local.groupby(customer_col)[date_col].transform('min')
    df_local['cohort_period'] = df_local['cohort'].dt.to_period(period)
    
    # Determine transaction period
    df_local['transaction_period'] = df_local[date_col].dt.to_period(period)
    
    # Calculate periods since cohort
    df_local['periods_since_cohort'] = (
        df_local['transaction_period'].astype('int64') - 
        df_local['cohort_period'].astype('int64')
    )
    
    # Cohort size (unique customers per cohort)
    cohort_sizes = df_local.groupby('cohort_period')[customer_col].nunique()
    
    # Retention matrix (count of active customers)
    retention_matrix = df_local.groupby(['cohort_period', 'periods_since_cohort'])[customer_col].nunique().unstack(fill_value=0)
    
    # Retention rate matrix (percentage)
    retention_rate = retention_matrix.divide(cohort_sizes, axis=0) * 100
    
    # Revenue cohort analysis (if value_col provided)
    revenue_analysis = None
    if value_col and value_col in df_local.columns:
        revenue_matrix = df_local.groupby(['cohort_period', 'periods_since_cohort'])[value_col].sum().unstack(fill_value=0)
        avg_revenue_per_customer = revenue_matrix.divide(retention_matrix.replace(0, np.nan))
        revenue_analysis = {
            'total_revenue_by_cohort': revenue_matrix.to_dict(),
            'avg_revenue_per_customer': avg_revenue_per_customer.to_dict(),
            'lifetime_value_by_cohort': revenue_matrix.sum(axis=1).to_dict()
        }
    
    # Calculate key metrics
    overall_retention = {}
    for period_num in retention_rate.columns:
        if period_num > 0:  # Skip period 0 (100% by definition)
            overall_retention[f'period_{period_num}'] = float(retention_rate[period_num].mean())
    
    # Identify best and worst cohorts
    if len(retention_rate) > 0 and len(retention_rate.columns) > 1:
        # Average retention across all periods (excluding period 0)
        avg_retention_by_cohort = retention_rate.iloc[:, 1:].mean(axis=1)
        best_cohort = avg_retention_by_cohort.idxmax()
        worst_cohort = avg_retention_by_cohort.idxmin()
    else:
        best_cohort = worst_cohort = None
    
    return {
        'cohort_sizes': cohort_sizes.to_dict(),
        'retention_matrix': retention_matrix.to_dict(),
        'retention_rate_matrix': retention_rate.to_dict(),
        'overall_retention_by_period': overall_retention,
        'best_cohort': str(best_cohort) if best_cohort else None,
        'worst_cohort': str(worst_cohort) if worst_cohort else None,
        'total_cohorts': len(cohort_sizes),
        'period_type': period,
        'revenue_analysis': revenue_analysis,
        'insights': {
            'avg_retention_period_1': float(retention_rate[1].mean()) if 1 in retention_rate.columns else None,
            'avg_retention_period_3': float(retention_rate[3].mean()) if 3 in retention_rate.columns else None,
            'avg_retention_period_6': float(retention_rate[6].mean()) if 6 in retention_rate.columns else None,
        }
    }


def customer_lifetime_value(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    value_col: str,
    prediction_months: int = 12
) -> Dict[str, Any]:
    """Calculate Customer Lifetime Value (CLV) using historical data.
    
    Args:
        df: Input DataFrame
        customer_col: Column with customer identifiers
        date_col: Column with transaction dates
        value_col: Column with monetary values
        prediction_months: Number of months to project CLV
        
    Returns:
        Dictionary with CLV metrics and customer segments
    """
    df_local = df.copy()
    df_local[date_col] = pd.to_datetime(df_local[date_col])
    
    # Calculate customer metrics
    customer_metrics = df_local.groupby(customer_col).agg({
        date_col: ['min', 'max', 'count'],
        value_col: ['sum', 'mean']
    })
    
    customer_metrics.columns = ['first_purchase', 'last_purchase', 'n_purchases', 'total_value', 'avg_value']
    
    # Calculate customer lifespan in days
    customer_metrics['lifespan_days'] = (
        customer_metrics['last_purchase'] - customer_metrics['first_purchase']
    ).dt.days
    
    # Average purchase frequency (purchases per month)
    customer_metrics['purchase_frequency'] = (
        customer_metrics['n_purchases'] / 
        (customer_metrics['lifespan_days'] / 30).replace(0, 1)
    )
    
    # Predicted CLV (simple model: avg_value * purchase_frequency * prediction_months)
    customer_metrics['predicted_clv'] = (
        customer_metrics['avg_value'] * 
        customer_metrics['purchase_frequency'] * 
        prediction_months
    )
    
    # Segment customers by CLV
    try:
        customer_metrics['clv_segment'] = pd.qcut(
            customer_metrics['predicted_clv'], 
            q=4, 
            labels=['Low', 'Medium', 'High', 'Very High'],
            duplicates='drop'
        )
    except Exception:
        customer_metrics['clv_segment'] = 'Medium'
    
    # Summary statistics
    clv_summary = {
        'total_customers': len(customer_metrics),
        'avg_clv': float(customer_metrics['predicted_clv'].mean()),
        'median_clv': float(customer_metrics['predicted_clv'].median()),
        'total_predicted_revenue': float(customer_metrics['predicted_clv'].sum()),
        'avg_purchase_frequency': float(customer_metrics['purchase_frequency'].mean()),
        'avg_customer_value': float(customer_metrics['avg_value'].mean()),
        'segment_distribution': customer_metrics['clv_segment'].value_counts().to_dict()
    }
    
    # Top customers by CLV
    top_customers = customer_metrics.nlargest(10, 'predicted_clv')[
        ['total_value', 'n_purchases', 'predicted_clv', 'clv_segment']
    ].to_dict('index')
    
    return {
        'summary': clv_summary,
        'top_customers': top_customers,
        'prediction_months': prediction_months,
        'customer_metrics': customer_metrics.to_dict('index')
    }
