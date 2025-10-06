# tools/statistical_tests.py
"""Statistical testing utilities."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List


def perform_t_test(df: pd.DataFrame, column: str, group_column: str) -> Dict[str, Any]:
    """Perform independent t-test between two groups.
    
    Args:
        df: Input DataFrame
        column: Numeric column to test
        group_column: Column defining the two groups
        
    Returns:
        Dictionary with test results
    """
    groups = df[group_column].unique()
    if len(groups) != 2:
        return {"error": "T-test requires exactly two groups."}
    group1 = df[df[group_column] == groups[0]][column]
    group2 = df[df[group_column] == groups[1]][column]
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return {"t_statistic": t_stat, "p_value": p_value, "groups": list(groups)}


def perform_chi_square(df: pd.DataFrame, column1: str, column2: str) -> Dict[str, Any]:
    """Perform chi-square test of independence.
    
    Args:
        df: Input DataFrame
        column1: First categorical column
        column2: Second categorical column
        
    Returns:
        Dictionary with test results
    """
    contingency_table = pd.crosstab(df[column1], df[column2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return {"chi2_statistic": chi2, "p_value": p, "degrees_of_freedom": dof}


def perform_anova(df: pd.DataFrame, column: str, group_column: str) -> Dict[str, Any]:
    """Perform ANOVA test.
    
    Args:
        df: Input DataFrame
        column: Numeric column to test
        group_column: Column defining groups
        
    Returns:
        Dictionary with test results
    """
    groups = [group[column].values for name, group in df.groupby(group_column)]
    if len(groups) < 2:
        return {"error": "ANOVA requires at least two groups."}
    f_stat, p_value = stats.f_oneway(*groups)
    return {"f_statistic": f_stat, "p_value": p_value}


def perform_kruskal_wallis(df: pd.DataFrame, column: str, group_column: str) -> Dict[str, Any]:
    """Perform Kruskal-Wallis H-test (non-parametric alternative to ANOVA).
    
    Args:
        df: Input DataFrame
        column: Numeric column to test
        group_column: Column defining groups
        
    Returns:
        Dictionary with test results
    """
    groups = [group[column].values for name, group in df.groupby(group_column)]
    if len(groups) < 2:
        return {"error": "Kruskal-Wallis requires at least two groups."}
    h_stat, p_value = stats.kruskal(*groups)
    return {"h_statistic": h_stat, "p_value": p_value}


def perform_bayesian_inference(df: pd.DataFrame, column: str, 
                                prior_mean: float = 0, prior_std: float = 1) -> Dict[str, Any]:
    """Simple Bayesian inference for mean.
    
    Args:
        df: Input DataFrame
        column: Numeric column to analyze
        prior_mean: Prior mean
        prior_std: Prior standard deviation
        
    Returns:
        Dictionary with posterior statistics
    """
    data = df[column].dropna()
    if len(data) == 0:
        return {"error": "No data"}
    likelihood_std = data.std()
    posterior_mean = (prior_mean / prior_std**2 + data.mean() * len(data) / likelihood_std**2) / (1/prior_std**2 + len(data)/likelihood_std**2)
    posterior_std = np.sqrt(1 / (1/prior_std**2 + len(data)/likelihood_std**2))
    return {"posterior_mean": posterior_mean, "posterior_std": posterior_std}


def correlation_tests(df: pd.DataFrame, column1: str, column2: str) -> Dict[str, Any]:
    """Perform multiple correlation tests (Pearson, Spearman, Kendall).
    
    Args:
        df: Input DataFrame
        column1: First numeric column
        column2: Second numeric column
        
    Returns:
        Dictionary with correlation results from all three tests
    """
    data1 = df[column1].dropna()
    data2 = df[column2].dropna()
    
    # Align data (use common indices)
    common_idx = data1.index.intersection(data2.index)
    data1 = data1.loc[common_idx]
    data2 = data2.loc[common_idx]
    
    # Pearson correlation (linear relationship)
    pearson_r, pearson_p = stats.pearsonr(data1, data2)
    
    # Spearman correlation (monotonic relationship)
    spearman_r, spearman_p = stats.spearmanr(data1, data2)
    
    # Kendall correlation (robust to outliers)
    kendall_tau, kendall_p = stats.kendalltau(data1, data2)
    
    return {
        "pearson": {"correlation": pearson_r, "p_value": pearson_p},
        "spearman": {"correlation": spearman_r, "p_value": spearman_p},
        "kendall": {"correlation": kendall_tau, "p_value": kendall_p},
        "sample_size": len(data1),
        "interpretation": {
            "pearson": "Measures linear relationship",
            "spearman": "Measures monotonic relationship",
            "kendall": "Robust to outliers"
        }
    }


def distribution_tests(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Perform normality tests on a column.
    
    Args:
        df: Input DataFrame
        column: Numeric column to test
        
    Returns:
        Dictionary with normality test results
    """
    data = df[column].dropna()
    
    if len(data) < 3:
        return {"error": "Need at least 3 data points for normality tests"}
    
    # Shapiro-Wilk test (best for n < 5000)
    if len(data) < 5000:
        shapiro_stat, shapiro_p = stats.shapiro(data)
    else:
        shapiro_stat, shapiro_p = None, None
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    return {
        "shapiro_wilk": {
            "statistic": shapiro_stat,
            "p_value": shapiro_p,
            "is_normal": shapiro_p > 0.05 if shapiro_p else None
        } if shapiro_stat else None,
        "kolmogorov_smirnov": {
            "statistic": ks_stat,
            "p_value": ks_p,
            "is_normal": ks_p > 0.05
        },
        "skewness": skewness,
        "kurtosis": kurtosis,
        "interpretation": {
            "skewness": "symmetric" if abs(skewness) < 0.5 else ("right-skewed" if skewness > 0 else "left-skewed"),
            "kurtosis": "normal" if abs(kurtosis) < 0.5 else ("heavy-tailed" if kurtosis > 0 else "light-tailed")
        }
    }


def ab_test_analysis(df: pd.DataFrame, group_column: str, metric_column: str) -> Dict[str, Any]:
    """Perform A/B test analysis with effect size.
    
    Args:
        df: Input DataFrame
        group_column: Column defining A/B groups
        metric_column: Metric to compare
        
    Returns:
        Dictionary with A/B test results including effect size
    """
    groups = df[group_column].unique()
    if len(groups) != 2:
        return {"error": "A/B test requires exactly two groups"}
    
    group_a = df[df[group_column] == groups[0]][metric_column].dropna()
    group_b = df[df[group_column] == groups[1]][metric_column].dropna()
    
    # T-test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt(((len(group_a) - 1) * group_a.std()**2 + (len(group_b) - 1) * group_b.std()**2) / (len(group_a) + len(group_b) - 2))
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_std if pooled_std > 0 else 0
    
    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    return {
        "group_a": {
            "name": groups[0],
            "mean": group_a.mean(),
            "std": group_a.std(),
            "size": len(group_a)
        },
        "group_b": {
            "name": groups[1],
            "mean": group_b.mean(),
            "std": group_b.std(),
            "size": len(group_b)
        },
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": cohens_d,
        "effect_size": effect_size,
        "interpretation": f"{'Significant' if p_value < 0.05 else 'Not significant'} difference with {effect_size} effect size"
    }


def fit_normal_distribution(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Fit normal distribution to data and return parameters.
    
    Args:
        df: Input DataFrame
        column: Column to fit
    
    Returns:
        Dictionary with mu and std parameters
    """
    from scipy.stats import norm
    
    data = pd.to_numeric(df[column], errors='coerce').dropna()
    if len(data) < 2:
        return {'error': 'Insufficient data for distribution fitting'}
    
    mu, std = norm.fit(data)
    return {'mu': float(mu), 'std': float(std)}


def perform_manova(df: pd.DataFrame, dependent_vars: List[str], independent_var: str) -> Dict[str, Any]:
    """Perform Multivariate Analysis of Variance (MANOVA).
    
    Args:
        df: Input DataFrame
        dependent_vars: List of dependent variable columns
        independent_var: Independent variable (grouping) column
    
    Returns:
        Dictionary with MANOVA results
    """
    try:
        import statsmodels.api as sm
        from statsmodels.multivariate.manova import MANOVA
        
        formula = f'{" + ".join(dependent_vars)} ~ {independent_var}'
        maov = MANOVA.from_formula(formula, data=df)
        result_text = maov.mv_test().summary().as_text()
        return {'summary': result_text}
    except Exception as e:
        return {'error': f'MANOVA failed: {str(e)}'}


def perform_survival_analysis(df: pd.DataFrame, time_column: str, event_column: str) -> Dict[str, Any]:
    """Perform a simple Kaplan-Meier survival analysis if lifelines is available.

    Args:
        df: Input DataFrame
        time_column: Column indicating durations/times
        event_column: Column indicating event occurrence (1) or censoring (0)

    Returns:
        Dictionary with survival function sampled values or error if lifelines missing.
    """
    try:
        from lifelines import KaplanMeierFitter
    except Exception:
        return {"error": "lifelines not installed"}

    durations = pd.to_numeric(df[time_column], errors='coerce')
    events = pd.to_numeric(df[event_column], errors='coerce')
    mask = durations.notna() & events.notna()
    if mask.sum() == 0:
        return {"error": "no valid data for survival analysis"}
    kmf = KaplanMeierFitter()
    kmf.fit(durations[mask], event_observed=events[mask])
    surv = kmf.survival_function_.reset_index()
    # Return a light subset
    head = surv.head(10)
    return {
        "survival_function": head.to_dict(orient='list'),
        "n": int(mask.sum())
    }
