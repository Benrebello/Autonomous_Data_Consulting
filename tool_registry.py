# tool_registry.py
"""Tool registry with metadata for dynamic tool management."""

from typing import Callable, Dict, Any, Optional
import pandas as pd


class ToolMetadata:
    """Metadata for a tool including default parameter generation.
    
    This class encapsulates all information about a tool:
    - The function itself
    - How to generate default parameters
    - Description and category for documentation
    """
    
    def __init__(
        self, 
        function: Callable,
        get_defaults: Callable[[pd.DataFrame], Dict[str, Any]],
        description: str = "",
        category: str = "general",
        requires_numeric: bool = False,
        requires_categorical: bool = False,
        min_rows: int = 0,
        min_numeric_cols: int = 0,
        min_categorical_cols: int = 0
    ):
        """Initialize tool metadata.
        
        Args:
            function: The actual tool function to execute
            get_defaults: Function that generates default parameters from a DataFrame
            description: Human-readable description of what the tool does
            category: Category for grouping (e.g., 'visualization', 'statistical_tests')
            requires_numeric: Whether tool requires numeric columns
            requires_categorical: Whether tool requires categorical columns
            min_rows: Minimum number of rows required
            min_numeric_cols: Minimum number of numeric columns required
            min_categorical_cols: Minimum number of categorical columns required
        """
        self.function = function
        self.get_defaults = get_defaults
        self.description = description
        self.category = category
        self.requires_numeric = requires_numeric
        self.requires_categorical = requires_categorical
        self.min_rows = min_rows
        self.min_numeric_cols = min_numeric_cols
        self.min_categorical_cols = min_categorical_cols
    
    def can_execute(self, df: pd.DataFrame) -> tuple[bool, Optional[str]]:
        """Check if tool can be executed on given DataFrame.
        
        Args:
            df: DataFrame to check
            
        Returns:
            Tuple of (can_execute: bool, reason: Optional[str])
        """
        if len(df) < self.min_rows:
            return False, f"Requires at least {self.min_rows} rows, found {len(df)}"
        
        # Check minimum numeric columns
        if self.min_numeric_cols > 0:
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) < self.min_numeric_cols:
                return False, f"Requires at least {self.min_numeric_cols} numeric columns, found {len(numeric_cols)}"
        
        # Check minimum categorical columns
        if self.min_categorical_cols > 0:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) < self.min_categorical_cols:
                return False, f"Requires at least {self.min_categorical_cols} categorical columns, found {len(cat_cols)}"
        
        # Legacy checks for backward compatibility
        if self.requires_numeric and self.min_numeric_cols == 0:
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) == 0:
                return False, "Requires at least one numeric column"
        
        if self.requires_categorical and self.min_categorical_cols == 0:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) == 0:
                return False, "Requires at least one categorical column"
        
        return True, None


def _get_numeric_cols(df: pd.DataFrame, n: int = 1):
    """Helper to get first n numeric columns."""
    cols = df.select_dtypes(include='number').columns.tolist()
    return cols[:n] if len(cols) >= n else cols


def _get_categorical_cols(df: pd.DataFrame, n: int = 1):
    """Helper to get first n categorical columns."""
    cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return cols[:n] if len(cols) >= n else cols


# Import tools from modular package
from tools.advanced_analytics import (
    forecast_time_series_arima,
    risk_assessment,
    sensitivity_analysis,
    monte_carlo_simulation,
    perform_causal_inference,
    perform_named_entity_recognition,
    text_summarization,
)
from sklearn.linear_model import LogisticRegression

from tools import (
    # Math Operations
    add_columns, subtract_columns, multiply_columns, divide_columns,
    apply_math_function, compute_numerical_derivative, compute_numerical_integral,
    # Financial Analytics
    calculate_npv, calculate_irr, calculate_volatility,
    black_scholes_call, black_scholes_put,
    # Advanced Math
    solve_linear_system, compute_eigenvalues_eigenvectors, linear_programming,
    # Geometry
    euclidean_distance, haversine_distance_df, polygon_area,
    # Statistical (additional)
    fit_normal_distribution, perform_manova,
)

from tools import (
    # Data Profiling
    descriptive_stats, get_data_types, check_duplicates,
    get_central_tendency, get_variability, get_ranges,
    get_value_counts, get_frequent_values,
    # Visualization
    plot_histogram, plot_boxplot, plot_scatter, plot_heatmap, plot_line_chart, plot_violin_plot, generate_chart,
    # Correlation
    correlation_matrix, correlation_tests, multicollinearity_detection,
    # Statistical Tests
    perform_t_test, distribution_tests,
    # Outlier Detection
    detect_outliers,
    # Machine Learning
    linear_regression, logistic_regression, random_forest_classifier, svm_classifier, knn_classifier,
    hyperparameter_tuning, feature_importance_analysis, model_evaluation_detailed, evaluate_model,
    pca_dimensionality, select_features, perform_multiple_regression,
    xgboost_classifier, lightgbm_classifier, model_comparison,
    # Data Cleaning
    clean_data,
    # Clustering
    run_kmeans_clustering, get_clusters_summary,
    # Time Series
    decompose_time_series, forecast_arima, add_time_features_from_seconds,
    get_temporal_patterns,
    # Feature Engineering
    create_polynomial_features, create_interaction_features, create_rolling_features, create_lag_features, create_binning,
    # Business Analytics
    rfm_analysis, calculate_growth_rate, ab_test_analysis, perform_abc_analysis, gradient_boosting_classifier,
    cohort_analysis, customer_lifetime_value,
    # Text Analysis
    sentiment_analysis, topic_modeling, generate_wordcloud,
    # Geospatial
    plot_geospatial_map,
    # Data Transformation
    sort_dataframe, group_and_aggregate, create_pivot_table,
    # File Operations
    read_odt_tables, export_to_excel, export_analysis_results,
)

# Tool Registry - Declarative tool definitions
TOOL_REGISTRY: Dict[str, ToolMetadata] = {
    # Data Profiling & Quality
    'descriptive_stats': ToolMetadata(
        function=descriptive_stats,
        get_defaults=lambda df: {'df': df},
        description="Generate descriptive statistics for all numeric columns",
        category="data_profiling"
    ),
    'pca_dimensionality': ToolMetadata(
        function=pca_dimensionality,
        get_defaults=lambda df: {
            'df': df,
            'n_components': 2
        },
        description="PCA dimensionality reduction",
        category="machine_learning",
        requires_numeric=True,
        min_rows=10
    ),
    'select_features': ToolMetadata(
        function=select_features,
        get_defaults=lambda df: {
            'df': df,
            'target_column': 'class' if 'class' in df.columns else (df.select_dtypes(include='number').columns[-1] if len(df.select_dtypes(include='number').columns)>0 else df.columns[-1]),
            'k': 10
        },
        description="Univariate feature selection (top-K)",
        category="machine_learning",
        requires_numeric=True,
        min_rows=20
    ),
    'perform_multiple_regression': ToolMetadata(
        function=perform_multiple_regression,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': df.select_dtypes(include='number').columns.tolist()[:-1] or df.columns.tolist()[:-1],
            'y_column': (df.select_dtypes(include='number').columns[-1] if len(df.select_dtypes(include='number').columns)>0 else df.columns[-1])
        },
        description="Multiple linear regression",
        category="machine_learning",
        requires_numeric=True,
        min_rows=20
    ),
    
    'get_data_types': ToolMetadata(
        function=get_data_types,
        get_defaults=lambda df: {'df': df},
        description="Return data types of each column",
        category="data_profiling"
    ),
    
    'get_central_tendency': ToolMetadata(
        function=get_central_tendency,
        get_defaults=lambda df: {'df': df},
        description="Calculate mean, median, and mode for numeric columns",
        category="data_profiling",
        requires_numeric=True
    ),
    
    'get_variability': ToolMetadata(
        function=get_variability,
        get_defaults=lambda df: {'df': df},
        description="Calculate standard deviation, variance, and range for numeric columns",
        category="data_profiling",
        requires_numeric=True
    ),
    
    'get_ranges': ToolMetadata(
        function=get_ranges,
        get_defaults=lambda df: {'df': df},
        description="Calculate minimum, maximum, and range for numeric columns",
        category="data_profiling",
        requires_numeric=True
    ),
    
    'get_value_counts': ToolMetadata(
        function=get_value_counts,
        get_defaults=lambda df: {
            'df': df,
            'column': df.columns[0]
        },
        description="Count unique values in a column",
        category="data_profiling"
    ),
    
    'get_frequent_values': ToolMetadata(
        function=get_frequent_values,
        get_defaults=lambda df: {
            'df': df,
            'column': df.columns[0],
            'top_n': 10
        },
        description="Get most and least frequent values in a column",
        category="data_profiling"
    ),
    
    'get_temporal_patterns': ToolMetadata(
        function=get_temporal_patterns,
        get_defaults=lambda df: {
            'df': df,
            'time_column': next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()), df.columns[0])
        },
        description="Identify temporal patterns and trends in time series data",
        category="time_series"
    ),
    
    'get_clusters_summary': ToolMetadata(
        function=get_clusters_summary,
        get_defaults=lambda df: {
            'df': df,
            'cluster_column': 'cluster' if 'cluster' in df.columns else df.columns[-1]
        },
        description="Summarize characteristics of clusters",
        category="clustering"
    ),
    
    'check_duplicates': ToolMetadata(
        function=check_duplicates,
        get_defaults=lambda df: {'df': df},
        description="Check for duplicate rows",
        category="data_quality"
    ),
    
    # Visualization
    'plot_histogram': ToolMetadata(
        function=plot_histogram,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0]
        },
        description="Generate histogram for a numeric column",
        category="visualization",
        requires_numeric=True
    ),
    
    'plot_boxplot': ToolMetadata(
        function=plot_boxplot,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0]
        },
        description="Generate boxplot to detect outliers",
        category="visualization",
        requires_numeric=True
    ),
    
    'plot_scatter': ToolMetadata(
        function=plot_scatter,
        get_defaults=lambda df: {
            'df': df,
            'x_column': _get_numeric_cols(df, 2)[0] if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[0],
            'y_column': _get_numeric_cols(df, 2)[1] if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        },
        description="Generate scatter plot between two variables",
        category="visualization",
        requires_numeric=True
    ),
    
    # Correlation & Statistical Tests
    'correlation_matrix': ToolMetadata(
        function=correlation_matrix,
        get_defaults=lambda df: {'df': df},
        description="Correlation matrix with p-values",
        category="correlation_analysis",
        requires_numeric=True,
        min_numeric_cols=2
    ),
    'correlation_tests': ToolMetadata(
        function=correlation_tests,
        get_defaults=lambda df: {
            'df': df,
            'column1': _get_numeric_cols(df, 2)[0] if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[0],
            'column2': _get_numeric_cols(df, 2)[1] if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[1] if len(df.columns) > 1 else df.columns[0]
        },
        description="Multiple correlation tests (Pearson, Spearman, Kendall)",
        category="correlation_analysis",
        requires_numeric=True,
        min_numeric_cols=2
    ),
    
    'perform_t_test': ToolMetadata(
        function=perform_t_test,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'group_column': _get_categorical_cols(df, 1)[0] if _get_categorical_cols(df, 1) else df.columns[0]
        },
        description="Perform independent t-test between two groups",
        category="statistical_tests",
        requires_numeric=True,
        requires_categorical=True
    ),
    
    # Outlier Detection
    'detect_outliers': ToolMetadata(
        function=detect_outliers,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'method': 'iqr'
        },
        description="Detect outliers using IQR or Z-score method",
        category="outlier_detection",
        requires_numeric=True
    ),
    
    # Machine Learning
    'linear_regression': ToolMetadata(
        function=linear_regression,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': _get_numeric_cols(df, 10)[:-1] if len(_get_numeric_cols(df, 10)) >= 2 else df.columns[:-1].tolist(),
            'y_column': _get_numeric_cols(df, 10)[-1] if len(_get_numeric_cols(df, 10)) >= 2 else df.columns[-1]
        },
        description="Fit linear regression model",
        category="machine_learning",
        requires_numeric=True,
        min_rows=30
    ),
    
    'random_forest_classifier': ToolMetadata(
        function=random_forest_classifier,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': [c for c in _get_numeric_cols(df, 10) if c != 'class'][:5],
            'y_column': 'class' if 'class' in df.columns else df.columns[-1]
        },
        description="Train random forest classifier",
        category="machine_learning",
        requires_numeric=True,
        min_rows=50
    ),
    
    # Data Cleaning
    'clean_data': ToolMetadata(
        function=clean_data,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'strategy': 'median'
        },
        description="Fill null values in a column",
        category="data_cleaning"
    ),
    
    # ==========================
    # Clustering
    # ==========================
    'run_kmeans_clustering': ToolMetadata(
        function=run_kmeans_clustering,
        get_defaults=lambda df: {
            'df': df,
            'columns': _get_numeric_cols(df, 2) if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[:2].tolist(),
            'n_clusters': 3
        },
        description="Perform K-Means clustering",
        category="clustering",
        requires_numeric=True,
        min_rows=20
    ),
    'get_clusters_summary': ToolMetadata(
        function=get_clusters_summary,
        get_defaults=lambda df: {'df': df, 'n_clusters': 3},
        description="Get K-Means cluster centers and labels",
        category="clustering",
        requires_numeric=True,
        min_rows=20
    ),

    # ==========================
    # Time Series
    # ==========================
    'decompose_time_series': ToolMetadata(
        function=decompose_time_series,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'period': 12
        },
        description="Decompose series into trend/seasonal/residual",
        category="time_series",
        requires_numeric=True,
        min_rows=30
    ),
    'forecast_arima': ToolMetadata(
        function=forecast_arima,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'order': (1,1,1),
            'steps': 10
        },
        description="ARIMA forecasting",
        category="time_series",
        requires_numeric=True,
        min_rows=30
    ),
    'add_time_features_from_seconds': ToolMetadata(
        function=add_time_features_from_seconds,
        get_defaults=lambda df: {
            'df': df,
            'time_column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'origin': '2000-01-01'
        },
        description="Derive datetime features from seconds column",
        category="time_series",
        requires_numeric=True
    ),

    # ==========================
    # Feature Engineering
    # ==========================
    'create_polynomial_features': ToolMetadata(
        function=create_polynomial_features,
        get_defaults=lambda df: {
            'df': df,
            'columns': _get_numeric_cols(df, 2) if len(_get_numeric_cols(df, 2)) >= 1 else df.columns[:1].tolist(),
            'degree': 2
        },
        description="Create polynomial features",
        category="feature_engineering",
        requires_numeric=True
    ),
    'create_interaction_features': ToolMetadata(
        function=create_interaction_features,
        get_defaults=lambda df: {
            'df': df,
            'columns': _get_numeric_cols(df, 2) if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[:2].tolist(),
        },
        description="Create pairwise interaction features",
        category="feature_engineering",
        requires_numeric=True
    ),
    'create_rolling_features': ToolMetadata(
        function=create_rolling_features,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'windows': [3,7,14]
        },
        description="Rolling statistics features",
        category="feature_engineering",
        requires_numeric=True
    ),
    'create_lag_features': ToolMetadata(
        function=create_lag_features,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'lags': [1,7,30]
        },
        description="Lag features for time series",
        category="feature_engineering",
        requires_numeric=True
    ),

    # ==========================
    # Business Analytics
    # ==========================
    'rfm_analysis': ToolMetadata(
        function=rfm_analysis,
        get_defaults=lambda df: {
            'df': df,
            'customer_col': df.columns[0],
            'date_col': df.columns[1] if len(df.columns) > 1 else df.columns[0],
            'value_col': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[-1],
            'reference_date': None
        },
        description="RFM segmentation",
        category="business_analytics"
    ),
    'calculate_growth_rate': ToolMetadata(
        function=calculate_growth_rate,
        get_defaults=lambda df: {
            'df': df,
            'value_column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[-1],
            'time_column': df.columns[0]
        },
        description="Calculate growth rate over time",
        category="business_analytics",
        requires_numeric=True
    ),
    'ab_test_analysis': ToolMetadata(
        function=ab_test_analysis,
        get_defaults=lambda df: {
            'df': df,
            'group_col': df.columns[0],
            'metric_col': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[-1]
        },
        description="A/B test with t-test and Cohen's d",
        category="business_analytics",
        requires_numeric=True
    ),

    # ==========================
    # Text & Geospatial
    # ==========================
    'sentiment_analysis': ToolMetadata(
        function=sentiment_analysis,
        get_defaults=lambda df: {'df': df, 'text_column': df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns)>0 else df.columns[0]},
        description="Compute sentiment polarity using TextBlob",
        category="text_analysis"
    ),
    'topic_modeling': ToolMetadata(
        function=topic_modeling,
        get_defaults=lambda df: {'df': df, 'text_column': df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns)>0 else df.columns[0], 'num_topics': 5},
        description="LDA topic modeling",
        category="text_analysis"
    ),
    'plot_geospatial_map': ToolMetadata(
        function=plot_geospatial_map,
        get_defaults=lambda df: {'df': df, 'lat_column': 'latitude', 'lon_column': 'longitude'},
        description="Scatter plot on lat/lon",
        category="geospatial"
    ),

    # ==========================
    # Data Transformation & File Ops
    # ==========================
    'sort_dataframe': ToolMetadata(
        function=sort_dataframe,
        get_defaults=lambda df: {'df': df, 'by': df.columns[0], 'ascending': True},
        description="Sort DataFrame",
        category="data_transformation"
    ),
    'group_and_aggregate': ToolMetadata(
        function=group_and_aggregate,
        get_defaults=lambda df: {
            'df': df,
            'group_by': [df.columns[0]],
            'agg_dict': {(_get_numeric_cols(df,1)[0] if _get_numeric_cols(df,1) and _get_numeric_cols(df,1)[0] != df.columns[0] else df.columns[-1]): 'mean'}
        },
        description="Group by and aggregate",
        category="data_transformation"
    ),
    'create_pivot_table': ToolMetadata(
        function=create_pivot_table,
        get_defaults=lambda df: {
            'df': df,
            'index': df.columns[0],
            'columns': df.columns[1] if len(df.columns) > 1 else df.columns[0],
            'values': (_get_numeric_cols(df,1)[0] if _get_numeric_cols(df,1) and _get_numeric_cols(df,1)[0] not in [df.columns[0], df.columns[1] if len(df.columns) > 1 else df.columns[0]] else df.columns[-1]),
            'aggfunc': 'mean'
        },
        description="Create pivot table",
        category="data_transformation"
    ),
    'read_odt_tables': ToolMetadata(
        function=read_odt_tables,
        get_defaults=lambda df: {},  # df not required
        description="Read ODT tables to DataFrames",
        category="file_operations"
    ),
    'export_to_excel': ToolMetadata(
        function=export_to_excel,
        get_defaults=lambda df: {'df': df, 'filename': 'export.xlsx', 'sheet_name': 'Data'},
        description="Export DataFrame to Excel (bytes)",
        category="file_operations"
    ),
    'export_analysis_results': ToolMetadata(
        function=export_analysis_results,
        get_defaults=lambda df: {'results': {'sample': df.head(5)} if df is not None else {}, 'filename': 'analysis_results.xlsx'},
        description="Export dict of results to Excel (bytes)",
        category="file_operations"
    ),
    
    # ==========================
    # Advanced Analytics
    # ==========================
    'forecast_time_series_arima': ToolMetadata(
        function=forecast_time_series_arima,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'periods': 10
        },
        description="ARIMA time series forecast",
        category="advanced_analytics",
        requires_numeric=True,
        min_rows=20
    ),
    'risk_assessment': ToolMetadata(
        function=risk_assessment,
        get_defaults=lambda df: {
            'df': df,
            'risk_factors': _get_numeric_cols(df, 3),
            'weights': None
        },
        description="Risk assessment with weighted factors",
        category="advanced_analytics",
        requires_numeric=True
    ),
    'sensitivity_analysis': ToolMetadata(
        function=sensitivity_analysis,
        get_defaults=lambda df: {
            'base_value': 100.0,
            'variable_changes': {'var1': [-0.1, 0, 0.1]},
            'impact_function': None
        },
        description="Sensitivity analysis",
        category="advanced_analytics"
    ),
    'monte_carlo_simulation': ToolMetadata(
        function=monte_carlo_simulation,
        get_defaults=lambda df: {
            'variables': {'x': {'mean': 0, 'std': 1}},
            'n_simulations': 1000,
            'output_function': None
        },
        description="Monte Carlo simulation",
        category="advanced_analytics"
    ),
    'perform_causal_inference': ToolMetadata(
        function=perform_causal_inference,
        get_defaults=lambda df: {
            'df': df,
            'treatment': df.columns[0],
            'outcome': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[-1],
            'controls': None
        },
        description="Causal inference with controls",
        category="advanced_analytics",
        requires_numeric=True,
        min_rows=30
    ),
    'perform_named_entity_recognition': ToolMetadata(
        function=perform_named_entity_recognition,
        get_defaults=lambda df: {
            'df': df,
            'text_column': df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns)>0 else df.columns[0]
        },
        description="Named entity recognition (requires spacy)",
        category="advanced_analytics"
    ),
    'text_summarization': ToolMetadata(
        function=text_summarization,
        get_defaults=lambda df: {
            'text': 'Sample text for summarization.',
            'max_sentences': 3
        },
        description="Extractive text summarization",
        category="advanced_analytics"
    ),
    'create_binning': ToolMetadata(
        function=create_binning,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'bins': 5,
            'strategy': 'quantile'
        },
        description="Create binned features",
        category="feature_engineering",
        requires_numeric=True
    ),
    'gradient_boosting_classifier': ToolMetadata(
        function=gradient_boosting_classifier,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': [c for c in _get_numeric_cols(df, 5) if c != 'class'],
            'y_column': 'class' if 'class' in df.columns else df.columns[-1],
            'n_estimators': 100
        },
        description="Gradient boosting classifier",
        category="machine_learning",
        requires_numeric=True,
        min_rows=50
    ),
    'perform_abc_analysis': ToolMetadata(
        function=perform_abc_analysis,
        get_defaults=lambda df: {
            'df': df,
            'value_column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[-1]
        },
        description="ABC analysis (Pareto)",
        category="business_analytics",
        requires_numeric=True
    ),
    
    # ==========================
    # Math Operations
    # ==========================
    'add_columns': ToolMetadata(
        function=add_columns,
        get_defaults=lambda df: {
            'df': df,
            'col1': _get_numeric_cols(df, 2)[0] if len(_get_numeric_cols(df, 2)) >= 1 else df.columns[0],
            'col2': _get_numeric_cols(df, 2)[1] if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[-1],
            'new_col': 'sum_result'
        },
        description="Add two columns element-wise",
        category="math_operations",
        requires_numeric=True
    ),
    'subtract_columns': ToolMetadata(
        function=subtract_columns,
        get_defaults=lambda df: {
            'df': df,
            'col1': _get_numeric_cols(df, 2)[0] if len(_get_numeric_cols(df, 2)) >= 1 else df.columns[0],
            'col2': _get_numeric_cols(df, 2)[1] if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[-1],
            'new_col': 'diff_result'
        },
        description="Subtract two columns element-wise",
        category="math_operations",
        requires_numeric=True
    ),
    'multiply_columns': ToolMetadata(
        function=multiply_columns,
        get_defaults=lambda df: {
            'df': df,
            'col1': _get_numeric_cols(df, 2)[0] if len(_get_numeric_cols(df, 2)) >= 1 else df.columns[0],
            'col2': _get_numeric_cols(df, 2)[1] if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[-1],
            'new_col': 'product_result'
        },
        description="Multiply two columns element-wise",
        category="math_operations",
        requires_numeric=True
    ),
    'divide_columns': ToolMetadata(
        function=divide_columns,
        get_defaults=lambda df: {
            'df': df,
            'col1': _get_numeric_cols(df, 2)[0] if len(_get_numeric_cols(df, 2)) >= 1 else df.columns[0],
            'col2': _get_numeric_cols(df, 2)[1] if len(_get_numeric_cols(df, 2)) >= 2 else df.columns[-1],
            'new_col': 'quotient_result'
        },
        description="Divide two columns element-wise",
        category="math_operations",
        requires_numeric=True
    ),
    'apply_math_function': ToolMetadata(
        function=apply_math_function,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'func_name': 'log',
            'new_col': None
        },
        description="Apply math function (log, exp, sqrt, sin, cos, tan, abs)",
        category="math_operations",
        requires_numeric=True
    ),
    'compute_numerical_derivative': ToolMetadata(
        function=compute_numerical_derivative,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'new_col': 'derivative',
            'dx': 1.0
        },
        description="Compute numerical derivative",
        category="math_operations",
        requires_numeric=True
    ),
    'compute_numerical_integral': ToolMetadata(
        function=compute_numerical_integral,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0],
            'new_col': 'integral'
        },
        description="Compute numerical integral (cumulative sum)",
        category="math_operations",
        requires_numeric=True
    ),
    
    # ==========================
    # Financial Analytics
    # ==========================
    'calculate_npv': ToolMetadata(
        function=calculate_npv,
        get_defaults=lambda df: {
            'rate': 0.1,
            'cashflows': [100, 200, 300, 400]
        },
        description="Calculate Net Present Value",
        category="financial_analytics"
    ),
    'calculate_irr': ToolMetadata(
        function=calculate_irr,
        get_defaults=lambda df: {
            'cashflows': [-1000, 300, 400, 500]
        },
        description="Calculate Internal Rate of Return",
        category="financial_analytics"
    ),
    'calculate_volatility': ToolMetadata(
        function=calculate_volatility,
        get_defaults=lambda df: {
            'returns': [0.01, -0.02, 0.03, -0.01, 0.02]
        },
        description="Calculate volatility (std dev of returns)",
        category="financial_analytics"
    ),
    'black_scholes_call': ToolMetadata(
        function=black_scholes_call,
        get_defaults=lambda df: {
            'S': 100.0,
            'K': 100.0,
            'T': 1.0,
            'r': 0.05,
            'sigma': 0.2
        },
        description="Black-Scholes call option pricing",
        category="financial_analytics"
    ),
    'black_scholes_put': ToolMetadata(
        function=black_scholes_put,
        get_defaults=lambda df: {
            'S': 100.0,
            'K': 100.0,
            'T': 1.0,
            'r': 0.05,
            'sigma': 0.2
        },
        description="Black-Scholes put option pricing",
        category="financial_analytics"
    ),
    
    # ==========================
    # Advanced Math
    # ==========================
    'solve_linear_system': ToolMetadata(
        function=solve_linear_system,
        get_defaults=lambda df: {
            'A': [[3, 2], [1, 2]],
            'b': [7, 4]
        },
        description="Solve linear system Ax = b",
        category="advanced_math"
    ),
    'compute_eigenvalues_eigenvectors': ToolMetadata(
        function=compute_eigenvalues_eigenvectors,
        get_defaults=lambda df: {
            'matrix': [[4, 2], [1, 3]]
        },
        description="Compute eigenvalues and eigenvectors",
        category="advanced_math"
    ),
    'linear_programming': ToolMetadata(
        function=linear_programming,
        get_defaults=lambda df: {
            'c': [1, 2],
            'A_ub': [[-1, 1], [1, 2]],
            'b_ub': [1, 4],
            'A_eq': None,
            'b_eq': None
        },
        description="Solve linear programming problem",
        category="advanced_math"
    ),
    
    # ==========================
    # Geometry
    # ==========================
    'euclidean_distance': ToolMetadata(
        function=euclidean_distance,
        get_defaults=lambda df: {
            'df': df,
            'x1': df.columns[0] if len(df.columns) > 0 else 'x1',
            'y1': df.columns[1] if len(df.columns) > 1 else 'y1',
            'x2': df.columns[2] if len(df.columns) > 2 else 'x2',
            'y2': df.columns[3] if len(df.columns) > 3 else 'y2',
            'new_col': 'euclidean_dist'
        },
        description="Calculate Euclidean distance between points",
        category="geometry"
    ),
    'haversine_distance_df': ToolMetadata(
        function=haversine_distance_df,
        get_defaults=lambda df: {
            'df': df,
            'lat1_col': 'latitude' if 'latitude' in df.columns else df.columns[0],
            'lon1_col': 'longitude' if 'longitude' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0],
            'lat2_col': 'latitude' if 'latitude' in df.columns else df.columns[0],
            'lon2_col': 'longitude' if 'longitude' in df.columns else df.columns[1] if len(df.columns) > 1 else df.columns[0],
            'new_col': 'haversine_dist_km'
        },
        description="Calculate Haversine distance (geographic)",
        category="geometry"
    ),
    'polygon_area': ToolMetadata(
        function=polygon_area,
        get_defaults=lambda df: {
            'points': [(0, 0), (1, 0), (1, 1), (0, 1)]
        },
        description="Calculate polygon area using Shoelace formula",
        category="geometry"
    ),
    
    # ==========================
    # Statistical Tests (Additional)
    # ==========================
    'fit_normal_distribution': ToolMetadata(
        function=fit_normal_distribution,
        get_defaults=lambda df: {
            'df': df,
            'column': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[0]
        },
        description="Fit normal distribution and return parameters",
        category="statistical_tests",
        requires_numeric=True
    ),
    'perform_manova': ToolMetadata(
        function=perform_manova,
        get_defaults=lambda df: {
            'df': df,
            'dependent_vars': _get_numeric_cols(df, 2),
            'independent_var': df.select_dtypes(include=['object', 'category']).columns[0] if len(df.select_dtypes(include=['object', 'category']).columns) > 0 else df.columns[0]
        },
        description="Multivariate Analysis of Variance",
        category="statistical_tests",
        requires_numeric=True,
        min_rows=30
    ),
    
    # ==========================
    # New Machine Learning Tools
    # ==========================
    'xgboost_classifier': ToolMetadata(
        function=xgboost_classifier,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': [c for c in _get_numeric_cols(df, 10) if c != 'class'][:5],
            'y_column': 'class' if 'class' in df.columns else df.columns[-1],
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'test_size': 0.2
        },
        description="XGBoost classifier (state-of-the-art gradient boosting)",
        category="machine_learning",
        requires_numeric=True,
        min_rows=50
    ),
    'lightgbm_classifier': ToolMetadata(
        function=lightgbm_classifier,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': [c for c in _get_numeric_cols(df, 10) if c != 'class'][:5],
            'y_column': 'class' if 'class' in df.columns else df.columns[-1],
            'n_estimators': 100,
            'max_depth': -1,
            'learning_rate': 0.1,
            'test_size': 0.2
        },
        description="LightGBM classifier (fast gradient boosting)",
        category="machine_learning",
        requires_numeric=True,
        min_rows=50
    ),
    'model_comparison': ToolMetadata(
        function=model_comparison,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': [c for c in _get_numeric_cols(df, 10) if c != 'class'][:5],
            'y_column': 'class' if 'class' in df.columns else df.columns[-1],
            'models': ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm'],
            'test_size': 0.2
        },
        description="Compare multiple ML models on same dataset",
        category="machine_learning",
        requires_numeric=True,
        min_rows=50
    ),
    
    # ==========================
    # New Business Analytics Tools
    # ==========================
    'cohort_analysis': ToolMetadata(
        function=cohort_analysis,
        get_defaults=lambda df: {
            'df': df,
            'customer_col': df.columns[0],
            'date_col': df.columns[1] if len(df.columns) > 1 else df.columns[0],
            'value_col': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else None,
            'period': 'M'
        },
        description="Cohort analysis for customer retention tracking",
        category="business_analytics",
        min_rows=30
    ),
    'customer_lifetime_value': ToolMetadata(
        function=customer_lifetime_value,
        get_defaults=lambda df: {
            'df': df,
            'customer_col': df.columns[0],
            'date_col': df.columns[1] if len(df.columns) > 1 else df.columns[0],
            'value_col': _get_numeric_cols(df, 1)[0] if _get_numeric_cols(df, 1) else df.columns[-1],
            'prediction_months': 12
        },
        description="Calculate Customer Lifetime Value (CLV)",
        category="business_analytics",
        requires_numeric=True,
        min_rows=30
    ),
    
    # ==========================
    # Additional Visualization
    # ==========================
    'plot_heatmap': ToolMetadata(
        function=plot_heatmap,
        get_defaults=lambda df: {
            'df': df,
            'columns': df.select_dtypes(include='number').columns.tolist()
        },
        description="Correlation heatmap",
        category="visualization",
        requires_numeric=True
    ),
    'plot_line_chart': ToolMetadata(
        function=plot_line_chart,
        get_defaults=lambda df: {
            'df': df,
            'x_column': df.columns[0],
            'y_column': df.select_dtypes(include='number').columns[0] if len(df.select_dtypes(include='number').columns)>0 else df.columns[-1]
        },
        description="Line chart for time series",
        category="visualization"
    ),
    'plot_violin_plot': ToolMetadata(
        function=plot_violin_plot,
        get_defaults=lambda df: {
            'df': df,
            'x_column': df.select_dtypes(include=['object','category']).columns[0] if len(df.select_dtypes(include=['object','category']).columns)>0 else df.columns[0],
            'y_column': df.select_dtypes(include='number').columns[0] if len(df.select_dtypes(include='number').columns)>0 else df.columns[-1]
        },
        description="Violin plot for distribution comparison",
        category="visualization"
    ),
    'generate_chart': ToolMetadata(
        function=generate_chart,
        get_defaults=lambda df: {
            'df': df,
            'chart_type': 'bar',
            'x_column': df.columns[0],
            'y_column': df.select_dtypes(include='number').columns[0] if len(df.select_dtypes(include='number').columns)>0 else None
        },
        description="Generic chart generator",
        category="visualization"
    ),

    # ==========================
    # Additional ML
    # ==========================
    'logistic_regression': ToolMetadata(
        function=logistic_regression,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': [c for c in df.select_dtypes(include='number').columns if c != 'class'][:5],
            'y_column': 'class' if 'class' in df.columns else df.select_dtypes(include='number').columns[-1] if len(df.select_dtypes(include='number').columns)>0 else df.columns[-1]
        },
        description="Fit logistic regression",
        category="machine_learning",
        requires_numeric=True,
        min_rows=30
    ),
    'svm_classifier': ToolMetadata(
        function=svm_classifier,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': df.select_dtypes(include='number').columns.tolist()[:5],
            'y_column': 'class' if 'class' in df.columns else df.columns[-1],
            'kernel': 'rbf'
        },
        description="Train SVM classifier",
        category="machine_learning",
        requires_numeric=True,
        min_rows=50
    ),
    'knn_classifier': ToolMetadata(
        function=knn_classifier,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': df.select_dtypes(include='number').columns.tolist()[:5],
            'y_column': 'class' if 'class' in df.columns else df.columns[-1],
            'n_neighbors': 5
        },
        description="Train KNN classifier",
        category="machine_learning",
        requires_numeric=True,
        min_rows=50
    ),
    'hyperparameter_tuning': ToolMetadata(
        function=hyperparameter_tuning,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': df.select_dtypes(include='number').columns.tolist()[:5],
            'y_column': 'class' if 'class' in df.columns else df.columns[-1],
            'model_type': 'random_forest'
        },
        description="GridSearchCV hyperparameter tuning",
        category="machine_learning",
        requires_numeric=True,
        min_rows=80
    ),
    'feature_importance_analysis': ToolMetadata(
        function=feature_importance_analysis,
        get_defaults=lambda df: {
            'df': df,
            'x_columns': df.select_dtypes(include='number').columns.tolist()[:5],
            'y_column': 'class' if 'class' in df.columns else df.columns[-1]
        },
        description="Random forest feature importance",
        category="machine_learning",
        requires_numeric=True,
        min_rows=50
    ),
    'model_evaluation_detailed': ToolMetadata(
        function=model_evaluation_detailed,
        get_defaults=lambda df: {
            'model': LogisticRegression(max_iter=200).fit(
                df[df.select_dtypes(include='number').columns.tolist()[:5]].fillna(0),
                df['class'].fillna(0) if 'class' in df.columns else df.iloc[:, -1].fillna(0)
            ),
            'X': df[df.select_dtypes(include='number').columns.tolist()[:5]],
            'y': df['class'] if 'class' in df.columns else df.iloc[:, -1]
        },
        description="Detailed metrics (accuracy, precision, recall, F1, ROC AUC)",
        category="machine_learning",
        requires_numeric=True
    ),
    'evaluate_model': ToolMetadata(
        function=evaluate_model,
        get_defaults=lambda df: {
            'model': None,  # caller should supply a model, default kept for compatibility
            'X': df[df.select_dtypes(include='number').columns.tolist()[:5]],
            'y': df['class'] if 'class' in df.columns else df.iloc[:, -1],
            'cv': 3
        },
        description="Cross-validation accuracy",
        category="machine_learning",
        requires_numeric=True
    ),

    # ==========================
    # Additional Statistical/Correlation
    # ==========================
    'plot_scatter': ToolMetadata(
        function=plot_scatter,
        get_defaults=lambda df: {
            'df': df,
            'x_column': df.select_dtypes(include='number').columns[0] if len(df.select_dtypes(include='number').columns)>0 else df.columns[0],
            'y_column': df.select_dtypes(include='number').columns[1] if len(df.select_dtypes(include='number').columns)>1 else df.columns[-1]
        },
        description="Scatter plot",
        category="visualization",
        requires_numeric=True,
        min_numeric_cols=2
    ),
    'multicollinearity_detection': ToolMetadata(
        function=multicollinearity_detection,
        get_defaults=lambda df: {'df': df, 'columns': df.select_dtypes(include='number').columns.tolist()},
        description="VIF multicollinearity detection",
        category="correlation_analysis",
        requires_numeric=True,
        min_numeric_cols=2
    ),
    'distribution_tests': ToolMetadata(
        function=distribution_tests,
        get_defaults=lambda df: {
            'df': df,
            'column': df.select_dtypes(include='number').columns[0] if len(df.select_dtypes(include='number').columns)>0 else df.columns[0]
        },
        description="Shapiro/Kolmogorov, skewness, kurtosis",
        category="statistical_tests",
        requires_numeric=True,
        min_rows=10
    ),
    'generate_wordcloud': ToolMetadata(
        function=generate_wordcloud,
        get_defaults=lambda df: {
            'df': df,
            'text_column': df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns)>0 else df.columns[0]
        },
        description="Generate wordcloud from text column",
        category="visualization"
    ),
}


def get_tool_function(tool_name: str) -> Optional[Callable]:
    """Get tool function by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool function or None if not found
    """
    metadata = TOOL_REGISTRY.get(tool_name)
    return metadata.function if metadata else None


def get_tool_defaults(tool_name: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Get default parameters for a tool given a DataFrame.
    
    Args:
        tool_name: Name of the tool
        df: DataFrame to generate defaults from
        
    Returns:
        Dictionary of default parameters or None if tool not found
    """
    metadata = TOOL_REGISTRY.get(tool_name)
    if metadata:
        try:
            return metadata.get_defaults(df)
        except Exception:
            return None
    return None


def get_tool_validation_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get validation requirements for a tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Dictionary with validation requirements or None if tool not found
    """
    metadata = TOOL_REGISTRY.get(tool_name)
    if metadata:
        return {
            'requires_numeric': metadata.requires_numeric,
            'requires_categorical': metadata.requires_categorical,
            'min_rows': metadata.min_rows,
            'min_numeric_cols': metadata.min_numeric_cols,
            'min_categorical_cols': metadata.min_categorical_cols,
            'category': metadata.category,
            'description': metadata.description
        }
    return None


def get_tools_by_category(category: str) -> list[str]:
    """Get tools filtered by category.
    
    Args:
        category: Category name
        
    Returns:
        List of tool names in that category
    """
    return [
        name for name, metadata in TOOL_REGISTRY.items()
        if metadata.category == category
    ]


def get_tool_categories() -> list[str]:
    """Get list of all tool categories.
    
    Returns:
        Sorted list of unique categories
    """
    categories = {metadata.category for metadata in TOOL_REGISTRY.values()}
    return sorted(categories)


def get_available_tools() -> list[str]:
    """Get list of all available tool names.
    
    Returns:
        List of all registered tool names
    """
    return list(TOOL_REGISTRY.keys())


def validate_tool_for_dataframe(tool_name: str, df: pd.DataFrame) -> tuple[bool, Optional[str]]:
    """Check if a tool can be executed on a given DataFrame.
    
    Args:
        tool_name: Name of the tool
        df: DataFrame to validate against
        
    Returns:
        Tuple of (can_execute: bool, reason: Optional[str])
    """
    metadata = TOOL_REGISTRY.get(tool_name)
    if not metadata:
        return False, f"Tool '{tool_name}' not found in registry"
    
    return metadata.can_execute(df)


def search_tools_by_keyword(keyword: str, df: Optional[pd.DataFrame] = None) -> list[str]:
    """Search tools by keyword in name or description.
    
    Args:
        keyword: Keyword to search for
        df: Optional DataFrame to validate tools against
        
    Returns:
        List of matching tool names
    """
    keyword_lower = keyword.lower()
    matches = []
    
    for tool_name, metadata in TOOL_REGISTRY.items():
        # Search in tool name
        if keyword_lower in tool_name.lower():
            matches.append(tool_name)
            continue
        
        # Search in description
        if keyword_lower in metadata.description.lower():
            matches.append(tool_name)
            continue
    
    # Filter by DataFrame compatibility if provided
    if df is not None:
        matches = [t for t in matches if validate_tool_for_dataframe(t, df)[0]]
    
    return matches


def find_best_tool_for_query(query: str, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """Find the best tool for a natural language query.
    
    Args:
        query: Natural language query
        df: Optional DataFrame to validate against
        
    Returns:
        Best matching tool name or None
    """
    import unicodedata
    
    def normalize_text(text: str) -> str:
        """Remove accents and normalize text for better matching."""
        text = text.lower()
        # Remove accents
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )
        return text
    
    query_lower = query.lower()
    query_normalized = normalize_text(query)
    
    # Keyword mappings for common queries (order matters - more specific first)
    keyword_mappings = {
        # Data description - SPECIFIC FIRST
        'tendência central': ['get_central_tendency'],
        'tendencia central': ['get_central_tendency'],
        'medidas de tendência': ['get_central_tendency'],
        'medidas de tendencia': ['get_central_tendency'],
        'média': ['get_central_tendency'],
        'media': ['get_central_tendency'],
        'mediana': ['get_central_tendency'],
        'moda': ['get_central_tendency'],
        'variância': ['get_variability'],
        'variancia': ['get_variability'],
        'variabilidade': ['get_variability'],
        'desvio padrão': ['get_variability'],
        'desvio padrao': ['get_variability'],
        'desvio': ['get_variability'],
        'intervalo': ['get_ranges'],
        'mínimo': ['get_ranges'],
        'minimo': ['get_ranges'],
        'máximo': ['get_ranges'],
        'maximo': ['get_ranges'],
        'tipo': ['get_data_types'],
        
        # Patterns and trends - SPECIFIC PATTERNS FIRST
        'padrões temporais': ['get_temporal_patterns'],
        'padroes temporais': ['get_temporal_patterns'],
        'temporal': ['get_temporal_patterns'],
        'temporais': ['get_temporal_patterns'],
        'sazonal': ['get_temporal_patterns'],
        'sazonalidade': ['get_temporal_patterns'],
        'frequente': ['get_frequent_values'],
        'mais frequente': ['get_frequent_values'],
        'menos frequente': ['get_frequent_values'],
        
        # Clustering - SPECIFIC FIRST
        'faça clustering dos dados': ['run_kmeans_clustering'],
        'faca clustering dos dados': ['run_kmeans_clustering'],
        'fazer clustering dos dados': ['run_kmeans_clustering'],
        'faça clustering': ['run_kmeans_clustering'],
        'faca clustering': ['run_kmeans_clustering'],
        'fazer clustering': ['run_kmeans_clustering'],
        'clustering': ['run_kmeans_clustering'],
        'clusterização': ['run_kmeans_clustering'],
        'clusterizacao': ['run_kmeans_clustering'],
        'agrupamento': ['run_kmeans_clustering'],
        'agrupamentos': ['run_kmeans_clustering'],
        'cluster': ['run_kmeans_clustering', 'get_clusters_summary'],
        
        # Anomalies
        'outlier': ['detect_outliers'],
        'atípico': ['detect_outliers'],
        'atipico': ['detect_outliers'],
        'anomalia': ['detect_outliers'],
        'valores atípicos': ['detect_outliers'],
        'valores atipicos': ['detect_outliers'],
        
        # Relationships
        'correlação': ['correlation_matrix'],
        'correlacao': ['correlation_matrix'],
        'relação': ['get_variable_relations'],
        'relacao': ['get_variable_relations'],
        'influência': ['get_influential_variables'],
        'influencia': ['get_influential_variables'],
        
        # Visualization
        'histograma': ['plot_histogram'],
        'boxplot': ['plot_boxplot'],
        'dispersão': ['plot_scatter'],
        'dispersao': ['plot_scatter'],
        'scatter': ['plot_scatter'],
        'heatmap': ['plot_heatmap'],
        'distribuição': ['plot_histogram'],
        'distribuicao': ['plot_histogram'],
        
        # Data quality
        'duplicata': ['check_duplicates'],
        'faltante': ['missing_data_analysis'],
        'qualidade': ['data_profiling'],
    }
    
    # Find matching tools - sort by keyword length (longest first) to match specific phrases first
    sorted_keywords = sorted(keyword_mappings.items(), key=lambda x: len(x[0]), reverse=True)
    
    for keyword, tools in sorted_keywords:
        # Try both original and normalized versions
        keyword_normalized = normalize_text(keyword)
        if keyword in query_lower or keyword_normalized in query_normalized:
            # Return first valid tool
            for tool in tools:
                if df is None:
                    return tool
                can_execute, _ = validate_tool_for_dataframe(tool, df)
                if can_execute:
                    return tool
    
    return None


def get_tools_for_dataframe(df: pd.DataFrame, category: Optional[str] = None) -> list[tuple[str, str]]:
    """Get all tools that can execute on a given DataFrame.
    
    Args:
        df: DataFrame to check
        category: Optional category filter
        
    Returns:
        List of tuples (tool_name, description) for valid tools
    """
    valid_tools = []
    
    tools_to_check = TOOL_REGISTRY.items()
    if category:
        tools_to_check = [(name, meta) for name, meta in tools_to_check if meta.category == category]
    
    for tool_name, metadata in tools_to_check:
        can_execute, _ = metadata.can_execute(df)
        if can_execute:
            valid_tools.append((tool_name, metadata.description))
    
    return valid_tools


def execute_tool(tool_name: str, df: pd.DataFrame, **kwargs) -> Any:
    """Execute a tool by name with automatic parameter handling.
    
    Args:
        tool_name: Name of the tool to execute
        df: DataFrame to operate on
        **kwargs: Additional parameters for the tool
        
    Returns:
        Result of tool execution
        
    Raises:
        ValueError: If tool not found or validation fails
    """
    metadata = TOOL_REGISTRY.get(tool_name)
    if not metadata:
        raise ValueError(f"Tool '{tool_name}' not found in registry")
    
    # Validate tool can execute
    can_execute, reason = metadata.can_execute(df)
    if not can_execute:
        raise ValueError(f"Cannot execute {tool_name}: {reason}")
    
    # Get defaults and merge with provided kwargs
    defaults = metadata.get_defaults(df)
    params = {**defaults, **kwargs}
    
    # Execute tool
    return metadata.function(**params)


def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """Get complete information about a tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Dictionary with complete tool information or None if not found
    """
    metadata = TOOL_REGISTRY.get(tool_name)
    if not metadata:
        return None
    
    return {
        'name': tool_name,
        'description': metadata.description,
        'category': metadata.category,
        'requires_numeric': metadata.requires_numeric,
        'requires_categorical': metadata.requires_categorical,
        'min_rows': metadata.min_rows,
        'min_numeric_cols': metadata.min_numeric_cols,
        'min_categorical_cols': metadata.min_categorical_cols,
        'function': metadata.function.__name__
    }


def get_tools_info_by_category() -> Dict[str, list[Dict[str, Any]]]:
    """Get all tools organized by category with complete information.
    
    Returns:
        Dictionary mapping category names to lists of tool info dicts
    """
    result = {}
    
    for tool_name, metadata in TOOL_REGISTRY.items():
        category = metadata.category
        if category not in result:
            result[category] = []
        
        result[category].append({
            'name': tool_name,
            'description': metadata.description,
            'requires_numeric': metadata.requires_numeric,
            'requires_categorical': metadata.requires_categorical,
            'min_rows': metadata.min_rows
        })
    
    return result


def recommend_tools_for_dataframe(df: pd.DataFrame, top_n: int = 10) -> list[Dict[str, Any]]:
    """Recommend most relevant tools for a DataFrame based on its characteristics.
    
    Args:
        df: DataFrame to analyze
        top_n: Number of recommendations to return
        
    Returns:
        List of recommended tools with scores and reasons
    """
    recommendations = []
    
    # Analyze DataFrame characteristics
    n_rows = len(df)
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    has_time = any('time' in c.lower() or 'date' in c.lower() for c in df.columns)
    has_class = 'class' in df.columns
    
    for tool_name, metadata in TOOL_REGISTRY.items():
        can_execute, _ = metadata.can_execute(df)
        if not can_execute:
            continue
        
        score = 0
        reasons = []
        
        # Score based on data characteristics
        if metadata.category == 'data_profiling':
            score += 10
            reasons.append("Essential for understanding data")
        
        if metadata.category == 'visualization' and len(num_cols) > 0:
            score += 8
            reasons.append("Good for exploring numeric data")
        
        if metadata.category == 'correlation_analysis' and len(num_cols) >= 2:
            score += 9
            reasons.append("Multiple numeric columns detected")
        
        if metadata.category == 'outlier_detection' and len(num_cols) > 0:
            score += 7
            reasons.append("Useful for data quality")
        
        if metadata.category == 'time_series' and has_time:
            score += 9
            reasons.append("Time column detected")
        
        if metadata.category == 'machine_learning' and has_class and len(num_cols) >= 2:
            score += 8
            reasons.append("Classification target detected")
        
        if metadata.category == 'clustering' and len(num_cols) >= 2 and n_rows >= 20:
            score += 7
            reasons.append("Sufficient data for clustering")
        
        if score > 0:
            recommendations.append({
                'tool': tool_name,
                'description': metadata.description,
                'category': metadata.category,
                'score': score,
                'reasons': reasons
            })
    
    # Sort by score and return top N
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:top_n]
