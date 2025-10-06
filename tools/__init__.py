# tools/__init__.py
"""
Modular tools package for data analysis.

This package organizes data analysis tools into specialized modules:
- data_cleaning: Data cleaning and preparation
- data_profiling: Data profiling and EDA
- statistical_tests: Statistical hypothesis testing
- correlation_analysis: Correlation and multicollinearity
- outlier_detection: Outlier detection and removal
- visualization: Charts and plots
- machine_learning: ML models and evaluation
- clustering: Clustering algorithms and summaries
- time_series: Time series analysis and forecasting
- feature_engineering: Feature creation utilities
- business_analytics: Business-oriented analytics (RFM, growth)
- text_analysis: NLP utilities (sentiment, topics)
- geospatial: Geospatial plotting
- data_transformation: Sorting/grouping/pivot/normalization
- file_operations: File I/O helpers (ODT/Excel)

All functions are re-exported here for backward compatibility.
"""

# Data Cleaning
from .data_cleaning import (
    clean_data,
    remove_duplicates,
    fill_missing_with_median,
    validate_and_correct_data_types,
    validate_and_clean_dataframe,
    smart_type_inference
)

# Data Profiling
from .data_profiling import (
    descriptive_stats,
    get_data_types,
    get_central_tendency,
    get_variability,
    get_ranges,
    calculate_min_max_per_variable,
    get_value_counts,
    get_frequent_values,
    check_duplicates,
    data_profiling,
    missing_data_analysis,
    cardinality_analysis,
    calculate_skewness_kurtosis,
    detect_data_quality_issues,
    get_exploratory_analysis
)

# Statistical Tests
from .statistical_tests import (
    perform_t_test,
    perform_chi_square,
    perform_anova,
    perform_kruskal_wallis,
    perform_bayesian_inference,
    perform_survival_analysis,
    fit_normal_distribution,
    perform_manova,
    correlation_tests,
    distribution_tests,
    ab_test_analysis
)

# Correlation Analysis
from .correlation_analysis import (
    correlation_matrix,
    get_variable_relations,
    get_influential_variables,
    multicollinearity_detection
)

# Outlier Detection
from .outlier_detection import (
    detect_outliers,
    get_outliers_summary,
    detect_and_remove_outliers
)

# Visualization
from .visualization import (
    plot_histogram,
    plot_boxplot,
    plot_scatter,
    generate_chart,
    plot_heatmap,
    plot_line_chart,
    plot_violin_plot,
    plot_geospatial_map,
    generate_wordcloud
)

# Machine Learning
from .machine_learning import (
    linear_regression,
    logistic_regression,
    random_forest_classifier,
    svm_classifier,
    knn_classifier,
    evaluate_model,
    pca_dimensionality,
    select_features,
    hyperparameter_tuning,
    feature_importance_analysis,
    model_evaluation_detailed,
    gradient_boosting_classifier,
    perform_multiple_regression,
    xgboost_classifier,
    lightgbm_classifier,
    model_comparison,
    normalize_data,
    impute_missing
)

# Clustering
from .clustering import (
    run_kmeans_clustering,
    get_clusters_summary,
)
# Backward-compat alias expected by app.py tool_mapping
cluster_with_kmeans = run_kmeans_clustering

# Time Series
from .time_series import (
    decompose_time_series,
    forecast_arima,
    add_time_features_from_seconds,
    get_temporal_patterns,
)

# Feature Engineering
from .feature_engineering import (
    create_polynomial_features,
    create_interaction_features,
    create_rolling_features,
    create_lag_features,
    create_binning,
)

# Business Analytics
from .business_analytics import (
    rfm_analysis,
    calculate_growth_rate,
    ab_test_analysis,
    perform_abc_analysis,
    cohort_analysis,
    customer_lifetime_value
)

# Advanced Analytics
from .advanced_analytics import (
    forecast_time_series_arima,
    risk_assessment,
    sensitivity_analysis,
    monte_carlo_simulation,
    perform_causal_inference,
    perform_named_entity_recognition,
    text_summarization,
)

# Math Operations
from .math_operations import (
    add_columns,
    subtract_columns,
    multiply_columns,
    divide_columns,
    apply_math_function,
    compute_numerical_derivative,
    compute_numerical_integral,
)

# Financial Analytics
from .financial_analytics import (
    calculate_npv,
    calculate_irr,
    calculate_volatility,
    black_scholes_call,
    black_scholes_put,
)

# Advanced Math
from .advanced_math import (
    solve_linear_system,
    compute_eigenvalues_eigenvectors,
    linear_programming,
)

# Geometry
from .geometry import (
    euclidean_distance,
    haversine_distance_df,
    polygon_area,
)

# Helpers (internal utilities)
from .helpers import (
    interpret_effect_size,
    ab_test_recommendation,
    classify_rfm_segment,
    get_imputation_recommendation,
    interpret_correlation,
    interpret_distribution,
    interpret_vif,
)

# Text Analysis
from .text_analysis import (
    sentiment_analysis,
    topic_modeling,
    generate_wordcloud,
)

# Geospatial
from .geospatial import (
    plot_geospatial_map,
)

# Data Transformation
from .data_transformation import (
    sort_dataframe,
    group_and_aggregate,
    create_pivot_table,
    join_datasets,
    join_datasets_on,
    compare_datasets,
    normalize_dataframe_columns,
)

# File Operations
from .file_operations import (
    read_odt_tables,
    export_to_excel,
    export_analysis_results,
)

# Define __all__ for explicit exports
__all__ = [
    # Data Cleaning
    'clean_data',
    'remove_duplicates',
    'fill_missing_with_median',
    'validate_and_correct_data_types',
    'validate_and_clean_dataframe',
    'smart_type_inference',
    
    # Data Profiling
    'descriptive_stats',
    'get_data_types',
    'get_central_tendency',
    'get_variability',
    'get_ranges',
    'calculate_min_max_per_variable',
    'get_value_counts',
    'get_frequent_values',
    'check_duplicates',
    'data_profiling',
    'missing_data_analysis',
    'cardinality_analysis',
    'calculate_skewness_kurtosis',
    'detect_data_quality_issues',
    'get_exploratory_analysis',
    
    # Statistical Tests
    'perform_t_test',
    'perform_chi_square',
    'perform_anova',
    'perform_kruskal_wallis',
    'perform_bayesian_inference',
    'perform_survival_analysis',
    'fit_normal_distribution',
    'perform_manova',
    'correlation_tests',
    'distribution_tests',
    'ab_test_analysis',
    
    # Correlation Analysis
    'correlation_matrix',
    'get_variable_relations',
    'get_influential_variables',
    'multicollinearity_detection',
    
    # Outlier Detection
    'detect_outliers',
    'get_outliers_summary',
    'detect_and_remove_outliers',
    
    # Visualization
    'plot_histogram',
    'plot_boxplot',
    'plot_scatter',
    'generate_chart',
    'plot_heatmap',
    'plot_line_chart',
    'plot_violin_plot',
    'plot_geospatial_map',
    'generate_wordcloud',
    
    # Machine Learning
    'linear_regression',
    'logistic_regression',
    'random_forest_classifier',
    'gradient_boosting_classifier',
    'svm_classifier',
    'knn_classifier',
    'normalize_data',
    'impute_missing',
    'evaluate_model',
    'hyperparameter_tuning',
    'feature_importance_analysis',
    'model_evaluation_detailed',
    'pca_dimensionality',
    'select_features',
    'perform_multiple_regression',
    
    # Clustering
    'run_kmeans_clustering',
    'cluster_with_kmeans',
    'get_clusters_summary',
    
    # Time Series
    'decompose_time_series',
    'forecast_arima',
    'add_time_features_from_seconds',
    'get_temporal_patterns',
    
    # Feature Engineering
    'create_polynomial_features',
    'create_interaction_features',
    'create_rolling_features',
    'create_lag_features',
    'create_binning',
    
    # Business Analytics
    'rfm_analysis',
    'calculate_growth_rate',
    'perform_abc_analysis',
    'ab_test_analysis',
    
    # Advanced Analytics
    'forecast_time_series_arima',
    'risk_assessment',
    'sensitivity_analysis',
    'monte_carlo_simulation',
    'perform_causal_inference',
    'perform_named_entity_recognition',
    'text_summarization',
    
    # Math Operations
    'add_columns',
    'subtract_columns',
    'multiply_columns',
    'divide_columns',
    'apply_math_function',
    'compute_numerical_derivative',
    'compute_numerical_integral',
    
    # Financial Analytics
    'calculate_npv',
    'calculate_irr',
    'calculate_volatility',
    'black_scholes_call',
    'black_scholes_put',
    
    # Advanced Math
    'solve_linear_system',
    'compute_eigenvalues_eigenvectors',
    'linear_programming',
    
    # Geometry
    'euclidean_distance',
    'haversine_distance_df',
    'polygon_area',
    
    # Helpers
    'interpret_effect_size',
    'ab_test_recommendation',
    'classify_rfm_segment',
    'get_imputation_recommendation',
    'interpret_correlation',
    'interpret_distribution',
    'interpret_vif',
    
    # Text Analysis
    'sentiment_analysis',
    'topic_modeling',
    'generate_wordcloud',
    
    # Geospatial
    'plot_geospatial_map',
    
    # Data Transformation
    'sort_dataframe',
    'group_and_aggregate',
    'create_pivot_table',
    'join_datasets',
    'join_datasets_on',
    'compare_datasets',
    'normalize_dataframe_columns',
    
    # File Operations
    'read_odt_tables',
    'export_to_excel',
    'export_analysis_results',
]
