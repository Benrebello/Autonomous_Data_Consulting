# Tools Reference Guide

Complete reference for all 125 exported analysis tools available in the system (211 total functions including internal utilities).

## Table of Contents

- [Data Profiling (17)](#data-profiling)
- [Statistical Tests (14)](#statistical-tests)
- [Machine Learning (42)](#machine-learning)
- [Advanced Analytics (12)](#advanced-analytics)
- [Financial Analytics (6)](#financial-analytics)
- [Math Operations (8)](#math-operations)
- [Advanced Math (7)](#advanced-math)
- [Geometry (5)](#geometry)
- [Feature Engineering (6)](#feature-engineering)
- [Business Analytics (9)](#business-analytics)
- [Time Series (8)](#time-series)
- [Text Analysis (7)](#text-analysis)
- [Visualization (11)](#visualization)
- [Clustering (6)](#clustering)
- [Correlation Analysis (6)](#correlation-analysis)
- [Outlier Detection (5)](#outlier-detection)
- [Data Transformation (13)](#data-transformation)
- [Data Cleaning (7)](#data-cleaning)
- [File Operations (10)](#file-operations)
- [Geospatial (3)](#geospatial)
- [Helpers (9)](#helpers)

---

## Data Profiling

**Module**: `tools/data_profiling.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `descriptive_stats` | Generate descriptive statistics | `df` |
| `get_data_types` | Return column data types | `df` |
| `get_central_tendency` | Mean and median for numeric columns | `df` |
| `get_variability` | Std and variance | `df` |
| `get_ranges` | Min and max values | `df` |
| `calculate_min_max_per_variable` | Per-column min/max | `df` |
| `get_value_counts` | Value frequency counts | `df`, `column` |
| `get_frequent_values` | Top N frequent values | `df`, `column`, `top_n` |
| `check_duplicates` | Detect duplicate rows | `df` |
| `data_profiling` | Comprehensive profiling | `df` |
| `missing_data_analysis` | Missing data patterns | `df` |
| `cardinality_analysis` | Cardinality and encoding recommendations | `df` |
| `calculate_skewness_kurtosis` | Distribution shape metrics | `df`, `columns` |
| `detect_data_quality_issues` | Quality issues detection | `df` |
| `get_exploratory_analysis` | Complete EDA report | `df` |

---

## Statistical Tests

**Module**: `tools/statistical_tests.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `perform_t_test` | Independent t-test | `df`, `column`, `group_column` |
| `perform_chi_square` | Chi-square test of independence | `df`, `column1`, `column2` |
| `perform_anova` | One-way ANOVA | `df`, `column`, `group_column` |
| `perform_kruskal_wallis` | Kruskal-Wallis H-test | `df`, `column`, `group_column` |
| `perform_bayesian_inference` | Bayesian mean estimation | `df`, `column`, `prior_mean`, `prior_std` |
| `perform_survival_analysis` | Kaplan-Meier survival analysis | `df`, `time_column`, `event_column` |
| `fit_normal_distribution` | Fit normal distribution | `df`, `column` |
| `perform_manova` | Multivariate ANOVA | `df`, `dependent_vars`, `independent_var` |
| `correlation_tests` | Correlation with significance | `df`, `col1`, `col2` |
| `distribution_tests` | Normality tests | `df`, `column` |
| `ab_test_analysis` | A/B test with Cohen's d | `df`, `group_col`, `metric_col` |

---

## Machine Learning

**Module**: `tools/machine_learning.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `linear_regression` | Linear regression model | `df`, `x_columns`, `y_column` |
| `logistic_regression` | Logistic regression | `df`, `x_columns`, `y_column` |
| `random_forest_classifier` | Random forest classifier | `df`, `x_columns`, `y_column`, `n_estimators` |
| `gradient_boosting_classifier` | Gradient boosting | `df`, `x_columns`, `y_column`, `n_estimators` |
| `svm_classifier` | Support Vector Machine | `df`, `x_columns`, `y_column`, `kernel` |
| `knn_classifier` | K-Nearest Neighbors | `df`, `x_columns`, `y_column`, `n_neighbors` |
| `normalize_data` | Normalize features | `df`, `columns` |
| `impute_missing` | Impute missing values | `df`, `strategy` |
| `evaluate_model` | Cross-validation evaluation | `model`, `X`, `y`, `cv` |
| `hyperparameter_tuning` | GridSearchCV tuning | `df`, `x_columns`, `y_column`, `model_type` |
| `feature_importance_analysis` | Feature importance | `df`, `x_columns`, `y_column` |
| `model_evaluation_detailed` | Detailed metrics | `model`, `X`, `y` |
| `pca_dimensionality` | PCA dimensionality reduction | `df`, `n_components` |
| `select_features` | Feature selection | `df`, `target_column`, `k` |
| `perform_multiple_regression` | Multiple regression | `df`, `x_columns`, `y_column` |

---

## Advanced Analytics

**Module**: `tools/advanced_analytics.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `forecast_time_series_arima` | ARIMA forecasting | `df`, `column`, `periods` |
| `risk_assessment` | Weighted risk scoring | `df`, `risk_factors`, `weights` |
| `sensitivity_analysis` | Sensitivity analysis | `base_value`, `variable_changes`, `impact_function` |
| `monte_carlo_simulation` | Monte Carlo simulation | `variables`, `n_simulations`, `output_function` |
| `perform_causal_inference` | Causal inference with controls | `df`, `treatment`, `outcome`, `controls` |
| `perform_named_entity_recognition` | NER (requires spacy) | `df`, `text_column` |
| `text_summarization` | Extractive summarization | `text`, `max_sentences` |

---

## Financial Analytics

**Module**: `tools/financial_analytics.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `calculate_npv` | Net Present Value | `rate`, `cashflows` |
| `calculate_irr` | Internal Rate of Return | `cashflows` |
| `calculate_volatility` | Volatility calculation | `returns` |
| `black_scholes_call` | Call option pricing | `S`, `K`, `T`, `r`, `sigma` |
| `black_scholes_put` | Put option pricing | `S`, `K`, `T`, `r`, `sigma` |

---

## Math Operations

**Module**: `tools/math_operations.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `add_columns` | Add two columns | `df`, `col1`, `col2`, `new_col` |
| `subtract_columns` | Subtract columns | `df`, `col1`, `col2`, `new_col` |
| `multiply_columns` | Multiply columns | `df`, `col1`, `col2`, `new_col` |
| `divide_columns` | Divide columns | `df`, `col1`, `col2`, `new_col` |
| `apply_math_function` | Apply math function | `df`, `column`, `func_name`, `new_col` |
| `compute_numerical_derivative` | Numerical derivative | `df`, `column`, `new_col`, `dx` |
| `compute_numerical_integral` | Numerical integral | `df`, `column`, `new_col` |

**Supported math functions**: `log`, `exp`, `sqrt`, `sin`, `cos`, `tan`, `abs`

---

## Advanced Math

**Module**: `tools/advanced_math.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `solve_linear_system` | Solve Ax = b | `A`, `b` |
| `compute_eigenvalues_eigenvectors` | Eigenvalue decomposition | `matrix` |
| `linear_programming` | Linear optimization | `c`, `A_ub`, `b_ub`, `A_eq`, `b_eq` |

---

## Geometry

**Module**: `tools/geometry.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `euclidean_distance` | 2D Euclidean distance | `df`, `x1`, `y1`, `x2`, `y2`, `new_col` |
| `haversine_distance_df` | Geographic distance | `df`, `lat1_col`, `lon1_col`, `lat2_col`, `lon2_col`, `new_col` |
| `polygon_area` | Polygon area (Shoelace) | `points` |

---

## Feature Engineering

**Module**: `tools/feature_engineering.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `create_polynomial_features` | Polynomial features | `df`, `columns`, `degree` |
| `create_interaction_features` | Interaction features | `df`, `columns` |
| `create_rolling_features` | Rolling statistics | `df`, `column`, `windows` |
| `create_lag_features` | Lag features | `df`, `column`, `lags` |
| `create_binning` | Binned features | `df`, `column`, `bins`, `strategy` |

---

## Business Analytics

**Module**: `tools/business_analytics.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `rfm_analysis` | RFM segmentation | `df`, `customer_col`, `date_col`, `value_col` |
| `calculate_growth_rate` | Growth rate over time | `df`, `value_column`, `time_column` |
| `ab_test_analysis` | A/B test analysis | `df`, `group_col`, `metric_col` |
| `perform_abc_analysis` | ABC/Pareto analysis | `df`, `value_column` |

---

## Time Series

**Module**: `tools/time_series.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `decompose_time_series` | Seasonal decomposition | `df`, `column`, `period` |
| `forecast_arima` | ARIMA forecasting | `df`, `column`, `order`, `steps` |
| `add_time_features_from_seconds` | Extract time features | `df`, `time_column`, `origin` |
| `get_temporal_patterns` | Temporal correlation | `df`, `time_column`, `value_column` |

---

## Text Analysis

**Module**: `tools/text_analysis.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `sentiment_analysis` | Sentiment polarity | `df`, `text_column` |
| `topic_modeling` | LDA topic modeling | `df`, `text_column`, `num_topics` |
| `generate_wordcloud` | Generate wordcloud | `df`, `text_column` |

---

## Visualization

**Module**: `tools/visualization.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `plot_histogram` | Histogram | `df`, `column` |
| `plot_boxplot` | Boxplot | `df`, `column` |
| `plot_scatter` | Scatter plot | `df`, `x_column`, `y_column` |
| `plot_heatmap` | Correlation heatmap | `df`, `columns` |
| `plot_line_chart` | Line chart | `df`, `x_column`, `y_column` |
| `plot_violin_plot` | Violin plot | `df`, `x_column`, `y_column` |
| `generate_chart` | Generic chart | `df`, `chart_type`, `x_column`, `y_column` |

---

## Clustering

**Module**: `tools/clustering.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `run_kmeans_clustering` | K-means clustering | `df`, `columns`, `n_clusters` |
| `cluster_with_kmeans` | Alias for run_kmeans_clustering | `df`, `columns`, `n_clusters` |
| `get_clusters_summary` | Cluster centers summary | `df`, `n_clusters` |

---

## Correlation Analysis

**Module**: `tools/correlation_analysis.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `correlation_matrix` | Correlation with p-values | `df` |
| `get_variable_relations` | Pairwise relations | `df`, `x_column`, `y_column` |
| `get_influential_variables` | Correlation with target | `df`, `target_column` |
| `multicollinearity_detection` | VIF analysis | `df`, `columns` |

---

## Outlier Detection

**Module**: `tools/outlier_detection.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `detect_outliers` | Detect outliers | `df`, `column`, `method` |
| `detect_and_remove_outliers` | Detect and remove | `df`, `column`, `method`, `threshold` |
| `get_outliers_summary` | Outlier summary | `df`, `column` |

**Methods**: `iqr`, `zscore`

---

## Data Transformation

**Module**: `tools/data_transformation.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `sort_dataframe` | Sort DataFrame | `df`, `by`, `ascending` |
| `group_and_aggregate` | Group and aggregate | `df`, `group_by`, `agg_dict` |
| `create_pivot_table` | Create pivot table | `df`, `index`, `columns`, `values`, `aggfunc` |
| `join_datasets` | Join on common column | `df1`, `df2`, `on_column` |
| `join_datasets_on` | Join on different keys | `df1`, `df2`, `left_on`, `right_on`, `how` |
| `compare_datasets` | Compare two DataFrames | `df1`, `df2` |
| `normalize_dataframe_columns` | Normalize column names | `df` |
| `remove_duplicates` | Remove duplicate rows | `df`, `subset`, `keep` |

---

## Data Cleaning

**Module**: `tools/data_cleaning.py`

| Function | Description | Key Parameters | Returns |
|----------|-------------|----------------|---------|
| `clean_data` | Fill missing values | `df`, `column`, `strategy` | DataFrame |
| `fill_missing_with_median` | Fill with median | `df`, `columns` | DataFrame |
| `validate_and_correct_data_types` | Auto-correct types | `df` | Tuple (DataFrame, report dict) |
| `validate_and_clean_dataframe` | Comprehensive cleaning | `df`, `remove_duplicates_flag`, `fill_numeric_nulls` | Dict with 'dataframe' and 'report' |
| `smart_type_inference` | Intelligent type inference | `df` | DataFrame |
| `detect_data_quality_issues` | Quality issue detection | `df` | Dict with quality metrics |

---

## File Operations

**Module**: `tools/file_operations.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `read_odt_tables` | Extract tables from ODT | `uploaded_file` |
| `export_to_excel` | Export to Excel bytes | `df`, `filename`, `sheet_name` |
| `export_analysis_results` | Export results dict | `results`, `filename` |

---

## Geospatial

**Module**: `tools/geospatial.py`

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `plot_geospatial_map` | Geographic scatter plot | `df`, `lat_column`, `lon_column` |

---

## Helpers

**Module**: `tools/helpers.py` (Internal utilities)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `interpret_effect_size` | Interpret Cohen's d | `d` |
| `ab_test_recommendation` | A/B test recommendation | `p_value`, `cohens_d`, `mean_a`, `mean_b` |
| `classify_rfm_segment` | RFM segment classification | `r_score`, `f_score`, `m_score` |
| `get_imputation_recommendation` | Imputation strategy | `df`, `column` |
| `interpret_correlation` | Correlation strength | `r` |
| `interpret_distribution` | Distribution shape | `skewness`, `kurtosis` |
| `interpret_vif` | VIF interpretation | `vif` |

---

## Tool Registry

All 86 public tools are registered in `tool_registry.py` with:
- **Automatic defaults**: Parameter generation from DataFrame
- **Validation rules**: Minimum rows, required column types
- **Category tags**: For organized discovery
- **Descriptions**: Human-readable explanations

### Usage Example

```python
from tool_registry import get_available_tools, get_tool_defaults

# Get all available tools
tools = get_available_tools()
print(f"Available tools: {len(tools)}")

# Get defaults for a specific tool
defaults = get_tool_defaults('correlation_matrix', df)
print(defaults)  # {'df': <DataFrame>}
```

---

## Adding New Tools

To add a new tool:

1. **Create function** in appropriate module (e.g., `tools/my_module.py`)
2. **Export** in `tools/__init__.py`
3. **Register** in `tool_registry.py` with metadata
4. **Write tests** in `tests/test_my_module.py`
5. **Update docs** in this file

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
