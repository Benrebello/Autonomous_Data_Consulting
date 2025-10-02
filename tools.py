# tools.py
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from typing import Dict, Any, List, Optional
from odf.opendocument import load as odf_load
from odf.table import Table, TableRow, TableCell
from odf.text import P
import tempfile
from io import BytesIO
import re
import unicodedata
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Data Architect Tools
def join_datasets(df1: pd.DataFrame, df2: pd.DataFrame, on_column: str) -> pd.DataFrame:
    """Joins two dataframes based on a common column."""
    return pd.merge(df1, df2, on=on_column)

def join_datasets_on(df1: pd.DataFrame, df2: pd.DataFrame, left_on: str, right_on: str, how: str = 'inner') -> pd.DataFrame:
    """Join two DataFrames using different key column names.

    Args:
        df1: Left DataFrame.
        df2: Right DataFrame.
        left_on: Key column in left DataFrame.
        right_on: Key column in right DataFrame.
        how: Join type (default 'inner').

    Returns:
        pd.DataFrame: Joined DataFrame.
    """
    return pd.merge(df1, df2, left_on=left_on, right_on=right_on, how=how)

def clean_data(df: pd.DataFrame, column: str, strategy: str = 'mean') -> pd.DataFrame:
    """Fills null values in a column based on a strategy."""
    if strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'median':
        fill_value = df[column].median()
    else: # mode
        fill_value = df[column].mode()[0]
    # Avoid chained assignment with inplace=True to prevent FutureWarning in pandas 3.0+
    df[column] = df[column].fillna(fill_value)
    return df

# Technical Analyst Tools (EDA)
def descriptive_stats(df: pd.DataFrame) -> dict:
    """Generates descriptive statistics for all numeric columns."""
    stats = df.describe().to_dict()
    types = df.dtypes.to_dict()
    null_counts = df.isnull().sum().to_dict()
    return {
        'stats': stats,
        'types': types,
        'null_counts': null_counts,
        'shape': df.shape
    }

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """Detects outliers using IQR or (modified) Z-score.

    - iqr: classic Tukey fences (1.5 * IQR)
    - zscore: Modified Z-Score based on median and MAD (threshold 3.5)
    """
    df = df.copy()
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        # Ensure native Python bools (dtype object) for identity checks in tests
        df['outlier'] = pd.Series([bool(v) for v in mask.tolist()], dtype=object)
    elif method == 'zscore':
        # Modified Z-Score using median and MAD (more robust for small samples)
        median = df[column].median()
        abs_dev = (df[column] - median).abs()
        MAD = abs_dev.median()
        if MAD == 0:
            # Fallback to standard deviation if MAD is zero
            mean = df[column].mean()
            std = df[column].std(ddof=0) if df[column].std(ddof=0) != 0 else 1.0
            z = (df[column] - mean) / std
            df['z_score'] = z
            mask = z.abs() > 3
        else:
            mod_z = 0.6745 * (df[column] - median) / MAD
            df['z_score'] = mod_z
            mask = mod_z.abs() > 3.5
        df['outlier'] = pd.Series([bool(v) for v in mask.tolist()], dtype=object)
    return df

def correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates the correlation matrix with statistical significance and interpretation.
    
    Returns correlation matrix with p-values and interpretation guidelines.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return {
            'error': 'Need at least 2 numeric columns for correlation analysis',
            'available_columns': list(numeric_df.columns)
        }
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Calculate p-values for each correlation
    n = len(numeric_df)
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                            columns=corr_matrix.columns, 
                            index=corr_matrix.index)
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i != j:
                r = corr_matrix.iloc[i, j]
                # Calculate t-statistic and p-value
                t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2) if abs(r) < 1 else np.inf
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                p_values.iloc[i, j] = p_val
            else:
                p_values.iloc[i, j] = 0.0
    
    # Identify significant correlations
    significant_correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle to avoid duplicates
                r = corr_matrix.iloc[i, j]
                p = p_values.iloc[i, j]
                
                # Interpret strength
                abs_r = abs(r)
                if abs_r < 0.1:
                    strength = "negligible"
                elif abs_r < 0.3:
                    strength = "weak"
                elif abs_r < 0.5:
                    strength = "moderate"
                elif abs_r < 0.7:
                    strength = "strong"
                else:
                    strength = "very strong"
                
                direction = "positive" if r > 0 else "negative"
                significant = p < 0.05
                
                significant_correlations.append({
                    'variable1': col1,
                    'variable2': col2,
                    'correlation': float(r),
                    'p_value': float(p),
                    'significant': significant,
                    'strength': strength,
                    'direction': direction,
                    'interpretation': f"{strength} {direction} correlation (r={r:.3f}, p={p:.4f})"
                })
    
    # Sort by absolute correlation value
    significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'p_values': p_values.to_dict(),
        'significant_correlations': significant_correlations,
        'sample_size': n,
        'interpretation_guide': {
            'correlation_strength': {
                '0.0-0.1': 'negligible',
                '0.1-0.3': 'weak',
                '0.3-0.5': 'moderate',
                '0.5-0.7': 'strong',
                '0.7-1.0': 'very strong'
            },
            'significance': 'p < 0.05 indicates statistically significant correlation',
            'warning': 'Correlation does not imply causation. Always consider context and other factors.'
        }
    }

def get_exploratory_analysis(df: pd.DataFrame) -> str:
    """Generates an Exploratory Data Analysis (EDA) report."""
    desc = descriptive_stats(df)
    corr_result = correlation_matrix(df)
    
    # Format correlation results
    if 'error' in corr_result:
        corr_str = f"Correlation: {corr_result['error']}"
    else:
        corr_str = f"Correlation Analysis (n={corr_result['sample_size']}):\n"
        corr_str += "Top correlations:\n"
        for corr in corr_result['significant_correlations'][:5]:
            corr_str += f"  {corr['variable1']} vs {corr['variable2']}: {corr['interpretation']}\n"
    
    report = f"Shape: {desc['shape']}\nTypes: {desc['types']}\nStats: {desc['stats']}\n{corr_str}"
    return report

# Business Analyst Tools (Visualizations)
def plot_histogram(df: pd.DataFrame, column: str) -> str:
    """Generates histogram for a column and stores bytes in memory."""
    plt.figure()
    sns.histplot(data=df, x=column)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    st.session_state.charts.append({'bytes': buf.getvalue(), 'caption': f'Histograma: {column}'})
    return f"Histogram for {column} generated."

def plot_boxplot(df: pd.DataFrame, column: str) -> str:
    """Generates boxplot to detect outliers and stores bytes in memory."""
    plt.figure()
    sns.boxplot(data=df, y=column)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    st.session_state.charts.append({'bytes': buf.getvalue(), 'caption': f'Boxplot: {column}'})
    return f"Boxplot for {column} generated."

def plot_scatter(df: pd.DataFrame, x_column: str, y_column: str) -> str:
    """Generates scatter plot and stores bytes in memory."""
    plt.figure()
    sns.scatterplot(data=df, x=x_column, y=y_column)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    st.session_state.charts.append({'bytes': buf.getvalue(), 'caption': f'Scatter: {x_column} x {y_column}'})
    return f"Scatter plot between {x_column} and {y_column} generated."

def generate_chart(df: pd.DataFrame, chart_type: str, x_column: str, y_column: str = None) -> str:
    """Generates a chart and stores bytes in memory."""
    plt.figure()
    if chart_type == 'bar':
        sns.barplot(data=df, x=x_column, y=y_column)
    elif chart_type == 'hist':
        sns.histplot(data=df, x=x_column)
    elif chart_type == 'scatter':
        sns.scatterplot(data=df, x=x_column, y=y_column)
    elif chart_type == 'box':
        sns.boxplot(data=df, y=x_column)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    if 'charts' not in st.session_state:
        st.session_state.charts = []
    st.session_state.charts.append({'bytes': buf.getvalue(), 'caption': f"{chart_type} chart: {x_column}{' vs ' + y_column if y_column else ''}"})
    return "Chart generated successfully."

# Data Scientist Tools
def run_kmeans_clustering(df: pd.DataFrame, columns: list, n_clusters: int) -> pd.DataFrame:
    """Runs K-Means clustering."""
    data_to_cluster = df[columns].dropna()
    scaled_data = StandardScaler().fit_transform(data_to_cluster)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(scaled_data)
    return df

def read_odt_tables(uploaded_file) -> Dict[str, pd.DataFrame]:
    """Read an ODT document and extract tables as DataFrames.

    Args:
        uploaded_file: Streamlit UploadedFile or file-like with read() returning bytes.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of table names to DataFrames. If a table has no name,
        a default name like 'table_1' is used.
    """
    # Read bytes without consuming the uploaded_file for other uses
    data = uploaded_file.read()
    # odfpy expects a path, so write to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.odt') as tmp:
        tmp.write(data)
        tmp.flush()
        doc = odf_load(tmp.name)

    tables = doc.getElementsByType(Table)
    result: Dict[str, pd.DataFrame] = {}
    for idx, tbl in enumerate(tables, start=1):
        rows = []
        for row in tbl.getElementsByType(TableRow):
            cells_text = []
            for cell in row.getElementsByType(TableCell):
                # Concatenate all paragraphs within the cell
                paragraphs = cell.getElementsByType(P)
                text_content = "\n".join([str(p) if isinstance(p, str) else ''.join([n.data for n in p.childNodes if hasattr(n, 'data')]) for p in paragraphs])
                cells_text.append(text_content)
            # Only append non-empty rows
            if any(cells_text):
                rows.append(cells_text)
        if rows:
            # Normalize row lengths
            max_len = max(len(r) for r in rows)
            rows = [r + [None] * (max_len - len(r)) for r in rows]
            df = pd.DataFrame(rows)
            # Try to promote first row to header if it looks like column names
            if not df.empty:
                first_row = df.iloc[0].tolist()
                # Heuristic: non-numeric strings and mostly unique
                def is_number_like(x):
                    try:
                        float(str(x).replace(',', '.'))
                        return True
                    except Exception:
                        return False
                non_numeric = [v for v in first_row if isinstance(v, str) and not is_number_like(v)]
                unique_ratio = len(set(first_row)) / max(1, len(first_row))
                if len(non_numeric) >= max(1, int(0.6 * len(first_row))) and unique_ratio > 0.8:
                    df.columns = first_row
                    df = df.iloc[1:].reset_index(drop=True)
            name = tbl.getAttribute('name') or f'table_{idx}'
            result[name] = df
    return result

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to snake_case ascii without special chars.

    - Lowercases
    - Removes accents
    - Replaces spaces and non-alphanumeric with underscores
    - Collapses multiple underscores and trims leading/trailing underscores
    """
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = unicodedata.normalize('NFKD', text)
        text = text.encode('ascii', 'ignore').decode('ascii')
        text = text.lower()
        text = re.sub(r'[^a-z0-9]+', '_', text)
        text = re.sub(r'_+', '_', text).strip('_')
        return text or 'col'
    df = df.copy()
    df.columns = [normalize(c) for c in df.columns]
    return df

# Additional specific analysis tools

def get_data_types(df):
    """Return the data types of each column."""
    return df.dtypes.astype(str).to_dict()

def get_central_tendency(df):
    """Return central tendency measures: mean and median for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}
    if not numeric_df.empty:
        result['mean'] = numeric_df.mean().to_dict()
        result['median'] = numeric_df.median().to_dict()
    return result

def get_variability(df):
    """Return variability measures: std and variance for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}
    if not numeric_df.empty:
        result['std'] = numeric_df.std().to_dict()
        result['var'] = numeric_df.var().to_dict()
    return result

def get_ranges(df):
    """Return min and max for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}
    if not numeric_df.empty:
        result['min'] = numeric_df.min().to_dict()
        result['max'] = numeric_df.max().to_dict()
    return result

def calculate_min_max_per_variable(df):
    """Calculate min and max for each numeric variable (column)."""
    numeric_df = df.select_dtypes(include=[np.number])
    result = {}
    if not numeric_df.empty:
        for col in numeric_df.columns:
            result[col] = {
                'min': numeric_df[col].min(),
                'max': numeric_df[col].max()
            }
    return result

def get_value_counts(df, column):
    """Return value counts for a specific column."""
    return df[column].value_counts().to_dict()

def get_frequent_values(df, column, top_n=10):
    """Return the most frequent values in a column."""
    return df[column].value_counts().head(top_n).to_dict()

def get_temporal_patterns(df, time_column, value_column):
    """Simple temporal pattern: correlation between time and value."""
    if time_column in df.columns and value_column in df.columns:
        corr = df[[time_column, value_column]].corr().iloc[0, 1]
        return {'correlation_time_value': corr}
    return {}

def get_clusters_summary(df, n_clusters=3):
    """Perform K-means and return cluster centers summary."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {}
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(numeric_df)
    return {
        'cluster_centers': kmeans.cluster_centers_.tolist(),
        'labels': kmeans.labels_.tolist()
    }

def get_outliers_summary(df, column):
    """Summarize outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return {
        'outlier_count': len(outliers),
        'outlier_percentage': len(outliers) / len(df) * 100,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

def get_variable_relations(df, x_column, y_column):
    """Get relation between two variables: correlation if numeric, crosstab if categorical."""
    if x_column in df.columns and y_column in df.columns:
        if df[x_column].dtype in ['int64', 'float64'] and df[y_column].dtype in ['int64', 'float64']:
            corr = df[[x_column, y_column]].corr().iloc[0, 1]
            return {'correlation': corr}
        else:
            crosstab = pd.crosstab(df[x_column], df[y_column])
            return {'crosstab': crosstab.to_dict()}
    return {}

def get_influential_variables(df, target_column):
    """Get correlation of all numeric variables with a numeric target.

    Returns empty dict if target is missing or non-numeric.
    """
    if target_column not in df.columns:
        return {}
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if target_column not in numeric_df.columns:
        return {}
    correlations = numeric_df.corr()[target_column].to_dict()
    return correlations

# Advanced analysis tools

def perform_t_test(df, column, group_column):
    """Perform independent t-test between two groups."""
    groups = df[group_column].unique()
    if len(groups) != 2:
        return {"error": "T-test requires exactly two groups."}
    group1 = df[df[group_column] == groups[0]][column]
    group2 = df[df[group_column] == groups[1]][column]
    t_stat, p_value = stats.ttest_ind(group1, group2)
    return {"t_statistic": t_stat, "p_value": p_value, "groups": list(groups)}

def perform_chi_square(df, column1, column2):
    """Perform chi-square test of independence."""
    contingency_table = pd.crosstab(df[column1], df[column2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return {"chi2_statistic": chi2, "p_value": p, "degrees_of_freedom": dof}

def linear_regression(df, x_columns, y_column):
    """Fit linear regression model."""
    X = df[x_columns]
    y = df[y_column]
    model = LinearRegression()
    model.fit(X, y)
    return {
        "coefficients": dict(zip(x_columns, model.coef_)),
        "intercept": model.intercept_,
        "score": model.score(X, y)
    }

def logistic_regression(df, x_columns, y_column):
    """Fit logistic regression for binary classification."""
    X = df[x_columns]
    y = df[y_column]
    model = LogisticRegression()
    model.fit(X, y)
    return {
        "coefficients": dict(zip(x_columns, model.coef_[0])),
        "intercept": model.intercept_[0],
        "score": model.score(X, y)
    }

def random_forest_classifier(df, x_columns, y_column, n_estimators=100):
    """Train random forest classifier."""
    X = df[x_columns]
    y = df[y_column]
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    return {
        "feature_importances": dict(zip(x_columns, model.feature_importances_)),
        "score": model.score(X, y)
    }

def normalize_data(df, columns):
    """Normalize specified columns using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled

def impute_missing(df, strategy='mean'):
    """Impute missing values."""
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed

def pca_dimensionality(df, n_components=2):
    """Perform PCA for dimensionality reduction."""
    numeric_df = df.select_dtypes(include=[np.number])
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(numeric_df)
    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "principal_components": principal_components.tolist()
    }

def decompose_time_series(df, column, period=12):
    """Decompose time series into trend, seasonal, and residual components."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    series = df[column]
    decomposition = seasonal_decompose(series, model='additive', period=period)
    return {
        'trend': decomposition.trend.tolist(),
        'seasonal': decomposition.seasonal.tolist(),
        'residual': decomposition.resid.tolist()
    }

def compare_datasets(df1, df2):
    """Compare two datasets: shapes, columns, stats."""
    comparison = {
        "shape1": list(df1.shape),
        "shape2": list(df2.shape),
        "columns1": list(df1.columns),
        "columns2": list(df2.columns),
        "common_columns": list(set(df1.columns) & set(df2.columns)),
        "unique_to_df1": list(set(df1.columns) - set(df2.columns)),
        "unique_to_df2": list(set(df2.columns) - set(df1.columns))
    }
    # Basic stats for common numeric columns
    common_numeric = [c for c in comparison["common_columns"] if c in df1.select_dtypes(include=[np.number]).columns]
    if common_numeric:
        stats1 = df1[common_numeric].describe().to_dict()
        stats2 = df2[common_numeric].describe().to_dict()
        comparison["stats_comparison"] = {"df1": stats1, "df2": stats2}
    return comparison

# Additional advanced tools

def plot_heatmap(df, columns=None):
    """Plot correlation heatmap for specified columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def evaluate_model(model, X, y, cv=5):
    """Evaluate a model with cross-validation."""
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'cv_scores': scores.tolist()
    }

def forecast_arima(df, column, order=(1,1,1), steps=10):
    """Forecast time series using ARIMA."""
    series = df[column]
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return {
        'forecast': forecast.tolist(),
        'aic': model_fit.aic
    }

def perform_anova(df, column, group_column):
    """Perform ANOVA test."""
    groups = [group[column].values for name, group in df.groupby(group_column)]
    if len(groups) < 2:
        return {"error": "ANOVA requires at least two groups."}
    f_stat, p_value = stats.f_oneway(*groups)
    return {"f_statistic": f_stat, "p_value": p_value}

def check_duplicates(df):
    """Check for duplicate rows."""
    duplicates = df.duplicated().sum()
    return {"duplicate_rows": duplicates, "percentage": duplicates / len(df) * 100}

def select_features(df, target_column, k=10):
    """Select top k features using mutual information (numeric predictors only).

    Falls back gracefully if there are insufficient numeric predictors or the target is missing.
    """
    if target_column not in df.columns:
        return {}
    # Use only numeric predictors
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X_cols = [c for c in numeric_cols if c != target_column]
    if not X_cols:
        return {"selected_features": [], "scores": []}
    X = df[X_cols]
    y = df[target_column]
    # Ensure k does not exceed available columns
    k_eff = min(k, X.shape[1])
    selector = SelectKBest(mutual_info_regression, k=k_eff)
    selector.fit(X, y)
    selected_features = [X.columns[i] for i, keep in enumerate(selector.get_support()) if keep]
    scores = selector.scores_.tolist() if hasattr(selector, 'scores_') else []
    return {"selected_features": selected_features, "scores": scores}

def generate_wordcloud(df, text_column):
    """Generate wordcloud from text column."""
    try:
        from wordcloud import WordCloud
    except Exception:
        # Graceful fallback when optional dependency is missing
        return None
    text = ' '.join(df[text_column].dropna().astype(str))
    wc = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def plot_line_chart(df, x_column, y_column):
    """Plot line chart for time series."""
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column], df[y_column])
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'Line Chart: {y_column} over {x_column}')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# Final set of comprehensive analysis tools

def plot_violin_plot(df, x_column, y_column):
    """Plot violin plot for distribution comparison."""
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=df[x_column], y=df[y_column])
    plt.title(f'Violin Plot: {y_column} by {x_column}')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def perform_kruskal_wallis(df, column, group_column):
    """Perform Kruskal-Wallis H-test."""
    groups = [group[column].values for name, group in df.groupby(group_column)]
    if len(groups) < 2:
        return {"error": "Kruskal-Wallis requires at least two groups."}
    h_stat, p_value = stats.kruskal(*groups)
    return {"h_statistic": h_stat, "p_value": p_value}

def svm_classifier(df, x_columns, y_column, kernel='rbf'):
    """Train SVM classifier."""
    X = df[x_columns]
    y = df[y_column]
    model = SVC(kernel=kernel)
    model.fit(X, y)
    score = model.score(X, y)
    return {"score": score}

def knn_classifier(df, x_columns, y_column, n_neighbors=5):
    """Train KNN classifier."""
    X = df[x_columns]
    y = df[y_column]
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    score = model.score(X, y)
    return {"score": score}

def sentiment_analysis(df, text_column):
    """Perform sentiment analysis on text column.

    Lazy-import TextBlob to avoid module import error during test collection
    when sentiment analysis is not used.
    """
    try:
        from textblob import TextBlob
    except Exception:
        return {"error": "textblob is not installed"}
    sentiments = []
    for text in df[text_column].dropna().astype(str):
        blob = TextBlob(text)
        sentiments.append(blob.sentiment.polarity)
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    return {"average_sentiment": avg_sentiment, "sentiments": sentiments}

def plot_geospatial_map(df, lat_column, lon_column):
    """Plot simple geospatial map if lat/lon available."""
    if lat_column in df.columns and lon_column in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df[lon_column], df[lat_column], alpha=0.5)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geospatial Scatter Plot')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    return None

def perform_survival_analysis(df, time_column, event_column):
    """Simple survival analysis using Kaplan-Meier."""
    try:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter()
        kmf.fit(df[time_column], event_observed=df[event_column])
        return {"median_survival": kmf.median_survival_time_, "survival_function": kmf.survival_function_.to_dict()}
    except ImportError:
        return {"error": "lifelines not installed"}

def topic_modeling(df, text_column, num_topics=5):
    """Simple topic modeling using LDA with safe guard for empty vocabulary."""
    texts = df[text_column].dropna().astype(str).tolist()
    vectorizer = CountVectorizer(stop_words='english')
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        # empty vocabulary or similar error
        return {}
    if X.shape[1] == 0:
        return {}
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    topics = {}
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
        topics[f'topic_{topic_idx}'] = top_words
    return topics

def perform_bayesian_inference(df, column, prior_mean=0, prior_std=1):
    """Simple Bayesian inference for mean."""
    data = df[column].dropna()
    if len(data) == 0:
        return {"error": "No data"}
    likelihood_std = data.std()
    posterior_mean = (prior_mean / prior_std**2 + data.mean() * len(data) / likelihood_std**2) / (1/prior_std**2 + len(data)/likelihood_std**2)
    posterior_std = np.sqrt(1 / (1/prior_std**2 + len(data)/likelihood_std**2))
    return {"posterior_mean": posterior_mean, "posterior_std": posterior_std}


# Expanded calculation tools

def add_columns(df, col1, col2, new_col):
    """Add two columns element-wise."""
    df[new_col] = df[col1] + df[col2]
    return df

def subtract_columns(df, col1, col2, new_col):
    """Subtract col2 from col1."""
    df[new_col] = df[col1] - df[col2]
    return df

def multiply_columns(df, col1, col2, new_col):
    """Multiply two columns."""
    df[new_col] = df[col1] * df[col2]
    return df

def divide_columns(df, col1, col2, new_col):
    """Divide col1 by col2."""
    df[new_col] = df[col1] / df[col2]
    return df

def apply_math_function(df, column, func_name, new_col=None):
    """Apply a math function to a column."""
    func_map = {
        'log': np.log,
        'exp': np.exp,
        'sqrt': np.sqrt,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'abs': np.abs
    }
    if func_name in func_map:
        if new_col:
            df[new_col] = func_map[func_name](df[column])
        else:
            df[column] = func_map[func_name](df[column])
    return df

def compute_numerical_derivative(df, column, new_col, dx=1):
    """Compute numerical derivative using numpy.gradient."""
    df[new_col] = np.gradient(df[column].values, dx)
    return df

def compute_numerical_integral(df, column, new_col):
    """Compute numerical integral using cumulative sum."""
    values = df[column].values
    integral = np.cumsum(values) * (df.index[1] - df.index[0] if len(df) > 1 else 1)
    df[new_col] = integral
    return df

def solve_linear_system(A, b):
    """Solve Ax = b for x."""
    A = np.array(A)
    b = np.array(b)
    x = np.linalg.solve(A, b)
    return x.tolist()

def compute_eigenvalues_eigenvectors(matrix):
    """Compute eigenvalues and eigenvectors."""
    matrix = np.array(matrix)
    eigvals, eigvecs = np.linalg.eig(matrix)
    return {'eigenvalues': eigvals.tolist(), 'eigenvectors': eigvecs.tolist()}

def linear_programming(c, A_ub, b_ub, A_eq=None, b_eq=None):
    """Solve linear programming problem."""
    from scipy.optimize import linprog
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    return {'success': res.success, 'x': res.x.tolist() if res.x is not None else None, 'fun': res.fun}

def calculate_npv(rate, cashflows):
    """Calculate NPV."""
    return sum(cf / (1 + rate)**t for t, cf in enumerate(cashflows))

def calculate_irr(cashflows):
    """Calculate IRR."""
    from scipy.optimize import fsolve
    def npv_func(rate):
        return sum(cf / (1 + rate)**t for t, cf in enumerate(cashflows))
    irr = fsolve(npv_func, 0.1)[0]
    return irr

def calculate_volatility(returns):
    """Calculate volatility (std dev)."""
    return np.std(returns)

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call

def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    from scipy.stats import norm
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put

def fit_normal_distribution(df, column):
    """Fit normal distribution."""
    from scipy.stats import norm
    mu, std = norm.fit(df[column])
    return {'mu': mu, 'std': std}

def perform_manova(df, dependent_vars, independent_var):
    """Perform MANOVA."""
    import statsmodels.api as sm
    from statsmodels.multivariate.manova import MANOVA
    formula = f'{" + ".join(dependent_vars)} ~ {independent_var}'
    maov = MANOVA.from_formula(formula, data=df)
    return maov.mv_test().summary().as_text()

def euclidean_distance(df, x1, y1, x2, y2, new_col):
    """Euclidean distance between points."""
    df[new_col] = np.sqrt((df[x1] - df[x2])**2 + (df[y1] - df[y2])**2)
    return df

def haversine_distance_df(df, lat1_col, lon1_col, lat2_col, lon2_col, new_col):
    """Haversine distance."""
    R = 6371  # km
    dlat = np.radians(df[lat2_col] - df[lat1_col])
    dlon = np.radians(df[lon2_col] - df[lon1_col])
    a = np.sin(dlat/2)**2 + np.cos(np.radians(df[lat1_col])) * np.cos(np.radians(df[lat2_col])) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    df[new_col] = R * c
    return df

def polygon_area(points):
    """Shoelace formula for polygon area."""
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def validate_and_correct_data_types(df):
    """Validate and attempt to correct data types in DataFrame."""
    corrected_df = df.copy()
    report = {}
    for col in df.columns:
        original_dtype = df[col].dtype
        # Try numeric conversion
        try:
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if not numeric_series.isna().all() and numeric_series.dtype != original_dtype:
                corrected_df[col] = numeric_series
                report[col] = f"Converted to numeric from {original_dtype}"
                continue
        except:
            pass
        # Reset if not
        corrected_df[col] = df[col]
        # Try datetime conversion ONLY for string-like columns (avoid converting numeric epoch-like values)
        try:
            is_string_like = pd.api.types.is_string_dtype(df[col])
        except Exception:
            is_string_like = (df[col].dtype == 'object')

        if is_string_like:
            try:
                datetime_series = pd.to_datetime(df[col], errors='coerce')
                if not datetime_series.isna().all() and datetime_series.dtype != original_dtype:
                    corrected_df[col] = datetime_series
                    report[col] = f"Converted to datetime from {original_dtype}"
                    continue
            except:
                pass
        # Reset
        corrected_df[col] = df[col]
        # Report for object columns
        if df[col].dtype == 'object':
            non_na = df[col].dropna()
            if len(non_na) > 0:
                types = set(type(x).__name__ for x in non_na)
                if len(types) > 1:
                    report[col] = f"Mixed types detected: {types}"
                elif len(types) == 1 and types == {'str'}:
                    report[col] = "String column, no conversion applied"
                else:
                    report[col] = f"Other type: {types}"
            else:
                report[col] = "All values are NaN"
        else:
            report[col] = f"Kept as {original_dtype}"
    return corrected_df, report


# Additional tools for expanded data analysis, engineering, business, strategic analysis, organization, and cleaning

def sort_dataframe(df, by, ascending=True):
    """Sort DataFrame by specified columns."""
    return df.sort_values(by=by, ascending=ascending)

def group_and_aggregate(df, group_by, agg_dict):
    """Group by columns and aggregate."""
    return df.groupby(group_by).agg(agg_dict).reset_index()

def create_pivot_table(df, index, columns, values, aggfunc='mean'):
    """Create pivot table."""
    return df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc).reset_index()

def remove_duplicates(df, subset=None, keep='first'):
    """Remove duplicate rows."""
    return df.drop_duplicates(subset=subset, keep=keep)

def fill_missing_with_median(df, columns):
    """Fill missing values with median for specified columns."""
    df_filled = df.copy()
    for col in columns:
        if col in df.columns:
            median_val = df[col].median()
            # Avoid chained assignment with inplace=True to prevent FutureWarning in pandas 3.0+
            df_filled[col] = df_filled[col].fillna(median_val)
    return df_filled

def detect_and_remove_outliers(df, column, method='iqr', threshold=1.5):
    """Detect and remove outliers using IQR or Z-score."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        from scipy.stats import zscore
        z_scores = zscore(df[column])
        return df[abs(z_scores) < 3]
    return df

def calculate_skewness_kurtosis(df, columns):
    """Calculate skewness and kurtosis for numeric columns."""
    from scipy.stats import skew, kurtosis
    result = {}
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            result[col] = {
                'skewness': skew(df[col].dropna()),
                'kurtosis': kurtosis(df[col].dropna())
            }
    return result

def perform_multiple_regression(df, x_columns, y_column):
    """Perform multiple linear regression."""
    X = df[x_columns]
    y = df[y_column]
    model = LinearRegression()
    model.fit(X, y)
    return {
        'coefficients': dict(zip(x_columns, model.coef_)),
        'intercept': model.intercept_,
        'r_squared': model.score(X, y)
    }

def cluster_with_kmeans(df, columns, n_clusters=3):
    """Perform K-means clustering and add cluster labels."""
    data = df[columns].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df['cluster'] = kmeans.fit_predict(data)
    return df

def calculate_growth_rate(df, value_column, time_column):
    """Calculate growth rate over time."""
    df_sorted = df.sort_values(time_column)
    df_sorted['growth_rate'] = df_sorted[value_column].pct_change()
    return df_sorted

def perform_abc_analysis(df, value_column, category_column, a_threshold=0.8, b_threshold=0.95):
    """Perform ABC analysis (Pareto) on categories."""
    grouped = df.groupby(category_column)[value_column].sum().sort_values(ascending=False)
    total = grouped.sum()
    cumsum = grouped.cumsum() / total
    abc = []
    for cat, cum in cumsum.items():
        if cum <= a_threshold:
            abc.append('A')
        elif cum <= b_threshold:
            abc.append('B')
        else:
            abc.append('C')
    return pd.DataFrame({'category': grouped.index, 'value': grouped.values, 'cumulative_pct': cumsum.values, 'class': abc})

def forecast_time_series_arima(df, column, periods=10):
    """Forecast time series using ARIMA."""
    series = df[column]
    model = ARIMA(series, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return {'forecast': forecast.tolist(), 'aic': model_fit.aic}

def risk_assessment(df, risk_factors, weights=None):
    """Simple risk assessment by weighted sum."""
    if weights is None:
        weights = [1/len(risk_factors)] * len(risk_factors)
    risk_score = sum(df[factor] * weight for factor, weight in zip(risk_factors, weights))
    df = df.copy()
    df['risk_score'] = risk_score
    return df

def sensitivity_analysis(base_value, variable_changes, impact_function):
    """Perform sensitivity analysis."""
    results = {}
    for var, changes in variable_changes.items():
        for change in changes:
            new_value = base_value * (1 + change)
            impact = impact_function(new_value)
            results[f"{var}_{change}"] = impact
    return results

def monte_carlo_simulation(variables, n_simulations=1000, output_function=None):
    """Simple Monte Carlo simulation."""
    import numpy as np
    results = []
    for _ in range(n_simulations):
        sim_values = {}
        for var, dist in variables.items():
            if dist['type'] == 'normal':
                sim_values[var] = np.random.normal(dist['mean'], dist['std'])
            elif dist['type'] == 'uniform':
                sim_values[var] = np.random.uniform(dist['low'], dist['high'])
        if output_function:
            result = output_function(sim_values)
            results.append(result)
    return {'results': results, 'mean': np.mean(results), 'std': np.std(results)}

def perform_causal_inference(df, treatment, outcome, controls=None):
    """Perform simple causal inference using OLS regression."""
    import statsmodels.api as sm
    if controls is None:
        controls = []
    X = df[[treatment] + controls]
    X = sm.add_constant(X)
    y = df[outcome]
    model = sm.OLS(y, X).fit()
    return {
        'coefficients': model.params.to_dict(),
        'p_values': model.pvalues.to_dict(),
        'r_squared': model.rsquared,
        'summary': model.summary().as_text()
    }

def perform_named_entity_recognition(df, text_column):
    """Extract named entities from text column using spaCy."""
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
    except Exception as e:
        return {
            'error': 'spaCy model en_core_web_sm not available. Install spaCy and the model to enable NER.',
            'exception': str(e)
        }
    entities_list = []
    for idx, text in df[text_column].dropna().items():
        doc = nlp(text)
        for ent in doc.ents:
            entities_list.append({
                'row_index': idx,
                'entity': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
    return entities_list

def text_summarization(text, max_sentences=3):
    """Extractive text summarization using NLTK."""
    try:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        from nltk.probability import FreqDist
        import heapq
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sentences:
            return text
        stop_words = set(stopwords.words('english'))
        word_frequencies = FreqDist()
        for word in word_tokenize(text):
            if word.lower() not in stop_words and word.isalnum():
                word_frequencies[word.lower()] += 1
        if not word_frequencies:
            return ' '.join(sentences[:max_sentences])
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies:
            word_frequencies[word] = word_frequencies[word] / max_frequency
        sentence_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in word_frequencies:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]
        summary_sentences = heapq.nlargest(max_sentences, sentence_scores, key=sentence_scores.get)
        return ' '.join(summary_sentences)
    except Exception:
        # Fallback: return first N sentences if NLTK resources are missing
        parts = text.split('. ')
        return '. '.join(parts[:max_sentences]).strip()


def add_time_features_from_seconds(df, time_column, origin="2000-01-01"):
    """Create datetime-based features from a numeric seconds column without mutating semantics.

    This function assumes `time_column` contains elapsed seconds (e.g., since the first observation),
    not a real timestamp. It creates an artificial datetime by adding the seconds to a fixed `origin`,
    solely for feature extraction purposes.

    Generated columns:
    - time_datetime: synthetic datetime constructed from origin + seconds
    - time_date: date part of the synthetic datetime
    - time_hour: hour of day [0-23]
    - time_dayofweek: day of week [0=Monday, 6=Sunday]
    - time_month: month number [1-12]

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    time_column : str
        Column name with numeric seconds.
    origin : str or pandas.Timestamp, optional
        Origin date used to create a synthetic datetime. Default is '2000-01-01'.

    Returns
    -------
    pandas.DataFrame
        A copy of df with additional time feature columns.
    """
    df2 = df.copy()
    # Ensure numeric seconds
    seconds = pd.to_numeric(df2[time_column], errors='coerce')
    # Build synthetic datetime for feature extraction only
    origin_ts = pd.to_datetime(origin)
    dt = origin_ts + pd.to_timedelta(seconds.fillna(0), unit='s')
    df2[f"{time_column}_datetime"] = dt
    df2[f"{time_column}_date"] = dt.dt.date
    df2[f"{time_column}_hour"] = dt.dt.hour
    df2[f"{time_column}_dayofweek"] = dt.dt.dayofweek
    df2[f"{time_column}_month"] = dt.dt.month
    return df2


# ============================================================================
# EXTENDED TOOLS - Advanced Data Analysis
# ============================================================================

# 1. DATA QUALITY & PROFILING

def data_profiling(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive data profiling with quality metrics.
    
    Returns detailed information about:
    - Data types and memory usage
    - Missing values patterns
    - Cardinality
    - Basic statistics
    - Data quality score
    """
    profile = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': {}
    }
    
    for col in df.columns:
        col_profile = {
            'dtype': str(df[col].dtype),
            'missing_count': int(df[col].isna().sum()),
            'missing_pct': float(df[col].isna().sum() / len(df) * 100),
            'unique_count': int(df[col].nunique()),
            'cardinality': float(df[col].nunique() / len(df)),
        }
        
        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_profile.update({
                'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                'std': float(df[col].std()) if not df[col].isna().all() else None,
                'min': float(df[col].min()) if not df[col].isna().all() else None,
                'max': float(df[col].max()) if not df[col].isna().all() else None,
                'zeros_count': int((df[col] == 0).sum()),
                'zeros_pct': float((df[col] == 0).sum() / len(df) * 100),
            })
        
        # Categorical columns
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            top_values = df[col].value_counts().head(5).to_dict()
            col_profile.update({
                'top_values': {str(k): int(v) for k, v in top_values.items()},
                'mode': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None,
            })
        
        profile['columns'][col] = col_profile
    
    # Overall quality score (0-100)
    missing_score = 100 - (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
    profile['quality_score'] = round(missing_score, 2)
    
    return profile


def missing_data_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Detailed missing data analysis with patterns and recommendations."""
    missing_summary = {
        'total_missing': int(df.isna().sum().sum()),
        'total_cells': int(df.shape[0] * df.shape[1]),
        'missing_pct': float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        'columns_with_missing': {},
        'rows_with_missing': int(df.isna().any(axis=1).sum()),
        'complete_rows': int((~df.isna().any(axis=1)).sum()),
    }
    
    # Per column analysis
    for col in df.columns:
        if df[col].isna().sum() > 0:
            missing_summary['columns_with_missing'][col] = {
                'count': int(df[col].isna().sum()),
                'pct': float(df[col].isna().sum() / len(df) * 100),
                'recommendation': _get_imputation_recommendation(df, col)
            }
    
    # Missing patterns (which columns tend to be missing together)
    if len(missing_summary['columns_with_missing']) > 1:
        missing_matrix = df.isna().astype(int)
        pattern_counts = missing_matrix.value_counts().head(5)
        missing_summary['common_patterns'] = {
            str(idx): int(count) for idx, count in pattern_counts.items()
        }
    
    return missing_summary


def _get_imputation_recommendation(df: pd.DataFrame, column: str) -> str:
    """Recommend imputation strategy based on data characteristics."""
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


def cardinality_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze cardinality of columns for encoding decisions."""
    analysis = {}
    
    for col in df.columns:
        unique_count = df[col].nunique()
        cardinality = unique_count / len(df)
        
        # Classification
        if cardinality < 0.01:
            category = "Low cardinality (< 1%)"
            encoding_rec = "One-hot encoding"
        elif cardinality < 0.05:
            category = "Medium cardinality (1-5%)"
            encoding_rec = "Target encoding or frequency encoding"
        elif cardinality < 0.5:
            category = "High cardinality (5-50%)"
            encoding_rec = "Target encoding or hashing"
        else:
            category = "Very high cardinality (>50%)"
            encoding_rec = "Consider dropping or feature hashing"
        
        analysis[col] = {
            'unique_count': int(unique_count),
            'cardinality': float(cardinality),
            'category': category,
            'encoding_recommendation': encoding_rec
        }
    
    return analysis


def distribution_tests(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Test if data follows normal distribution."""
    data = df[column].dropna()
    
    if len(data) < 3:
        return {'error': 'Insufficient data for distribution tests'}
    
    results = {}
    
    # Shapiro-Wilk test (good for n < 5000)
    if len(data) <= 5000:
        stat, p_value = stats.shapiro(data)
        results['shapiro_wilk'] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': bool(p_value > 0.05)
        }
    
    # Kolmogorov-Smirnov test
    stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    results['kolmogorov_smirnov'] = {
        'statistic': float(stat),
        'p_value': float(p_value),
        'is_normal': bool(p_value > 0.05)
    }
    
    # D'Agostino-Pearson test
    if len(data) >= 8:
        stat, p_value = stats.normaltest(data)
        results['dagostino_pearson'] = {
            'statistic': float(stat),
            'p_value': float(p_value),
            'is_normal': bool(p_value > 0.05)
        }
    
    # Descriptive measures
    results['descriptive'] = {
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
        'interpretation': _interpret_distribution(stats.skew(data), stats.kurtosis(data))
    }
    
    return results


def _interpret_distribution(skewness: float, kurtosis: float) -> str:
    """Interpret distribution shape."""
    interp = []
    
    if abs(skewness) < 0.5:
        interp.append("approximately symmetric")
    elif skewness > 0:
        interp.append("right-skewed (positive skew)")
    else:
        interp.append("left-skewed (negative skew)")
    
    if abs(kurtosis) < 0.5:
        interp.append("normal tail behavior")
    elif kurtosis > 0:
        interp.append("heavy tails (leptokurtic)")
    else:
        interp.append("light tails (platykurtic)")
    
    return ", ".join(interp)


# 2. FEATURE ENGINEERING

def create_polynomial_features(df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
    """Create polynomial features up to specified degree."""
    from sklearn.preprocessing import PolynomialFeatures
    
    df_result = df.copy()
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    poly_features = poly.fit_transform(df[columns])
    feature_names = poly.get_feature_names_out(columns)
    
    # Add new features
    for i, name in enumerate(feature_names):
        if name not in columns:  # Skip original features
            df_result[name] = poly_features[:, i]
    
    return df_result


def create_interaction_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Create interaction features between specified columns."""
    df_result = df.copy()
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            interaction_name = f"{col1}_x_{col2}"
            df_result[interaction_name] = df[col1] * df[col2]
    
    return df_result


def create_binning(df: pd.DataFrame, column: str, bins: int = 5, strategy: str = 'quantile') -> pd.DataFrame:
    """Discretize continuous variable into bins."""
    df_result = df.copy()
    
    if strategy == 'quantile':
        df_result[f"{column}_binned"] = pd.qcut(df[column], q=bins, labels=False, duplicates='drop')
    elif strategy == 'uniform':
        df_result[f"{column}_binned"] = pd.cut(df[column], bins=bins, labels=False)
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")
    
    return df_result


def create_rolling_features(df: pd.DataFrame, column: str, windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
    """Create rolling statistics features for time series."""
    df_result = df.copy()
    
    for window in windows:
        df_result[f"{column}_rolling_mean_{window}"] = df[column].rolling(window=window).mean()
        df_result[f"{column}_rolling_std_{window}"] = df[column].rolling(window=window).std()
        df_result[f"{column}_rolling_min_{window}"] = df[column].rolling(window=window).min()
        df_result[f"{column}_rolling_max_{window}"] = df[column].rolling(window=window).max()
    
    return df_result


def create_lag_features(df: pd.DataFrame, column: str, lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
    """Create lag features for time series."""
    df_result = df.copy()
    
    for lag in lags:
        df_result[f"{column}_lag_{lag}"] = df[column].shift(lag)
    
    return df_result


# 3. ADVANCED STATISTICAL ANALYSIS

def correlation_tests(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    """Perform multiple correlation tests."""
    data1 = df[col1].dropna()
    data2 = df[col2].dropna()
    
    # Align data
    common_idx = data1.index.intersection(data2.index)
    data1 = data1.loc[common_idx]
    data2 = data2.loc[common_idx]
    
    if len(data1) < 3:
        return {'error': 'Insufficient data for correlation tests'}
    
    results = {}
    
    # Pearson (linear correlation)
    r, p = stats.pearsonr(data1, data2)
    results['pearson'] = {
        'correlation': float(r),
        'p_value': float(p),
        'significant': bool(p < 0.05),
        'interpretation': _interpret_correlation(r)
    }
    
    # Spearman (monotonic correlation)
    r, p = stats.spearmanr(data1, data2)
    results['spearman'] = {
        'correlation': float(r),
        'p_value': float(p),
        'significant': bool(p < 0.05),
        'interpretation': _interpret_correlation(r)
    }
    
    # Kendall (ordinal correlation)
    tau, p = stats.kendalltau(data1, data2)
    results['kendall'] = {
        'correlation': float(tau),
        'p_value': float(p),
        'significant': bool(p < 0.05),
        'interpretation': _interpret_correlation(tau)
    }
    
    return results


def _interpret_correlation(r: float) -> str:
    """Interpret correlation coefficient."""
    abs_r = abs(r)
    
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if r > 0 else "negative"
    
    return f"{strength} {direction} correlation"


def multicollinearity_detection(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """Calculate VIF (Variance Inflation Factor) to detect multicollinearity."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    X = df[columns].dropna()
    
    if X.shape[0] < X.shape[1] + 1:
        return {'error': 'Insufficient samples for VIF calculation'}
    
    vif_data = {}
    for i, col in enumerate(columns):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data[col] = {
                'vif': float(vif),
                'interpretation': _interpret_vif(vif)
            }
        except Exception as e:
            vif_data[col] = {'error': str(e)}
    
    return vif_data


def _interpret_vif(vif: float) -> str:
    """Interpret VIF value."""
    if vif < 5:
        return "Low multicollinearity"
    elif vif < 10:
        return "Moderate multicollinearity - consider investigation"
    else:
        return "High multicollinearity - consider removing variable"


# 4. ADVANCED MACHINE LEARNING

def gradient_boosting_classifier(df: pd.DataFrame, x_columns: List[str], y_column: str, 
                                n_estimators: int = 100, learning_rate: float = 0.1) -> Dict[str, Any]:
    """Train Gradient Boosting classifier."""
    from sklearn.ensemble import GradientBoostingClassifier
    
    X = df[x_columns].dropna()
    y = df[y_column].loc[X.index]
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )
    model.fit(X, y)
    
    return {
        'feature_importances': dict(zip(x_columns, model.feature_importances_.tolist())),
        'score': float(model.score(X, y)),
        'n_estimators': n_estimators,
        'learning_rate': learning_rate
    }


def hyperparameter_tuning(df: pd.DataFrame, x_columns: List[str], y_column: str, 
                         model_type: str = 'random_forest') -> Dict[str, Any]:
    """Perform grid search for hyperparameter tuning."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    
    X = df[x_columns].dropna()
    y = df[y_column].loc[X.index]
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    else:
        return {'error': f'Unsupported model type: {model_type}'}
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': float(grid_search.best_score_),
        'all_results': {
            str(params): float(score) 
            for params, score in zip(grid_search.cv_results_['params'], 
                                    grid_search.cv_results_['mean_test_score'])
        }
    }


def feature_importance_analysis(df: pd.DataFrame, x_columns: List[str], y_column: str) -> Dict[str, Any]:
    """Calculate feature importance using multiple methods."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    
    X = df[x_columns].dropna()
    y = df[y_column].loc[X.index]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Built-in feature importance
    builtin_importance = dict(zip(x_columns, model.feature_importances_.tolist()))
    
    # Permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_importance_dict = dict(zip(x_columns, perm_importance.importances_mean.tolist()))
    
    return {
        'builtin_importance': builtin_importance,
        'permutation_importance': perm_importance_dict,
        'top_5_features': sorted(builtin_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    }


def model_evaluation_detailed(model, X, y) -> Dict[str, Any]:
    """Detailed model evaluation with multiple metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
    
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
    }
    
    if y_pred_proba is not None and len(np.unique(y)) == 2:
        results['roc_auc'] = float(roc_auc_score(y, y_pred_proba))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    results['confusion_matrix'] = cm.tolist()
    
    # Classification report
    results['classification_report'] = classification_report(y, y_pred, output_dict=True, zero_division=0)
    
    return results


# 5. BUSINESS ANALYTICS

def rfm_analysis(df: pd.DataFrame, customer_col: str, date_col: str, 
                value_col: str, reference_date: Optional[str] = None) -> pd.DataFrame:
    """Perform RFM (Recency, Frequency, Monetary) analysis."""
    if reference_date is None:
        reference_date = df[date_col].max()
    else:
        reference_date = pd.to_datetime(reference_date)
    
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    rfm = df_copy.groupby(customer_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency
        customer_col: 'count',  # Frequency
        value_col: 'sum'  # Monetary
    })
    
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Create RFM scores (1-5)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Combined RFM score
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Segment classification
    rfm['Segment'] = rfm['RFM_Score'].apply(_classify_rfm_segment)
    
    return rfm.reset_index()


def _classify_rfm_segment(score: str) -> str:
    """Classify customer segment based on RFM score."""
    r, f, m = int(score[0]), int(score[1]), int(score[2])
    
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3 and m >= 3:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New Customers"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r <= 2 and f <= 2:
        return "Lost"
    else:
        return "Potential Loyalists"


def ab_test_analysis(df: pd.DataFrame, group_col: str, metric_col: str) -> Dict[str, Any]:
    """Analyze A/B test results with statistical significance."""
    groups = df[group_col].unique()
    
    if len(groups) != 2:
        return {'error': 'A/B test requires exactly 2 groups'}
    
    group_a = df[df[group_col] == groups[0]][metric_col].dropna()
    group_b = df[df[group_col] == groups[1]][metric_col].dropna()
    
    # T-test
    t_stat, p_value = stats.ttest_ind(group_a, group_b)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(group_a) - 1) * group_a.std()**2 + 
                          (len(group_b) - 1) * group_b.std()**2) / 
                         (len(group_a) + len(group_b) - 2))
    cohens_d = (group_a.mean() - group_b.mean()) / pooled_std
    
    # Confidence intervals
    ci_a = stats.t.interval(0.95, len(group_a)-1, loc=group_a.mean(), scale=stats.sem(group_a))
    ci_b = stats.t.interval(0.95, len(group_b)-1, loc=group_b.mean(), scale=stats.sem(group_b))
    
    return {
        'group_a': {
            'name': str(groups[0]),
            'mean': float(group_a.mean()),
            'std': float(group_a.std()),
            'count': int(len(group_a)),
            'ci_95': [float(ci_a[0]), float(ci_a[1])]
        },
        'group_b': {
            'name': str(groups[1]),
            'mean': float(group_b.mean()),
            'std': float(group_b.std()),
            'count': int(len(group_b)),
            'ci_95': [float(ci_b[0]), float(ci_b[1])]
        },
        'test_results': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'cohens_d': float(cohens_d),
            'effect_size': _interpret_effect_size(cohens_d)
        },
        'recommendation': _ab_test_recommendation(p_value, cohens_d, group_a.mean(), group_b.mean())
    }


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def _ab_test_recommendation(p_value: float, cohens_d: float, mean_a: float, mean_b: float) -> str:
    """Provide recommendation based on A/B test results."""
    if p_value >= 0.05:
        return "No significant difference detected. Consider running test longer or with larger sample."
    
    winner = "B" if mean_b > mean_a else "A"
    effect = _interpret_effect_size(cohens_d)
    
    return f"Group {winner} performs significantly better with {effect} effect size. Consider implementing."


# 6. DATA VALIDATION & AUTO-CORRECTION

def validate_and_clean_dataframe(df: pd.DataFrame, aggressive: bool = False) -> Dict[str, Any]:
    """Comprehensive data validation and automatic cleaning.
    
    Detects and fixes common issues:
    - String values in numeric columns (e.g., '97 min' -> 97.0)
    - Mixed types in columns
    - Invalid numeric formats
    - Date/time parsing issues
    - Encoding problems
    
    Args:
        df: DataFrame to validate and clean
        aggressive: If True, applies more aggressive cleaning (may lose data)
    
    Returns:
        Dict with cleaned DataFrame and detailed report
    """
    df_clean = df.copy()
    report = {
        'original_shape': df.shape,
        'issues_found': [],
        'corrections_applied': [],
        'columns_modified': {},
        'warnings': []
    }
    
    for col in df_clean.columns:
        col_issues = []
        col_corrections = []
        
        # Skip if all null
        if df_clean[col].isna().all():
            report['warnings'].append(f"{col}: All values are null")
            continue
        
        # Check for mixed types
        non_null_values = df_clean[col].dropna()
        if len(non_null_values) == 0:
            continue
            
        types_found = set(type(v).__name__ for v in non_null_values.head(100))
        
        if len(types_found) > 1:
            col_issues.append(f"Mixed types: {types_found}")
        
        # Try to detect and fix numeric columns with string contamination
        if df_clean[col].dtype == 'object':
            # Sample values to check if should be numeric
            sample = non_null_values.head(50)
            numeric_like = 0
            
            for val in sample:
                val_str = str(val).strip()
                # Check for patterns like "97 min", "1.5k", "$100", "50%"
                if re.search(r'[\d,\.]+', val_str):
                    numeric_like += 1
            
            # If most values look numeric, try to clean
            if numeric_like / len(sample) > 0.7:
                col_issues.append("Numeric values stored as strings")
                
                def clean_numeric_string(val):
                    """Extract numeric value from string."""
                    if pd.isna(val):
                        return np.nan
                    
                    val_str = str(val).strip()
                    
                    # Remove common units and symbols
                    val_str = re.sub(r'[^\d\.\-\+eE,]', '', val_str)
                    val_str = val_str.replace(',', '')
                    
                    try:
                        return float(val_str) if val_str else np.nan
                    except:
                        return np.nan
                
                cleaned_series = df_clean[col].apply(clean_numeric_string)
                
                # Check if cleaning was successful
                valid_conversions = cleaned_series.notna().sum()
                total_non_null = non_null_values.count()
                
                if valid_conversions / total_non_null > 0.8:
                    df_clean[col] = cleaned_series
                    col_corrections.append(f"Converted to numeric ({valid_conversions}/{total_non_null} values)")
                else:
                    report['warnings'].append(
                        f"{col}: Could not reliably convert to numeric "
                        f"({valid_conversions}/{total_non_null} successful)"
                    )
        
        # Try to detect and parse dates
        if df_clean[col].dtype == 'object' and not aggressive:
            sample = non_null_values.head(20)
            date_like = sum(1 for v in sample if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}', str(v)))
            
            if date_like / len(sample) > 0.5:
                col_issues.append("Potential date column stored as string")
                try:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    col_corrections.append("Converted to datetime")
                except:
                    report['warnings'].append(f"{col}: Failed to convert to datetime")
        
        # Check for whitespace issues
        if df_clean[col].dtype == 'object':
            has_leading_trailing = any(
                str(v) != str(v).strip() 
                for v in non_null_values.head(50) 
                if pd.notna(v)
            )
            
            if has_leading_trailing:
                col_issues.append("Leading/trailing whitespace detected")
                df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                col_corrections.append("Removed whitespace")
        
        # Check for encoding issues
        if df_clean[col].dtype == 'object':
            sample_str = ' '.join(str(v) for v in non_null_values.head(20))
            if any(ord(c) > 127 for c in sample_str):
                # Has non-ASCII characters - might be encoding issue
                try:
                    df_clean[col] = df_clean[col].apply(
                        lambda x: x.encode('latin1').decode('utf-8') if isinstance(x, str) else x
                    )
                    col_corrections.append("Fixed encoding issues")
                except:
                    pass
        
        # Store findings
        if col_issues:
            report['issues_found'].extend([f"{col}: {issue}" for issue in col_issues])
        if col_corrections:
            report['corrections_applied'].extend([f"{col}: {corr}" for corr in col_corrections])
            report['columns_modified'][col] = col_corrections
    
    report['final_shape'] = df_clean.shape
    report['success'] = len(report['corrections_applied']) > 0 or len(report['issues_found']) == 0
    
    return {
        'dataframe': df_clean,
        'report': report
    }


def smart_type_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Intelligently infer and convert column types.
    
    More aggressive than validate_and_clean_dataframe.
    Useful when data types are completely wrong.
    """
    df_result = df.copy()
    
    for col in df_result.columns:
        if df_result[col].dtype == 'object':
            # Try numeric first
            try:
                # Remove common non-numeric characters
                cleaned = df_result[col].astype(str).str.replace(r'[^\d\.\-\+eE]', '', regex=True)
                numeric_series = pd.to_numeric(cleaned, errors='coerce')
                
                # If more than 80% converted successfully, use it
                if numeric_series.notna().sum() / len(df_result) > 0.8:
                    df_result[col] = numeric_series
                    continue
            except:
                pass
            
            # Try datetime
            try:
                datetime_series = pd.to_datetime(df_result[col], errors='coerce')
                if datetime_series.notna().sum() / len(df_result) > 0.8:
                    df_result[col] = datetime_series
                    continue
            except:
                pass
            
            # Try boolean
            unique_vals = df_result[col].dropna().unique()
            if len(unique_vals) <= 2:
                bool_map = {
                    'true': True, 'false': False,
                    'yes': True, 'no': False,
                    '1': True, '0': False,
                    't': True, 'f': False,
                    'y': True, 'n': False
                }
                try:
                    df_result[col] = df_result[col].str.lower().map(bool_map)
                    if df_result[col].notna().sum() / len(df_result) > 0.8:
                        continue
                except:
                    pass
    
    return df_result


def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect data quality issues without modifying data.
    
    Returns comprehensive report of potential problems.
    """
    issues = {
        'critical': [],
        'warnings': [],
        'info': [],
        'recommendations': []
    }
    
    # Check for empty DataFrame
    if df.empty:
        issues['critical'].append("DataFrame is empty")
        return issues
    
    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        dup_pct = (dup_count / len(df)) * 100
        if dup_pct > 10:
            issues['warnings'].append(f"High duplicate rate: {dup_count} rows ({dup_pct:.1f}%)")
        else:
            issues['info'].append(f"Duplicates found: {dup_count} rows ({dup_pct:.1f}%)")
    
    # Check each column
    for col in df.columns:
        # Missing data
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        if missing_pct > 50:
            issues['critical'].append(f"{col}: {missing_pct:.1f}% missing data")
        elif missing_pct > 20:
            issues['warnings'].append(f"{col}: {missing_pct:.1f}% missing data")
        
        # Check for constant columns
        if df[col].nunique() == 1:
            issues['warnings'].append(f"{col}: Constant column (only one unique value)")
        
        # Check for high cardinality in object columns
        if df[col].dtype == 'object':
            cardinality = df[col].nunique() / len(df)
            if cardinality > 0.9:
                issues['info'].append(f"{col}: Very high cardinality ({cardinality:.1%})")
        
        # Check for potential type issues
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(50)
            if len(sample) > 0:
                # Check if looks numeric
                numeric_pattern = sum(1 for v in sample if re.search(r'^\s*[\d\.\-\+eE,]+\s*[a-zA-Z]*\s*$', str(v)))
                if numeric_pattern / len(sample) > 0.7:
                    issues['recommendations'].append(
                        f"{col}: Appears to contain numeric values stored as strings"
                    )
                
                # Check if looks like dates
                date_pattern = sum(1 for v in sample if re.search(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}', str(v)))
                if date_pattern / len(sample) > 0.5:
                    issues['recommendations'].append(
                        f"{col}: Appears to contain dates stored as strings"
                    )
    
    # Overall recommendations
    if any('numeric values stored as strings' in r for r in issues['recommendations']):
        issues['recommendations'].append(
            "Run validate_and_clean_dataframe() to automatically fix type issues"
        )
    
    return issues


# 7. DATA EXPORT

def export_to_excel(df: pd.DataFrame, filename: str = "export.xlsx", 
                   sheet_name: str = "Data") -> bytes:
    """Export DataFrame to Excel with formatting."""
    from io import BytesIO
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Get workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return output.getvalue()


def export_analysis_results(results: Dict[str, Any], filename: str = "analysis_results.xlsx") -> bytes:
    """Export analysis results to formatted Excel file."""
    from io import BytesIO
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Convert dict results to DataFrames and write to different sheets
        for sheet_name, data in results.items():
            if isinstance(data, pd.DataFrame):
                data.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit
            elif isinstance(data, dict):
                df = pd.DataFrame([data]).T
                df.columns = ['Value']
                df.to_excel(writer, sheet_name=sheet_name[:31])
    
    output.seek(0)
    return output.getvalue()
