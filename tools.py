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
from wordcloud import WordCloud
import numpy as np
from typing import Dict
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
    df[column].fillna(fill_value, inplace=True)
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

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()

def get_exploratory_analysis(df: pd.DataFrame) -> str:
    """Generates an Exploratory Data Analysis (EDA) report."""
    desc = descriptive_stats(df)
    corr = correlation_matrix(df)
    report = f"Shape: {desc['shape']}\nTypes: {desc['types']}\nStats: {desc['stats']}\nCorr: {corr.to_string()}"
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
    text = ' '.join(df[text_column].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
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
            df_filled[col].fillna(median_val, inplace=True)
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
