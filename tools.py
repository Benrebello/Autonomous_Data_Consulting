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
from textblob import TextBlob
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
    """Detects outliers using IQR or Z-score."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['outlier'] = ((df[column] < lower_bound) | (df[column] > upper_bound))
    elif method == 'zscore':
        mean = df[column].mean()
        std = df[column].std()
        df['z_score'] = (df[column] - mean) / std
        df['outlier'] = (df['z_score'].abs() > 3)
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
    """Get correlation of all variables with target."""
    if target_column not in df.columns:
        return {}
    correlations = df.corr()[target_column].to_dict()
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
    """Decompose time series into trend, seasonal, residual."""
    decomposition = seasonal_decompose(df[column], model='additive', period=period)
    return {
        "trend": decomposition.trend.tolist(),
        "seasonal": decomposition.seasonal.tolist(),
        "residual": decomposition.resid.tolist()
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
    """Select top k features using mutual information."""
    if target_column not in df.columns:
        return {}
    X = df.drop(columns=[target_column])
    y = df[target_column]
    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return {"selected_features": selected_features, "scores": selector.scores_.tolist()}

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
    """Perform sentiment analysis on text column."""
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
    """Simple topic modeling using LDA."""
    texts = df[text_column].dropna().astype(str).tolist()
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
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
