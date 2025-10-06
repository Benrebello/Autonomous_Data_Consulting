# tools/machine_learning.py
"""Machine learning and predictive modeling tools."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, Any, List, Optional


def linear_regression(df: pd.DataFrame, x_columns: List[str], y_column: str) -> Dict[str, Any]:
    """Fit linear regression model.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        
    Returns:
        Dictionary with model coefficients and score
    """
    X = df[x_columns]
    y = df[y_column]
    model = LinearRegression()
    model.fit(X, y)
    return {
        "coefficients": dict(zip(x_columns, model.coef_)),
        "intercept": model.intercept_,
        "score": model.score(X, y)
    }


def logistic_regression(df: pd.DataFrame, x_columns: List[str], y_column: str) -> Dict[str, Any]:
    """Fit logistic regression for binary classification.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        
    Returns:
        Dictionary with model coefficients and score
    """
    X = df[x_columns]
    y = df[y_column]
    model = LogisticRegression()
    model.fit(X, y)
    return {
        "coefficients": dict(zip(x_columns, model.coef_[0])),
        "intercept": model.intercept_[0],
        "score": model.score(X, y)
    }


def random_forest_classifier(df: pd.DataFrame, x_columns: List[str], 
                             y_column: str, n_estimators: int = 100) -> Dict[str, Any]:
    """Train random forest classifier.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        n_estimators: Number of trees
        
    Returns:
        Dictionary with feature importances and score
    """
    X = df[x_columns]
    y = df[y_column]
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X, y)
    return {
        "feature_importances": dict(zip(x_columns, model.feature_importances_)),
        "score": model.score(X, y)
    }


def gradient_boosting_classifier(df: pd.DataFrame, x_columns: List[str], 
                                 y_column: str, n_estimators: int = 100) -> Dict[str, Any]:
    """Train gradient boosting classifier.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        n_estimators: Number of boosting stages
        
    Returns:
        Dictionary with feature importances and score
    """
    X = df[x_columns]
    y = df[y_column]
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return {
        "feature_importances": dict(zip(x_columns, model.feature_importances_)),
        "score": model.score(X, y)
    }


def svm_classifier(df: pd.DataFrame, x_columns: List[str], 
                   y_column: str, kernel: str = 'rbf') -> Dict[str, Any]:
    """Train SVM classifier.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        kernel: Kernel type ('rbf', 'linear', 'poly')
        
    Returns:
        Dictionary with model score
    """
    X = df[x_columns]
    y = df[y_column]
    model = SVC(kernel=kernel)
    model.fit(X, y)
    score = model.score(X, y)
    return {"score": score}


def knn_classifier(df: pd.DataFrame, x_columns: List[str], 
                   y_column: str, n_neighbors: int = 5) -> Dict[str, Any]:
    """Train KNN classifier.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        n_neighbors: Number of neighbors
        
    Returns:
        Dictionary with model score
    """
    X = df[x_columns]
    y = df[y_column]
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    score = model.score(X, y)
    return {"score": score}


def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normalize specified columns using StandardScaler.
    
    Args:
        df: Input DataFrame
        columns: List of columns to normalize
        
    Returns:
        DataFrame with normalized columns
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled


def impute_missing(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """Impute missing values.
    
    Args:
        df: Input DataFrame
        strategy: Imputation strategy ('mean', 'median', 'most_frequent')
        
    Returns:
        DataFrame with imputed values
    """
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
    """Evaluate a model with cross-validation.
    
    Args:
        model: Scikit-learn model
        X: Features
        y: Target
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    return {
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std(),
        'cv_scores': scores.tolist()
    }


def hyperparameter_tuning(df: pd.DataFrame, x_columns: List[str], y_column: str,
                         model_type: str = 'random_forest') -> Dict[str, Any]:
    """Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        model_type: Type of model ('random_forest', 'gradient_boosting')
        
    Returns:
        Dictionary with best parameters and score
    """
    X = df[x_columns]
    y = df[y_column]
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    else:
        return {"error": f"Unsupported model type: {model_type}"}
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    return {
        'best_parameters': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'all_results': {
            'mean_test_scores': grid_search.cv_results_['mean_test_score'].tolist(),
            'params': grid_search.cv_results_['params']
        }
    }


def feature_importance_analysis(df: pd.DataFrame, x_columns: List[str], 
                               y_column: str) -> Dict[str, Any]:
    """Analyze feature importance using Random Forest.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        
    Returns:
        Dictionary with feature importances sorted by importance
    """
    X = df[x_columns]
    y = df[y_column]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importances = dict(zip(x_columns, model.feature_importances_))
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'feature_importances': dict(sorted_importances),
        'top_5_features': [feat for feat, _ in sorted_importances[:5]],
        'model_score': model.score(X, y)
    }


def model_evaluation_detailed(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Detailed model evaluation with multiple metrics.
    
    Args:
        model: Trained scikit-learn model
        X: Features
        y: True labels
        
    Returns:
        Dictionary with detailed evaluation metrics
    """
    y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    
    # For binary classification
    if len(np.unique(y)) == 2:
        precision = precision_score(y, y_pred, average='binary')
        recall = recall_score(y, y_pred, average='binary')
        f1 = f1_score(y, y_pred, average='binary')
        
        # ROC AUC if model has predict_proba
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, y_proba)
        else:
            roc_auc = None
    else:
        # Multi-class
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        roc_auc = None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'interpretation': {
            'accuracy': f'{accuracy*100:.2f}% of predictions are correct',
            'precision': f'{precision*100:.2f}% of positive predictions are actually positive',
            'recall': f'{recall*100:.2f}% of actual positives were identified',
            'f1_score': f'Harmonic mean of precision and recall: {f1:.3f}'
        }
    }


def pca_dimensionality(df: pd.DataFrame, n_components: int = 2) -> Dict[str, Any]:
    """Perform PCA dimensionality reduction on numeric columns.

    Args:
        df: Input DataFrame
        n_components: Number of principal components

    Returns:
        Dictionary with principal components and explained variance ratio
    """
    num_cols = df.select_dtypes(include='number').columns
    X = df[num_cols].dropna()
    if X.empty:
        return {'error': 'No numeric data available for PCA'}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
    pcs = pca.fit_transform(X_scaled)
    return {
        'principal_components': pcs.tolist(),
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'components': pca.components_.tolist()
    }


def select_features(df: pd.DataFrame, target_column: str, k: int = 10) -> Dict[str, Any]:
    """Select top-K features using univariate regression tests.

    Args:
        df: Input DataFrame
        target_column: Target column name (numeric)
        k: Number of features to select

    Returns:
        Dictionary with selected feature names and scores
    """
    if target_column not in df.columns:
        return {'error': f'target_column {target_column} not in DataFrame'}
    num_cols = [c for c in df.select_dtypes(include='number').columns if c != target_column]
    if not num_cols:
        return {'error': 'No numeric features available'}
    X = df[num_cols].fillna(0)
    y = df[target_column].fillna(0)
    selector = SelectKBest(score_func=f_regression, k=min(k, len(num_cols)))
    selector.fit(X, y)
    scores = selector.scores_
    support_mask = selector.get_support()
    # Convert numpy bool to Python bool explicitly
    selected = [col for col, m in zip(num_cols, support_mask.tolist()) if m]
    return {
        'selected_features': selected,
        'scores': {col: float(score) for col, score in zip(num_cols, (scores.tolist() if scores is not None else [])) if score is not None and not np.isnan(score)}
    }


def perform_multiple_regression(df: pd.DataFrame, x_columns: list, y_column: str) -> Dict[str, Any]:
    """Perform multiple linear regression.

    Args:
        df: Input DataFrame
        x_columns: Predictor columns
        y_column: Target column

    Returns:
        Dictionary with coefficients, intercept and score
    """
    X = df[x_columns]
    y = df[y_column]
    model = LinearRegression()
    model.fit(X, y)
    return {
        'coefficients': dict(zip(x_columns, model.coef_)),
        'intercept': float(model.intercept_),
        'score': float(model.score(X, y))
    }


def xgboost_classifier(
    df: pd.DataFrame, 
    x_columns: List[str], 
    y_column: str, 
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Train XGBoost classifier with train/test split.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate (eta)
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary with feature importances, train/test scores, and predictions
    """
    try:
        import xgboost as xgb
    except ImportError:
        return {
            'error': 'XGBoost not installed. Install with: pip install xgboost',
            'fallback': 'Using GradientBoostingClassifier instead'
        }
    
    X = df[x_columns]
    y = df[y_column]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Predictions
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return {
        'feature_importances': dict(zip(x_columns, model.feature_importances_)),
        'train_score': float(train_score),
        'test_score': float(test_score),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'model_type': 'XGBoost',
        'params': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }
    }


def lightgbm_classifier(
    df: pd.DataFrame, 
    x_columns: List[str], 
    y_column: str, 
    n_estimators: int = 100,
    max_depth: int = -1,
    learning_rate: float = 0.1,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Train LightGBM classifier with train/test split.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth (-1 for no limit)
        learning_rate: Learning rate
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary with feature importances, train/test scores, and predictions
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return {
            'error': 'LightGBM not installed. Install with: pip install lightgbm',
            'fallback': 'Using GradientBoostingClassifier instead'
        }
    
    X = df[x_columns]
    y = df[y_column]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Predictions
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return {
        'feature_importances': dict(zip(x_columns, model.feature_importances_)),
        'train_score': float(train_score),
        'test_score': float(test_score),
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'model_type': 'LightGBM',
        'params': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }
    }


def model_comparison(
    df: pd.DataFrame,
    x_columns: List[str],
    y_column: str,
    models: Optional[List[str]] = None,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Compare multiple classification models on the same dataset.
    
    Args:
        df: Input DataFrame
        x_columns: List of predictor column names
        y_column: Target column name
        models: List of model names to compare (default: all available)
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary with comparison results for each model
    """
    if models is None:
        models = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
    
    X = df[x_columns]
    y = df[y_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    results = {}
    
    for model_name in models:
        try:
            if model_name == 'random_forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == 'gradient_boosting':
                model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            elif model_name == 'xgboost':
                try:
                    import xgboost as xgb
                    model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
                except ImportError:
                    results[model_name] = {'error': 'XGBoost not installed'}
                    continue
            elif model_name == 'lightgbm':
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
                except ImportError:
                    results[model_name] = {'error': 'LightGBM not installed'}
                    continue
            else:
                results[model_name] = {'error': f'Unknown model: {model_name}'}
                continue
            
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            results[model_name] = {
                'train_score': float(train_score),
                'test_score': float(test_score),
                'overfitting': float(train_score - test_score)
            }
        except Exception as e:
            results[model_name] = {'error': str(e)}
    
    # Rank by test score
    valid_results = {k: v for k, v in results.items() if 'test_score' in v}
    if valid_results:
        best_model = max(valid_results.items(), key=lambda x: x[1]['test_score'])
        results['best_model'] = best_model[0]
        results['best_test_score'] = best_model[1]['test_score']
    
    return results
