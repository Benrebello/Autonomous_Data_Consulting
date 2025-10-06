# automl_intelligence.py
"""AutoML Intelligence System for automatic plan optimization and strategy recommendation."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from tool_registry import TOOL_REGISTRY, get_tool_validation_info


@dataclass
class DataProfile:
    """Data profiling results for intelligent decision making."""
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    missing_pct: float
    duplicate_pct: float
    has_target: bool
    is_balanced: Optional[bool]
    is_time_series: bool
    data_quality_score: float  # 0-100


class AutoMLIntelligence:
    """Intelligent system for automatic plan optimization and strategy recommendation."""
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[str, Any]:
        """Initialize strategy recommendation rules."""
        return {
            'imputation': {
                'mean': {'condition': 'numeric_normal_distributed', 'priority': 1},
                'median': {'condition': 'numeric_skewed', 'priority': 2},
                'mode': {'condition': 'categorical', 'priority': 1},
                'knn': {'condition': 'missing_pct < 20 and n_rows > 100', 'priority': 3},
                'drop': {'condition': 'missing_pct > 50', 'priority': 4}
            },
            'scaling': {
                'standard': {'condition': 'normal_distributed', 'priority': 1},
                'minmax': {'condition': 'bounded_range', 'priority': 2},
                'robust': {'condition': 'has_outliers', 'priority': 3}
            },
            'model_selection': {
                'xgboost': {'condition': 'n_rows >= 50 and classification', 'priority': 1},
                'lightgbm': {'condition': 'n_rows >= 1000 and classification', 'priority': 1},
                'random_forest': {'condition': 'n_rows >= 50 and need_interpretability', 'priority': 2},
                'logistic_regression': {'condition': 'n_rows < 50 or linear_separable', 'priority': 3}
            },
            'hyperparameter_tuning': {
                'grid_search': {'condition': 'n_rows < 1000 and time_available', 'priority': 1},
                'random_search': {'condition': 'n_rows >= 1000', 'priority': 2},
                'bayesian': {'condition': 'expensive_model', 'priority': 3}
            }
        }
    
    def profile_data(self, df: pd.DataFrame, target_col: Optional[str] = None) -> DataProfile:
        """Profile dataset for intelligent decision making.
        
        Args:
            df: Input DataFrame
            target_col: Optional target column name
            
        Returns:
            DataProfile with comprehensive data characteristics
        """
        n_rows, n_cols = df.shape
        
        # Column types
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        
        n_numeric = len(numeric_cols)
        n_categorical = len(categorical_cols)
        n_datetime = len(datetime_cols)
        
        # Missing data
        missing_pct = (df.isnull().sum().sum() / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
        
        # Duplicates
        duplicate_pct = (df.duplicated().sum() / n_rows) * 100 if n_rows > 0 else 0
        
        # Target analysis
        has_target = target_col is not None and target_col in df.columns
        is_balanced = None
        if has_target and df[target_col].dtype in ['object', 'category', 'int64']:
            value_counts = df[target_col].value_counts()
            if len(value_counts) > 0:
                balance_ratio = value_counts.min() / value_counts.max()
                is_balanced = balance_ratio > 0.7  # Consider balanced if ratio > 70%
        
        # Time series detection
        is_time_series = n_datetime > 0 or any('date' in col.lower() or 'time' in col.lower() for col in df.columns)
        
        # Data quality score (0-100)
        quality_score = 100.0
        quality_score -= missing_pct  # Penalize missing data
        quality_score -= duplicate_pct * 0.5  # Penalize duplicates (less severe)
        quality_score = max(0, min(100, quality_score))
        
        return DataProfile(
            n_rows=n_rows,
            n_cols=n_cols,
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_datetime=n_datetime,
            missing_pct=missing_pct,
            duplicate_pct=duplicate_pct,
            has_target=has_target,
            is_balanced=is_balanced,
            is_time_series=is_time_series,
            data_quality_score=quality_score
        )
    
    def recommend_imputation_strategy(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Recommend best imputation strategy for a column.
        
        Args:
            df: Input DataFrame
            column: Column name to impute
            
        Returns:
            Dictionary with recommended strategy and reasoning
        """
        col_data = df[column]
        missing_pct = (col_data.isnull().sum() / len(col_data)) * 100
        
        # If too much missing, recommend dropping
        if missing_pct > 50:
            return {
                'strategy': 'drop',
                'reason': f'Column has {missing_pct:.1f}% missing values (>50%)',
                'confidence': 0.9
            }
        
        # For categorical
        if col_data.dtype in ['object', 'category']:
            return {
                'strategy': 'mode',
                'reason': 'Categorical column - mode imputation preserves distribution',
                'confidence': 0.85
            }
        
        # For numeric
        if col_data.dtype in ['int64', 'float64']:
            # Check for skewness
            try:
                skewness = col_data.dropna().skew()
                if abs(skewness) > 1:
                    return {
                        'strategy': 'median',
                        'reason': f'Numeric column with high skewness ({skewness:.2f}) - median is robust',
                        'confidence': 0.9
                    }
                else:
                    return {
                        'strategy': 'mean',
                        'reason': f'Numeric column with low skewness ({skewness:.2f}) - mean preserves distribution',
                        'confidence': 0.85
                    }
            except Exception:
                return {
                    'strategy': 'median',
                    'reason': 'Numeric column - median is safe default',
                    'confidence': 0.7
                }
        
        # Default
        return {
            'strategy': 'drop',
            'reason': 'Unknown column type',
            'confidence': 0.5
        }
    
    def recommend_model(self, profile: DataProfile, task_type: str = 'classification') -> Dict[str, Any]:
        """Recommend best ML model based on data profile.
        
        Args:
            profile: DataProfile object
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with recommended model and reasoning
        """
        recommendations = []
        
        # XGBoost - best for most cases
        if profile.n_rows >= 50 and profile.n_numeric >= 2:
            recommendations.append({
                'model': 'xgboost_classifier' if task_type == 'classification' else 'xgboost_regressor',
                'reason': 'State-of-the-art performance, handles missing values, feature importance',
                'confidence': 0.95,
                'priority': 1
            })
        
        # LightGBM - best for large datasets
        if profile.n_rows >= 1000 and profile.n_numeric >= 2:
            recommendations.append({
                'model': 'lightgbm_classifier' if task_type == 'classification' else 'lightgbm_regressor',
                'reason': 'Fast training, excellent for large datasets, lower memory usage',
                'confidence': 0.95,
                'priority': 1
            })
        
        # Random Forest - good interpretability
        if profile.n_rows >= 50:
            recommendations.append({
                'model': 'random_forest_classifier' if task_type == 'classification' else 'random_forest_regressor',
                'reason': 'Good baseline, interpretable feature importance, robust to overfitting',
                'confidence': 0.85,
                'priority': 2
            })
        
        # Logistic/Linear Regression - simple baseline
        if task_type == 'classification':
            recommendations.append({
                'model': 'logistic_regression',
                'reason': 'Simple baseline, fast training, good for linearly separable data',
                'confidence': 0.7,
                'priority': 3
            })
        else:
            recommendations.append({
                'model': 'linear_regression',
                'reason': 'Simple baseline, interpretable coefficients',
                'confidence': 0.7,
                'priority': 3
            })
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (x['priority'], -x['confidence']))
        
        return {
            'primary_recommendation': recommendations[0] if recommendations else None,
            'alternatives': recommendations[1:3] if len(recommendations) > 1 else [],
            'all_options': recommendations
        }
    
    def should_tune_hyperparameters(self, profile: DataProfile, model: str) -> Dict[str, Any]:
        """Decide if hyperparameter tuning is worth it.
        
        Args:
            profile: DataProfile object
            model: Model name
            
        Returns:
            Dictionary with recommendation and reasoning
        """
        # Don't tune if data is too small
        if profile.n_rows < 100:
            return {
                'should_tune': False,
                'reason': 'Dataset too small (<100 rows) - tuning may overfit',
                'confidence': 0.9
            }
        
        # Don't tune simple models
        if model in ['logistic_regression', 'linear_regression']:
            return {
                'should_tune': False,
                'reason': 'Simple model with few hyperparameters - default values usually sufficient',
                'confidence': 0.85
            }
        
        # Tune complex models with sufficient data
        if model in ['xgboost_classifier', 'lightgbm_classifier', 'random_forest_classifier']:
            if profile.n_rows >= 500:
                return {
                    'should_tune': True,
                    'reason': f'Complex model with sufficient data ({profile.n_rows} rows) - tuning can improve performance 5-15%',
                    'confidence': 0.9,
                    'method': 'random_search' if profile.n_rows >= 1000 else 'grid_search'
                }
            else:
                return {
                    'should_tune': True,
                    'reason': 'Complex model - light tuning recommended',
                    'confidence': 0.7,
                    'method': 'grid_search',
                    'param_grid_size': 'small'
                }
        
        # Default: don't tune
        return {
            'should_tune': False,
            'reason': 'Default parameters usually sufficient',
            'confidence': 0.6
        }
    
    def optimize_plan(self, plan: Dict[str, Any], df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Optimize execution plan based on data profiling and intelligent recommendations.
        
        Args:
            plan: Original execution plan
            df: Input DataFrame
            target_col: Optional target column
            
        Returns:
            Optimized plan with intelligent enhancements
        """
        # Profile data
        profile = self.profile_data(df, target_col)
        
        # Get execution plan tasks
        tasks = plan.get('execution_plan', [])
        if not tasks:
            return plan
        
        optimized_tasks = []
        optimization_log = []
        
        for task in tasks:
            tool_name = task.get('tool_to_use', '')
            optimized_task = task.copy()
            
            # Optimize ML model selection
            if tool_name in ['random_forest_classifier', 'logistic_regression', 'gradient_boosting_classifier']:
                model_rec = self.recommend_model(profile, 'classification')
                if model_rec['primary_recommendation']:
                    recommended_model = model_rec['primary_recommendation']['model']
                    if recommended_model != tool_name:
                        optimized_task['tool_to_use'] = recommended_model
                        optimization_log.append({
                            'original_tool': tool_name,
                            'optimized_tool': recommended_model,
                            'reason': model_rec['primary_recommendation']['reason'],
                            'confidence': model_rec['primary_recommendation']['confidence']
                        })
            
            # Add hyperparameter tuning if beneficial
            if tool_name in ['xgboost_classifier', 'lightgbm_classifier', 'random_forest_classifier']:
                tune_rec = self.should_tune_hyperparameters(profile, tool_name)
                if tune_rec['should_tune']:
                    # Add tuning task after model training
                    tuning_task = {
                        'task_id': len(optimized_tasks) + 100,
                        'description': f'Hyperparameter tuning for {tool_name}',
                        'agent_responsible': 'DataScientistAgent',
                        'tool_to_use': 'hyperparameter_tuning',
                        'dependencies': [task['task_id']],
                        'inputs': {
                            'df': '@df',
                            'x_columns': task.get('inputs', {}).get('x_columns', []),
                            'y_column': task.get('inputs', {}).get('y_column', 'target'),
                            'model_type': tool_name.replace('_classifier', '')
                        },
                        'output_variable': f"tuned_{task.get('output_variable', 'model')}"
                    }
                    optimization_log.append({
                        'added_task': 'hyperparameter_tuning',
                        'reason': tune_rec['reason'],
                        'confidence': tune_rec['confidence']
                    })
            
            optimized_tasks.append(optimized_task)
        
        # Create optimized plan
        optimized_plan = plan.copy()
        optimized_plan['execution_plan'] = optimized_tasks
        optimized_plan['optimization_metadata'] = {
            'data_profile': {
                'n_rows': profile.n_rows,
                'n_cols': profile.n_cols,
                'n_numeric': profile.n_numeric,
                'n_categorical': profile.n_categorical,
                'data_quality_score': profile.data_quality_score,
                'missing_pct': profile.missing_pct
            },
            'optimizations_applied': optimization_log,
            'optimizer_version': '1.0'
        }
        
        return optimized_plan
    
    def get_data_quality_recommendations(self, profile: DataProfile) -> List[Dict[str, Any]]:
        """Get recommendations for improving data quality.
        
        Args:
            profile: DataProfile object
            
        Returns:
            List of recommendations with priorities
        """
        recommendations = []
        
        # Missing data
        if profile.missing_pct > 5:
            severity = 'high' if profile.missing_pct > 20 else 'medium'
            recommendations.append({
                'issue': 'missing_data',
                'severity': severity,
                'message': f'{profile.missing_pct:.1f}% of data is missing',
                'recommendation': 'Apply intelligent imputation or consider dropping columns with >50% missing',
                'priority': 1 if severity == 'high' else 2
            })
        
        # Duplicates
        if profile.duplicate_pct > 1:
            recommendations.append({
                'issue': 'duplicates',
                'severity': 'medium',
                'message': f'{profile.duplicate_pct:.1f}% of rows are duplicates',
                'recommendation': 'Remove duplicates to avoid data leakage and improve model performance',
                'priority': 2
            })
        
        # Small dataset
        if profile.n_rows < 100:
            recommendations.append({
                'issue': 'small_dataset',
                'severity': 'high',
                'message': f'Only {profile.n_rows} rows available',
                'recommendation': 'Consider collecting more data or using simpler models to avoid overfitting',
                'priority': 1
            })
        
        # Few features
        if profile.n_numeric < 2 and profile.n_categorical < 2:
            recommendations.append({
                'issue': 'few_features',
                'severity': 'medium',
                'message': 'Very few features available for modeling',
                'recommendation': 'Consider feature engineering to create additional predictive features',
                'priority': 2
            })
        
        # Imbalanced target
        if profile.has_target and profile.is_balanced is False:
            recommendations.append({
                'issue': 'imbalanced_target',
                'severity': 'high',
                'message': 'Target variable is imbalanced',
                'recommendation': 'Use stratified sampling, class weights, or SMOTE for better model performance',
                'priority': 1
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations
