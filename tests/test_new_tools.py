# tests/test_new_tools.py
"""Tests for new tools: XGBoost, LightGBM, Cohort Analysis, CLV."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_xgboost_classifier():
    """Test XGBoost classifier with graceful fallback."""
    from tools.machine_learning import xgboost_classifier
    
    # Create synthetic classification dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'class': np.random.choice(['A', 'B'], 100)
    })
    
    result = xgboost_classifier(
        df=df,
        x_columns=['feature1', 'feature2', 'feature3'],
        y_column='class',
        n_estimators=10,  # Small for testing
        test_size=0.3
    )
    
    # Should return either valid results or graceful error
    assert isinstance(result, dict)
    
    if 'error' not in result:
        # XGBoost is installed
        assert 'train_score' in result
        assert 'test_score' in result
        assert 'feature_importances' in result
        assert result['model_type'] == 'XGBoost'
        assert 0 <= result['train_score'] <= 1
        assert 0 <= result['test_score'] <= 1
    else:
        # XGBoost not installed - should have graceful error
        assert 'fallback' in result


def test_lightgbm_classifier():
    """Test LightGBM classifier with graceful fallback."""
    from tools.machine_learning import lightgbm_classifier
    
    # Create synthetic classification dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'class': np.random.choice(['A', 'B'], 100)
    })
    
    result = lightgbm_classifier(
        df=df,
        x_columns=['feature1', 'feature2', 'feature3'],
        y_column='class',
        n_estimators=10,
        test_size=0.3
    )
    
    # Should return either valid results or graceful error
    assert isinstance(result, dict)
    
    if 'error' not in result:
        # LightGBM is installed
        assert 'train_score' in result
        assert 'test_score' in result
        assert 'feature_importances' in result
        assert result['model_type'] == 'LightGBM'
        assert 0 <= result['train_score'] <= 1
        assert 0 <= result['test_score'] <= 1
    else:
        # LightGBM not installed - should have graceful error
        assert 'fallback' in result


def test_model_comparison():
    """Test model comparison functionality."""
    from tools.machine_learning import model_comparison
    
    # Create synthetic classification dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'class': np.random.choice(['A', 'B'], 100)
    })
    
    result = model_comparison(
        df=df,
        x_columns=['feature1', 'feature2', 'feature3'],
        y_column='class',
        models=['random_forest', 'gradient_boosting'],  # Use only sklearn models
        test_size=0.3
    )
    
    assert isinstance(result, dict)
    assert 'random_forest' in result
    assert 'gradient_boosting' in result
    
    # Check that at least one model succeeded
    valid_models = [k for k, v in result.items() if isinstance(v, dict) and 'test_score' in v]
    assert len(valid_models) >= 1
    
    # Check best model selection
    if 'best_model' in result:
        assert result['best_model'] in valid_models
        assert 'best_test_score' in result


def test_cohort_analysis():
    """Test cohort analysis for customer retention."""
    from tools.business_analytics import cohort_analysis
    
    # Create synthetic transaction data
    np.random.seed(42)
    n_customers = 50
    n_transactions = 200
    
    customers = [f'C{i:03d}' for i in range(n_customers)]
    base_date = datetime(2024, 1, 1)
    
    transactions = []
    for _ in range(n_transactions):
        customer = np.random.choice(customers)
        days_offset = np.random.randint(0, 180)
        date = base_date + timedelta(days=days_offset)
        value = np.random.uniform(10, 500)
        transactions.append({
            'customer_id': customer,
            'transaction_date': date,
            'value': value
        })
    
    df = pd.DataFrame(transactions)
    
    result = cohort_analysis(
        df=df,
        customer_col='customer_id',
        date_col='transaction_date',
        value_col='value',
        period='M'
    )
    
    assert isinstance(result, dict)
    assert 'cohort_sizes' in result
    assert 'retention_matrix' in result
    assert 'retention_rate_matrix' in result
    assert 'overall_retention_by_period' in result
    assert 'total_cohorts' in result
    assert result['period_type'] == 'M'
    assert 'insights' in result
    
    # Should have revenue analysis since value_col was provided
    assert 'revenue_analysis' in result
    if result['revenue_analysis']:
        assert 'total_revenue_by_cohort' in result['revenue_analysis']


def test_customer_lifetime_value():
    """Test CLV calculation."""
    from tools.business_analytics import customer_lifetime_value
    
    # Create synthetic customer transaction data
    np.random.seed(42)
    base_date = datetime(2024, 1, 1)
    
    transactions = []
    for customer_id in range(30):
        n_purchases = np.random.randint(1, 10)
        for purchase in range(n_purchases):
            date = base_date + timedelta(days=np.random.randint(0, 365))
            value = np.random.uniform(20, 200)
            transactions.append({
                'customer_id': f'C{customer_id:03d}',
                'purchase_date': date,
                'amount': value
            })
    
    df = pd.DataFrame(transactions)
    
    result = customer_lifetime_value(
        df=df,
        customer_col='customer_id',
        date_col='purchase_date',
        value_col='amount',
        prediction_months=12
    )
    
    assert isinstance(result, dict)
    assert 'summary' in result
    assert 'top_customers' in result
    assert 'prediction_months' in result
    
    summary = result['summary']
    assert 'total_customers' in summary
    assert 'avg_clv' in summary
    assert 'median_clv' in summary
    assert 'total_predicted_revenue' in summary
    assert 'avg_purchase_frequency' in summary
    assert 'segment_distribution' in summary
    
    # Validate metrics are reasonable
    assert summary['total_customers'] == 30
    assert summary['avg_clv'] > 0
    assert summary['total_predicted_revenue'] > 0


def test_tool_registry_new_tools():
    """Test that new tools are properly registered."""
    from tool_registry import TOOL_REGISTRY, get_tool_validation_info
    
    # Check new ML tools
    assert 'xgboost_classifier' in TOOL_REGISTRY
    assert 'lightgbm_classifier' in TOOL_REGISTRY
    assert 'model_comparison' in TOOL_REGISTRY
    
    # Check new business tools
    assert 'cohort_analysis' in TOOL_REGISTRY
    assert 'customer_lifetime_value' in TOOL_REGISTRY
    
    # Validate metadata
    xgb_info = get_tool_validation_info('xgboost_classifier')
    assert xgb_info is not None
    assert xgb_info['min_rows'] == 50
    assert xgb_info['requires_numeric'] is True
    assert xgb_info['category'] == 'machine_learning'
    
    cohort_info = get_tool_validation_info('cohort_analysis')
    assert cohort_info is not None
    assert cohort_info['min_rows'] == 30
    assert cohort_info['category'] == 'business_analytics'


def test_tool_validation_functions():
    """Test new validation helper functions."""
    from tool_registry import (
        get_tool_validation_info, 
        get_tools_by_category,
        TOOL_REGISTRY
    )
    
    # Test get_tools_by_category
    ml_tools = get_tools_by_category('machine_learning')
    assert 'xgboost_classifier' in ml_tools
    assert 'lightgbm_classifier' in ml_tools
    assert 'random_forest_classifier' in ml_tools
    
    business_tools = get_tools_by_category('business_analytics')
    assert 'cohort_analysis' in business_tools
    assert 'customer_lifetime_value' in business_tools
    assert 'rfm_analysis' in business_tools
    
    # Test validation via ToolMetadata.can_execute
    df_small = pd.DataFrame({'a': [1, 2, 3]})
    xgb_metadata = TOOL_REGISTRY['xgboost_classifier']
    can_execute, reason = xgb_metadata.can_execute(df_small)
    assert can_execute is False
    assert 'rows' in reason.lower()
    
    df_large = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'target': np.random.choice(['A', 'B'], 100)
    })
    can_execute, reason = xgb_metadata.can_execute(df_large)
    assert can_execute is True
    assert reason is None
