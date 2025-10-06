# Testing Strategy

This document describes the testing strategy for the Autonomous Data Consulting project.

## Test Suite Overview

**Status**: 23/23 tests passing (100%)

### Test Files

```
tests/
├── conftest.py                           # Shared fixtures
├── test_clustering.py                    # K-means clustering tests
├── test_feature_engineering.py           # Feature creation tests
├── test_business_analytics.py            # RFM, growth rate, A/B tests
├── test_time_series_extra.py             # ARIMA, decomposition tests
├── test_text_analysis.py                 # Sentiment, topics, wordcloud
├── test_data_transformation_basic.py     # Sort, group, pivot tests
├── test_file_operations.py               # Excel export tests
├── test_time_features.py                 # Time feature extraction
├── test_types.py                         # Type validation tests
├── test_tools_mapping.py                 # Integration test (81 tools)
└── ... (2 more test files)
```

## Testing Approach

### 1. Unit Testing
- Each module tested independently
- Mock external dependencies (LLM, optional libraries)
- Focus on function correctness

### 2. Integration Testing
- `test_tools_mapping.py`: Validates all 81 registered tools
- Tests automatic parameter generation
- Ensures tools execute without errors
- Validates return types

### 3. Defensive Testing
- Graceful handling of missing dependencies (textblob, lifelines, spacy)
- Error dictionaries returned when libraries unavailable
- Tests pass even without optional dependencies

## Running Tests

### All Tests
```bash
pytest -q
# 23 passed, 17 warnings in 14.38s
```

### Specific Module
```bash
pytest tests/test_clustering.py -v
```

### With Coverage
```bash
pytest --cov=tools --cov-report=html
```

### Verbose Output
```bash
pytest -v -s
```

## Test Coverage by Module

| Module | Tests | Coverage |
|--------|-------|----------|
| Clustering | 3 | 100% |
| Feature Engineering | 4 | 100% |
| Business Analytics | 3 | 100% |
| Time Series | 2 | 100% |
| Text Analysis | 2 | Defensive |
| Data Transformation | 3 | 100% |
| File Operations | 2 | 100% |
| Tool Mapping | 1 | Integration |
| Type Validation | 1 | 100% |
| Time Features | 1 | 100% |

## Test Patterns

### Standard Test Pattern
```python
def test_function_name():
    # Arrange
    df = pd.DataFrame({'col': [1, 2, 3]})
    
    # Act
    result = function_under_test(df, param='value')
    
    # Assert
    assert isinstance(result, expected_type)
    assert 'expected_key' in result
```

### Defensive Test Pattern (Optional Dependencies)
```python
def test_optional_feature():
    df = pd.DataFrame({'text': ['sample']})
    result = function_with_optional_dep(df, 'text')
    
    # Accept both success and graceful error
    assert isinstance(result, dict)
    # If library missing, should return error dict
    if 'error' in result:
        assert 'library' in result['error'].lower()
    else:
        # If library present, validate success
        assert 'expected_key' in result
```

## Continuous Integration

### Pre-commit Checks
```bash
# Run before committing
pytest -q
python3 -c "import app; print('✅ Imports OK')"
```

### CI Pipeline (Recommended)
```yaml
- Run pytest
- Check import errors
- Validate tool registry
- Generate coverage report
```

## Adding New Tests

### For New Tools
1. Create test file: `tests/test_<module_name>.py`
2. Import the tool from `tools`
3. Create test data
4. Test normal cases
5. Test edge cases
6. Test error handling

### Example
```python
# tests/test_my_new_tool.py
import pandas as pd
from tools import my_new_tool

def test_my_new_tool_basic():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = my_new_tool(df, 'a')
    
    assert isinstance(result, dict)
    assert 'output' in result
    assert result['output'] > 0
```

## Known Test Warnings

### Non-Critical Warnings
- **Streamlit cache warnings**: Expected in test environment
- **Pandas FutureWarnings**: Deprecated `errors='ignore'` parameter
- **Statsmodels convergence**: Some models may not converge on small test data
- **Matplotlib deprecations**: Seaborn compatibility warnings

These warnings don't affect functionality and are tracked for future updates.

## Test Maintenance

### Regular Tasks
- [ ] Update tests when adding new tools
- [ ] Maintain 100% pass rate
- [ ] Review and update test data
- [ ] Keep fixtures up to date
- [ ] Document new test patterns

### Quality Metrics
- **Pass Rate**: 100% (23/23)
- **Execution Time**: ~14 seconds
- **Coverage**: Core modules fully covered
- **Integration**: All registered tools validated

## Notes
- Optional dependencies are lazy-imported (ReportLab/TextBlob) to avoid breaking collection.
- Prefer small, deterministic datasets in tests.
