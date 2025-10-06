# tests/test_data_connectors.py
"""Tests for data connectors system."""

import pytest
import pandas as pd
import numpy as np
from data_connectors import (
    ConnectionConfig,
    SQLConnector,
    APIConnector,
    DataSourceManager,
    create_api_connector
)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'value': [10.5, 20.3, 15.7, 30.2, 25.8]
    })


def test_connection_config_creation():
    """Test ConnectionConfig creation."""
    config = ConnectionConfig(
        source_type='postgresql',
        host='localhost',
        port=5432,
        database='testdb',
        username='user',
        password='pass'
    )
    
    assert config.source_type == 'postgresql'
    assert config.host == 'localhost'
    assert config.port == 5432
    assert config.database == 'testdb'


def test_sqlite_connector_basic():
    """Test SQLite connector with in-memory database."""
    config = ConnectionConfig(
        source_type='sqlite',
        database=':memory:'
    )
    
    connector = SQLConnector(config)
    
    # Test connection
    assert connector.connect() is True
    
    # Create test table and insert data
    with connector:
        # Create table
        connector.query("CREATE TABLE test (id INTEGER, name TEXT, value REAL)")
        
        # Insert data
        connector.query("INSERT INTO test VALUES (1, 'Alice', 10.5)")
        connector.query("INSERT INTO test VALUES (2, 'Bob', 20.3)")
        
        # Query data
        df = connector.query("SELECT * FROM test")
        
        assert len(df) == 2
        assert 'id' in df.columns
        assert 'name' in df.columns
        assert df.iloc[0]['name'] == 'Alice'


def test_sqlite_connector_list_tables():
    """Test listing tables in SQLite."""
    config = ConnectionConfig(
        source_type='sqlite',
        database=':memory:'
    )
    
    connector = SQLConnector(config)
    
    with connector:
        # Create test tables
        connector.query("CREATE TABLE table1 (id INTEGER)")
        connector.query("CREATE TABLE table2 (id INTEGER)")
        
        tables = connector.list_tables()
        
        assert 'table1' in tables
        assert 'table2' in tables


def test_sqlite_connector_get_schema():
    """Test getting table schema."""
    config = ConnectionConfig(
        source_type='sqlite',
        database=':memory:'
    )
    
    connector = SQLConnector(config)
    
    with connector:
        connector.query("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT NOT NULL, value REAL)")
        
        schema = connector.get_table_schema('test')
        
        assert isinstance(schema, pd.DataFrame)
        assert len(schema) == 3  # 3 columns


def test_api_connector_mock():
    """Test API connector with mock endpoint."""
    # Using httpbin.org as a test API
    config = ConnectionConfig(
        source_type='api',
        api_endpoint='https://httpbin.org'
    )
    
    connector = APIConnector(config)
    
    # Test connection
    test_result = connector.test_connection()
    assert test_result['status'] == 'success'


def test_api_connector_json_response():
    """Test API connector with JSON response."""
    config = ConnectionConfig(
        source_type='api',
        api_endpoint='https://jsonplaceholder.typicode.com'
    )
    
    connector = APIConnector(config)
    
    with connector:
        # Get posts (returns array)
        df = connector.query('posts', params={'_limit': 5})
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'id' in df.columns
        assert 'title' in df.columns


def test_data_source_manager_registration():
    """Test DataSourceManager source registration."""
    manager = DataSourceManager()
    
    # Register SQLite source
    config = ConnectionConfig(
        source_type='sqlite',
        database=':memory:'
    )
    
    result = manager.register_source('test_db', config)
    assert result is True
    
    # Check source is registered
    sources = manager.list_sources()
    assert len(sources) == 1
    assert sources[0]['name'] == 'test_db'
    assert sources[0]['type'] == 'sqlite'


def test_data_source_manager_load_data():
    """Test loading data through DataSourceManager."""
    manager = DataSourceManager()
    
    # Register SQLite source
    config = ConnectionConfig(
        source_type='sqlite',
        database=':memory:'
    )
    manager.register_source('test_db', config)
    
    # Create table
    manager.load_data('test_db', "CREATE TABLE test (id INTEGER, name TEXT)", cache=False)
    
    # Insert data
    manager.load_data('test_db', "INSERT INTO test VALUES (1, 'Alice')", cache=False)
    manager.load_data('test_db', "INSERT INTO test VALUES (2, 'Bob')", cache=False)
    
    # Query data
    df = manager.load_data('test_db', "SELECT * FROM test", cache=True)
    
    assert len(df) == 2
    assert df.iloc[0]['name'] == 'Alice'


def test_data_source_manager_caching():
    """Test data caching in DataSourceManager."""
    manager = DataSourceManager()
    
    config = ConnectionConfig(
        source_type='sqlite',
        database=':memory:'
    )
    manager.register_source('test_db', config)
    
    # Load with caching
    manager.load_data('test_db', "SELECT 1 as test", cache=True)
    
    # Check cached data
    cached = manager.get_cached_data('test_db')
    assert cached is not None
    assert len(cached) == 1
    
    # Clear cache
    manager.clear_cache('test_db')
    cached = manager.get_cached_data('test_db')
    assert cached is None


def test_data_source_manager_multiple_sources():
    """Test managing multiple data sources."""
    manager = DataSourceManager()
    
    # Register multiple sources
    config1 = ConnectionConfig(source_type='sqlite', database=':memory:')
    config2 = ConnectionConfig(source_type='api', api_endpoint='https://httpbin.org')
    
    manager.register_source('db1', config1)
    manager.register_source('api1', config2)
    
    sources = manager.list_sources()
    assert len(sources) == 2
    
    source_names = [s['name'] for s in sources]
    assert 'db1' in source_names
    assert 'api1' in source_names


def test_data_source_manager_test_connections():
    """Test testing all connections."""
    manager = DataSourceManager()
    
    config = ConnectionConfig(source_type='sqlite', database=':memory:')
    manager.register_source('test_db', config)
    
    results = manager.test_all_connections()
    
    assert 'test_db' in results
    assert results['test_db']['status'] == 'success'


def test_create_api_connector_convenience():
    """Test convenience function for API connector."""
    connector = create_api_connector(
        api_endpoint='https://httpbin.org',
        api_key='test_key'
    )
    
    assert isinstance(connector, APIConnector)
    assert connector.config.api_endpoint == 'https://httpbin.org'
    assert connector.config.api_key == 'test_key'


def test_data_source_manager_refresh_data():
    """Test refreshing cached data."""
    manager = DataSourceManager()
    
    config = ConnectionConfig(source_type='sqlite', database=':memory:')
    manager.register_source('test_db', config)
    
    # Initial load
    manager.load_data('test_db', "CREATE TABLE test (id INTEGER)", cache=False)
    manager.load_data('test_db', "INSERT INTO test VALUES (1)", cache=False)
    df1 = manager.load_data('test_db', "SELECT * FROM test", cache=True)
    
    # Add more data
    manager.load_data('test_db', "INSERT INTO test VALUES (2)", cache=False)
    
    # Refresh
    df2 = manager.refresh_data('test_db', "SELECT * FROM test")
    
    assert len(df1) == 1
    assert len(df2) == 2


def test_sql_connector_context_manager():
    """Test SQL connector as context manager."""
    config = ConnectionConfig(source_type='sqlite', database=':memory:')
    connector = SQLConnector(config)
    
    with connector as conn:
        df = conn.query("SELECT 1 as test")
        assert len(df) == 1
    
    # Connection should be closed after context
    assert connector._connection is None


def test_api_connector_with_headers():
    """Test API connector with custom headers."""
    config = ConnectionConfig(
        source_type='api',
        api_endpoint='https://httpbin.org',
        additional_params={
            'headers': {
                'Custom-Header': 'test-value'
            }
        }
    )
    
    connector = APIConnector(config)
    connector.connect()
    
    assert 'Custom-Header' in connector._connection.headers
    assert connector._connection.headers['Custom-Header'] == 'test-value'
    
    connector.disconnect()


def test_data_source_manager_error_handling():
    """Test error handling in DataSourceManager."""
    manager = DataSourceManager()
    
    # Try to load from non-existent source
    with pytest.raises(ValueError, match="not registered"):
        manager.load_data('non_existent', "SELECT 1")
    
    # Try to register with invalid source type
    config = ConnectionConfig(source_type='invalid_type')
    with pytest.raises(RuntimeError):
        manager.register_source('test', config)


def test_sql_connector_query_with_params():
    """Test SQL connector with parameterized queries."""
    config = ConnectionConfig(source_type='sqlite', database=':memory:')
    connector = SQLConnector(config)
    
    with connector:
        # Create and populate table
        connector.query("CREATE TABLE test (id INTEGER, value TEXT)")
        connector.query("INSERT INTO test VALUES (1, 'A'), (2, 'B'), (3, 'C')")
        
        # Parameterized query
        df = connector.query("SELECT * FROM test WHERE id > 1")
        
        assert len(df) == 2
        assert df.iloc[0]['id'] == 2


def test_api_connector_post_request():
    """Test API connector with POST request."""
    config = ConnectionConfig(
        source_type='api',
        api_endpoint='https://httpbin.org'
    )
    
    connector = APIConnector(config)
    
    with connector:
        # POST request with JSON data
        df = connector.query(
            'post',
            method='POST',
            json={'test': 'data'}
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


def test_connection_config_with_connection_string():
    """Test ConnectionConfig with connection string."""
    config = ConnectionConfig(
        source_type='postgresql',
        connection_string='postgresql://user:pass@localhost:5432/testdb'
    )
    
    assert config.connection_string is not None
    assert config.source_type == 'postgresql'
