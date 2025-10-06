# data_connectors.py
"""Data source connectors for external data integration."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json


@dataclass
class ConnectionConfig:
    """Configuration for data source connection."""
    source_type: str  # 'sql', 'bigquery', 'api', 'csv', 'excel'
    connection_string: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    api_key: Optional[str] = None
    api_endpoint: Optional[str] = None
    project_id: Optional[str] = None  # For BigQuery
    credentials_path: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None


class DataConnector(ABC):
    """Abstract base class for data connectors."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._connection = None
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to data source."""
        pass
    
    @abstractmethod
    def query(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute query and return DataFrame."""
        pass
    
    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """Test connection and return status."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class SQLConnector(DataConnector):
    """Connector for SQL databases (PostgreSQL, MySQL, SQLite, SQL Server)."""
    
    def connect(self) -> bool:
        """Establish SQL database connection."""
        try:
            import sqlalchemy
            from sqlalchemy import create_engine
            
            if self.config.connection_string:
                engine = create_engine(self.config.connection_string)
            else:
                # Build connection string from components
                if self.config.source_type == 'postgresql':
                    conn_str = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
                elif self.config.source_type == 'mysql':
                    conn_str = f"mysql+pymysql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
                elif self.config.source_type == 'sqlite':
                    conn_str = f"sqlite:///{self.config.database}"
                elif self.config.source_type == 'mssql':
                    conn_str = f"mssql+pyodbc://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}?driver=ODBC+Driver+17+for+SQL+Server"
                else:
                    raise ValueError(f"Unsupported SQL type: {self.config.source_type}")
                
                engine = create_engine(conn_str)
            
            self._connection = engine
            return True
        except ImportError:
            raise ImportError("SQLAlchemy not installed. Install with: pip install sqlalchemy")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SQL database: {str(e)}")
    
    def disconnect(self) -> bool:
        """Close SQL database connection."""
        if self._connection:
            self._connection.dispose()
            self._connection = None
        return True
    
    def query(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute SQL query and return DataFrame.
        
        Args:
            query: SQL query string
            **kwargs: Additional parameters for pd.read_sql
            
        Returns:
            DataFrame with query results
        """
        if not self._connection:
            self.connect()
        
        try:
            # For non-SELECT queries (CREATE, INSERT, etc.), execute and return empty DataFrame
            query_upper = query.strip().upper()
            if not query_upper.startswith('SELECT'):
                from sqlalchemy import text
                with self._connection.connect() as conn:
                    conn.execute(text(query))
                    conn.commit()
                return pd.DataFrame()
            
            df = pd.read_sql(query, self._connection, **kwargs)
            return df
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test SQL connection."""
        try:
            self.connect()
            # Try simple query
            df = self.query("SELECT 1 as test")
            self.disconnect()
            return {
                'status': 'success',
                'message': 'Connection successful',
                'test_query_result': df.to_dict()
            }
        except Exception as e:
            return {
                'status': 'failed',
                'message': str(e)
            }
    
    def list_tables(self) -> List[str]:
        """List all tables in database."""
        if not self._connection:
            self.connect()
        
        try:
            from sqlalchemy import inspect
            inspector = inspect(self._connection)
            return inspector.get_table_names()
        except Exception as e:
            raise RuntimeError(f"Failed to list tables: {str(e)}")
    
    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        """Get schema information for a table."""
        if not self._connection:
            self.connect()
        
        try:
            from sqlalchemy import inspect
            inspector = inspect(self._connection)
            columns = inspector.get_columns(table_name)
            return pd.DataFrame(columns)
        except Exception as e:
            raise RuntimeError(f"Failed to get table schema: {str(e)}")


class BigQueryConnector(DataConnector):
    """Connector for Google BigQuery."""
    
    def connect(self) -> bool:
        """Establish BigQuery connection."""
        try:
            from google.cloud import bigquery
            from google.oauth2 import service_account
            
            if self.config.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.config.credentials_path
                )
                client = bigquery.Client(
                    credentials=credentials,
                    project=self.config.project_id
                )
            else:
                # Use default credentials
                client = bigquery.Client(project=self.config.project_id)
            
            self._connection = client
            return True
        except ImportError:
            raise ImportError("Google Cloud BigQuery not installed. Install with: pip install google-cloud-bigquery")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to BigQuery: {str(e)}")
    
    def disconnect(self) -> bool:
        """Close BigQuery connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
        return True
    
    def query(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute BigQuery query and return DataFrame.
        
        Args:
            query: SQL query string
            **kwargs: Additional parameters for query
            
        Returns:
            DataFrame with query results
        """
        if not self._connection:
            self.connect()
        
        try:
            query_job = self._connection.query(query)
            df = query_job.to_dataframe()
            return df
        except Exception as e:
            raise RuntimeError(f"BigQuery execution failed: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test BigQuery connection."""
        try:
            self.connect()
            # Try simple query
            df = self.query("SELECT 1 as test")
            self.disconnect()
            return {
                'status': 'success',
                'message': 'BigQuery connection successful',
                'project_id': self.config.project_id,
                'test_query_result': df.to_dict()
            }
        except Exception as e:
            return {
                'status': 'failed',
                'message': str(e)
            }
    
    def list_datasets(self) -> List[str]:
        """List all datasets in project."""
        if not self._connection:
            self.connect()
        
        try:
            datasets = list(self._connection.list_datasets())
            return [dataset.dataset_id for dataset in datasets]
        except Exception as e:
            raise RuntimeError(f"Failed to list datasets: {str(e)}")
    
    def list_tables(self, dataset_id: str) -> List[str]:
        """List all tables in a dataset."""
        if not self._connection:
            self.connect()
        
        try:
            tables = self._connection.list_tables(dataset_id)
            return [table.table_id for table in tables]
        except Exception as e:
            raise RuntimeError(f"Failed to list tables: {str(e)}")


class APIConnector(DataConnector):
    """Connector for REST APIs."""
    
    def connect(self) -> bool:
        """Prepare API connection (no persistent connection needed)."""
        try:
            import requests
            self._connection = requests.Session()
            
            # Set default headers
            if self.config.api_key:
                self._connection.headers.update({
                    'Authorization': f'Bearer {self.config.api_key}'
                })
            
            if self.config.additional_params:
                self._connection.headers.update(self.config.additional_params.get('headers', {}))
            
            return True
        except ImportError:
            raise ImportError("Requests library not installed. Install with: pip install requests")
    
    def disconnect(self) -> bool:
        """Close API session."""
        if self._connection:
            self._connection.close()
            self._connection = None
        return True
    
    def query(self, endpoint: str, method: str = 'GET', **kwargs) -> pd.DataFrame:
        """Execute API request and return DataFrame.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional parameters (params, json, data, etc.)
            
        Returns:
            DataFrame with API response data
        """
        if not self._connection:
            self.connect()
        
        try:
            url = f"{self.config.api_endpoint}/{endpoint}" if self.config.api_endpoint else endpoint
            
            response = self._connection.request(method, url, **kwargs)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find data array in response
                if 'data' in data and isinstance(data['data'], (list, dict)):
                    if isinstance(data['data'], list):
                        df = pd.DataFrame(data['data'])
                    else:
                        df = pd.DataFrame([data['data']])
                elif 'results' in data and isinstance(data['results'], list):
                    df = pd.DataFrame(data['results'])
                elif 'items' in data and isinstance(data['items'], list):
                    df = pd.DataFrame(data['items'])
                else:
                    # Single record - flatten nested dict
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"Unexpected API response type: {type(data)}")
            
            return df
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test API connection."""
        try:
            self.connect()
            # Try to make a simple request to a known endpoint
            test_url = self.config.api_endpoint if self.config.api_endpoint else 'https://httpbin.org/get'
            response = self._connection.get(test_url, timeout=5)
            self.disconnect()
            return {
                'status': 'success' if response.status_code == 200 else 'failed',
                'message': f'API connection test returned status {response.status_code}',
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'status': 'failed',
                'message': str(e)
            }


class DataSourceManager:
    """Manager for multiple data sources with unified interface."""
    
    def __init__(self):
        self.connectors: Dict[str, DataConnector] = {}
        self.active_sources: Dict[str, pd.DataFrame] = {}
    
    def register_source(self, name: str, config: ConnectionConfig) -> bool:
        """Register a new data source.
        
        Args:
            name: Unique name for this source
            config: Connection configuration
            
        Returns:
            True if registration successful
        """
        try:
            # Create appropriate connector
            if config.source_type in ['postgresql', 'mysql', 'sqlite', 'mssql']:
                connector = SQLConnector(config)
            elif config.source_type == 'bigquery':
                connector = BigQueryConnector(config)
            elif config.source_type == 'api':
                connector = APIConnector(config)
            else:
                raise ValueError(f"Unsupported source type: {config.source_type}")
            
            self.connectors[name] = connector
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to register source '{name}': {str(e)}")
    
    def load_data(self, source_name: str, query: str, cache: bool = True, **kwargs) -> pd.DataFrame:
        """Load data from registered source.
        
        Args:
            source_name: Name of registered source
            query: Query string (SQL, endpoint, etc.)
            cache: Whether to cache the result
            **kwargs: Additional parameters for query
            
        Returns:
            DataFrame with loaded data
        """
        if source_name not in self.connectors:
            raise ValueError(f"Source '{source_name}' not registered")
        
        connector = self.connectors[source_name]
        
        with connector:
            df = connector.query(query, **kwargs)
        
        if cache:
            self.active_sources[source_name] = df
        
        return df
    
    def get_cached_data(self, source_name: str) -> Optional[pd.DataFrame]:
        """Get cached data for a source."""
        return self.active_sources.get(source_name)
    
    def list_sources(self) -> List[Dict[str, Any]]:
        """List all registered sources."""
        return [
            {
                'name': name,
                'type': connector.config.source_type,
                'cached': name in self.active_sources
            }
            for name, connector in self.connectors.items()
        ]
    
    def test_all_connections(self) -> Dict[str, Dict[str, Any]]:
        """Test all registered connections."""
        results = {}
        for name, connector in self.connectors.items():
            results[name] = connector.test_connection()
        return results
    
    def refresh_data(self, source_name: str, query: str, **kwargs) -> pd.DataFrame:
        """Refresh cached data from source.
        
        Args:
            source_name: Name of registered source
            query: Query string
            **kwargs: Additional parameters
            
        Returns:
            Refreshed DataFrame
        """
        return self.load_data(source_name, query, cache=True, **kwargs)
    
    def clear_cache(self, source_name: Optional[str] = None):
        """Clear cached data.
        
        Args:
            source_name: Specific source to clear, or None to clear all
        """
        if source_name:
            self.active_sources.pop(source_name, None)
        else:
            self.active_sources.clear()
    
    def export_config(self, filepath: str, exclude_credentials: bool = True):
        """Export data source configurations to file.
        
        Args:
            filepath: Path to save configuration
            exclude_credentials: Whether to exclude sensitive credentials
        """
        configs = {}
        for name, connector in self.connectors.items():
            config_dict = {
                'source_type': connector.config.source_type,
                'host': connector.config.host,
                'port': connector.config.port,
                'database': connector.config.database,
                'api_endpoint': connector.config.api_endpoint,
                'project_id': connector.config.project_id
            }
            
            if not exclude_credentials:
                config_dict.update({
                    'username': connector.config.username,
                    'password': connector.config.password,
                    'api_key': connector.config.api_key,
                    'credentials_path': connector.config.credentials_path
                })
            
            configs[name] = config_dict
        
        with open(filepath, 'w') as f:
            json.dump(configs, f, indent=2)
    
    def import_config(self, filepath: str):
        """Import data source configurations from file.
        
        Args:
            filepath: Path to configuration file
        """
        with open(filepath, 'r') as f:
            configs = json.load(f)
        
        for name, config_dict in configs.items():
            config = ConnectionConfig(**config_dict)
            self.register_source(name, config)


# Convenience functions for quick setup
def create_postgres_connector(host: str, database: str, username: str, password: str, port: int = 5432) -> SQLConnector:
    """Create PostgreSQL connector with simplified parameters."""
    config = ConnectionConfig(
        source_type='postgresql',
        host=host,
        port=port,
        database=database,
        username=username,
        password=password
    )
    return SQLConnector(config)


def create_bigquery_connector(project_id: str, credentials_path: Optional[str] = None) -> BigQueryConnector:
    """Create BigQuery connector with simplified parameters."""
    config = ConnectionConfig(
        source_type='bigquery',
        project_id=project_id,
        credentials_path=credentials_path
    )
    return BigQueryConnector(config)


def create_api_connector(api_endpoint: str, api_key: Optional[str] = None, headers: Optional[Dict] = None) -> APIConnector:
    """Create API connector with simplified parameters."""
    config = ConnectionConfig(
        source_type='api',
        api_endpoint=api_endpoint,
        api_key=api_key,
        additional_params={'headers': headers} if headers else None
    )
    return APIConnector(config)
