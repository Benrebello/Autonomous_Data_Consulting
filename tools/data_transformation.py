# tools/data_transformation.py
"""Data transformation utilities (sorting, grouping, pivoting, normalization)."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Union


def normalize_data(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Normalize specified columns using StandardScaler.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    scaler = StandardScaler()
    df_scaled = df.copy()
    if columns:
        df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled


def sort_dataframe(df: pd.DataFrame, by: Union[str, List[str]], ascending: Union[bool, List[bool]] = True) -> pd.DataFrame:
    """Sort DataFrame by specified columns."""
    return df.sort_values(by=by, ascending=ascending)


def group_and_aggregate(df: pd.DataFrame, group_by: List[str], agg_dict: Dict[str, Any]) -> pd.DataFrame:
    """Group by columns and aggregate.
    
    Args:
        df: Input DataFrame
        group_by: Columns to group by
        agg_dict: Aggregations, e.g., {"col": "mean"} or {"col": ["mean", "sum"]}
    
    Returns:
        Aggregated DataFrame with index reset
    """
    return df.groupby(group_by).agg(agg_dict).reset_index()


def create_pivot_table(df: pd.DataFrame, index: Union[str, List[str]], columns: Union[str, List[str]],
                       values: Union[str, List[str]], aggfunc: Union[str, List[str], Dict[str, Any]] = 'mean') -> pd.DataFrame:
    """Create pivot table and reset index for usability."""
    return df.pivot_table(index=index, columns=columns, values=values, aggfunc=aggfunc).reset_index()


def join_datasets(df1: pd.DataFrame, df2: pd.DataFrame, on_column: str) -> pd.DataFrame:
    """Join two DataFrames using the same key column name on both sides.
    
    Args:
        df1: Left DataFrame
        df2: Right DataFrame
        on_column: Column name present in both frames to join on
    
    Returns:
        Merged DataFrame (inner join)
    """
    return pd.merge(df1, df2, on=on_column)


def join_datasets_on(df1: pd.DataFrame, df2: pd.DataFrame, left_on: str, right_on: str, how: str = 'inner') -> pd.DataFrame:
    """Join two DataFrames using different key column names.
    
    Args:
        df1: Left DataFrame
        df2: Right DataFrame
        left_on: Join key column on the left
        right_on: Join key column on the right
        how: Join type (inner, left, right, outer)
    
    Returns:
        Merged DataFrame
    """
    return pd.merge(df1, df2, left_on=left_on, right_on=right_on, how=how)


def compare_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
    """Compare two DataFrames: shapes and column overlaps.

    Returns a dict summarizing shapes and column differences/overlaps.
    """
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    return {
        'shape_df1': list(df1.shape),
        'shape_df2': list(df2.shape),
        'common_columns': sorted(list(cols1 & cols2)),
        'only_in_df1': sorted(list(cols1 - cols2)),
        'only_in_df2': sorted(list(cols2 - cols1)),
    }


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to snake_case ascii without special chars.

    - Lowercases
    - Removes accents
    - Replaces spaces and special chars with underscores
    - Removes consecutive underscores
    """
    import re
    import unicodedata
    
    def normalize_name(name: str) -> str:
        # Remove accents
        name = unicodedata.normalize('NFKD', name)
        name = name.encode('ascii', 'ignore').decode('ascii')
        # Lowercase and replace spaces/special with underscore
        name = name.lower()
        name = re.sub(r'[^a-z0-9]+', '_', name)
        # Remove leading/trailing underscores and collapse consecutive
        name = re.sub(r'_+', '_', name).strip('_')
        return name or 'col'
    
    df_result = df.copy()
    df_result.columns = [normalize_name(str(c)) for c in df_result.columns]
    return df_result
