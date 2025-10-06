# INTEGRATION_EXAMPLE.py
"""
Example of how to integrate the new refactored modules into existing code.

This file demonstrates:
1. Using AppState for state management
2. Using tool_registry for dynamic tool discovery
3. Importing from modular tools package
4. Backward compatibility
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any

# ============================================================================
# EXAMPLE 1: State Management Integration
# ============================================================================

def example_state_management():
    """Example of using AppState instead of scattered st.session_state."""
    
    # OLD WAY (still works, but not recommended)
    # if 'discovery_active' not in st.session_state:
    #     st.session_state['discovery_active'] = False
    # st.session_state['last_tool'] = 'correlation_matrix'
    
    # NEW WAY (recommended)
    from state import AppState
    
    state = AppState.get()
    
    # Read state
    if state.discovery_active:
        current_step = state.discovery_step
        print(f"Discovery is active at step {current_step}")
    
    # Modify state
    state.last_tool = 'correlation_matrix'
    state.last_objective = 'Analyze correlations between variables'
    
    # Add to collections
    state.add_to_memory({
        'timestamp': '2025-10-03',
        'action': 'correlation_analysis',
        'result': 'Found 5 significant correlations'
    })
    
    state.add_to_chat_history('user', 'What are the correlations?')
    state.add_to_chat_history('assistant', 'Here are the top correlations...')
    
    # Save state
    state.save()
    
    print("✅ State management example completed")


# ============================================================================
# EXAMPLE 2: Tool Registry Integration
# ============================================================================

def example_tool_registry():
    """Example of using tool_registry for dynamic tool management."""
    
    from tool_registry import (
        get_tool_function,
        get_tool_defaults,
        get_available_tools,
        get_tools_by_category,
        TOOL_REGISTRY
    )
    
    # Create sample data
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000],
        'department': ['IT', 'HR', 'IT', 'Sales', 'HR']
    })
    
    # 1. Get available tools
    all_tools = get_available_tools()
    print(f"📋 Available tools: {len(all_tools)}")
    print(f"   First 5: {all_tools[:5]}")
    
    # 2. Get tools by category
    viz_tools = get_tools_by_category('visualization')
    print(f"📊 Visualization tools: {viz_tools}")
    
    # 3. Get tool function dynamically
    tool_name = 'correlation_matrix'
    tool_func = get_tool_function(tool_name)
    if tool_func:
        result = tool_func(df)
        print(f"✅ Executed {tool_name}")
    
    # 4. Get defaults for a tool
    defaults = get_tool_defaults('plot_histogram', df)
    print(f"📝 Defaults for plot_histogram: {defaults}")
    
    # 5. Check if tool can execute
    metadata = TOOL_REGISTRY.get('linear_regression')
    if metadata:
        can_run, reason = metadata.can_execute(df)
        print(f"🔍 Can run linear_regression? {can_run}")
        if not can_run:
            print(f"   Reason: {reason}")
    
    print("✅ Tool registry example completed")


# ============================================================================
# EXAMPLE 3: Modular Tools Import
# ============================================================================

def example_modular_imports():
    """Example of importing from modular tools package."""
    
    # Create sample data
    df = pd.DataFrame({
        'value': [10, 20, 15, 100, 18, 22, 19],  # 100 is outlier
        'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A']
    })
    
    # Method 1: Import from specific module
    from tools.outlier_detection import detect_outliers, get_outliers_summary
    from tools.statistical_tests import perform_t_test
    from tools.data_profiling import descriptive_stats
    
    # Detect outliers
    df_with_outliers = detect_outliers(df, 'value', method='iqr')
    outlier_count = df_with_outliers['outlier'].sum()
    print(f"🔍 Found {outlier_count} outliers")
    
    # Get outlier summary
    summary = get_outliers_summary(df, 'value')
    print(f"📊 Outlier summary: {summary['outlier_percentage']:.1f}%")
    
    # Descriptive stats
    stats = descriptive_stats(df)
    print(f"📈 Data shape: {stats['shape']}")
    
    # Method 2: Import from tools package (backward compatible)
    from tools import correlation_matrix, clean_data
    
    print("✅ Modular imports example completed")


# ============================================================================
# EXAMPLE 4: Refactored _fill_default_inputs_for_task
# ============================================================================

def fill_default_inputs_for_task_OLD(tool: str, inputs: dict, df: pd.DataFrame) -> dict:
    """OLD implementation (200+ lines of if/elif)."""
    
    numeric_cols = list(df.select_dtypes(include='number').columns)
    
    if tool == 'descriptive_stats':
        inputs = {'df': df}
    elif tool == 'plot_histogram':
        inputs = {'df': df, 'column': numeric_cols[0] if numeric_cols else df.columns[0]}
    elif tool == 'detect_outliers':
        inputs = {'df': df, 'column': numeric_cols[0] if numeric_cols else df.columns[0], 'method': 'iqr'}
    elif tool == 'correlation_matrix':
        inputs = {'df': df}
    # ... 50+ more elif statements
    
    return inputs


def fill_default_inputs_for_task_NEW(tool: str, inputs: dict, df: pd.DataFrame) -> dict:
    """NEW implementation (5 lines using registry)."""
    
    from tool_registry import get_tool_defaults
    
    if df is None or df.empty:
        return inputs
    
    return get_tool_defaults(tool, df) if tool else inputs


def example_refactored_defaults():
    """Example comparing old vs new default filling."""
    
    df = pd.DataFrame({
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]
    })
    
    # Old way
    old_defaults = fill_default_inputs_for_task_OLD('plot_histogram', {}, df)
    print(f"📝 Old way defaults: {old_defaults}")
    
    # New way
    new_defaults = fill_default_inputs_for_task_NEW('plot_histogram', {}, df)
    print(f"✨ New way defaults: {new_defaults}")
    
    print("✅ Refactored defaults example completed")


# ============================================================================
# EXAMPLE 5: Dynamic Tool List Generation
# ============================================================================

def example_dynamic_tool_list():
    """Example of generating tool list dynamically for prompts."""
    
    from tool_registry import get_available_tools
    
    # OLD WAY (hardcoded in prompts.py)
    old_tools_list = "clean_data | descriptive_stats | detect_outliers | correlation_matrix | ..."
    
    # NEW WAY (generated dynamically)
    new_tools_list = " | ".join(get_available_tools())
    
    print("📋 OLD tool list (hardcoded):")
    print(f"   {old_tools_list}")
    print()
    print("✨ NEW tool list (dynamic):")
    print(f"   {new_tools_list}")
    
    # Use in prompt
    prompt_template = """
    You are a data analyst. Choose one of these tools:
    {tools_list}
    """
    
    formatted_prompt = prompt_template.format(tools_list=new_tools_list)
    print()
    print("📝 Formatted prompt:")
    print(formatted_prompt)
    
    print("✅ Dynamic tool list example completed")


# ============================================================================
# EXAMPLE 6: Complete Integration in AnalysisPipeline
# ============================================================================

class AnalysisPipelineRefactored:
    """Example of refactored AnalysisPipeline using new modules."""
    
    def __init__(self, dataframes: Dict[str, pd.DataFrame]):
        from tool_registry import get_available_tools, TOOL_REGISTRY
        from state import AppState
        
        self.dataframes = dataframes
        self.state = AppState.get()
        
        # Dynamic tool list (always up-to-date)
        self.available_tools = get_available_tools()
        self.available_tools_str = " | ".join(self.available_tools)
        
        # Tool registry for metadata
        self.tool_registry = TOOL_REGISTRY
        
        print(f"✅ Pipeline initialized with {len(self.available_tools)} tools")
    
    def execute_tool(self, tool_name: str, custom_inputs: dict = None):
        """Execute a tool with automatic default filling."""
        from tool_registry import get_tool_function, get_tool_defaults
        
        # Get tool function
        tool_func = get_tool_function(tool_name)
        if not tool_func:
            return {"error": f"Tool {tool_name} not found"}
        
        # Get default DataFrame
        df = next(iter(self.dataframes.values())) if self.dataframes else None
        
        # Fill defaults if no custom inputs
        if custom_inputs:
            inputs = custom_inputs
        else:
            inputs = get_tool_defaults(tool_name, df)
        
        # Execute tool
        try:
            result = tool_func(**inputs)
            
            # Update state
            self.state.last_tool = tool_name
            self.state.last_result = result
            self.state.save()
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def get_tool_info(self, tool_name: str):
        """Get information about a tool."""
        metadata = self.tool_registry.get(tool_name)
        if not metadata:
            return None
        
        return {
            'name': tool_name,
            'description': metadata.description,
            'category': metadata.category,
            'requires_numeric': metadata.requires_numeric,
            'requires_categorical': metadata.requires_categorical,
            'min_rows': metadata.min_rows
        }


def example_complete_integration():
    """Example of complete integration."""
    
    # Create sample data
    df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'salary': [50000, 60000, 70000, 80000, 90000],
        'department': ['IT', 'HR', 'IT', 'Sales', 'HR']
    })
    
    # Initialize refactored pipeline
    pipeline = AnalysisPipelineRefactored({'main': df})
    
    # Execute tool with automatic defaults
    result = pipeline.execute_tool('correlation_matrix')
    print(f"📊 Correlation result: {list(result.keys())}")
    
    # Get tool info
    info = pipeline.get_tool_info('linear_regression')
    print(f"ℹ️  Tool info: {info}")
    
    print("✅ Complete integration example completed")


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("INTEGRATION EXAMPLES - Refactored Architecture")
    print("=" * 80)
    print()
    
    print("1️⃣  State Management")
    print("-" * 80)
    example_state_management()
    print()
    
    print("2️⃣  Tool Registry")
    print("-" * 80)
    example_tool_registry()
    print()
    
    print("3️⃣  Modular Imports")
    print("-" * 80)
    example_modular_imports()
    print()
    
    print("4️⃣  Refactored Defaults")
    print("-" * 80)
    example_refactored_defaults()
    print()
    
    print("5️⃣  Dynamic Tool List")
    print("-" * 80)
    example_dynamic_tool_list()
    print()
    
    print("6️⃣  Complete Integration")
    print("-" * 80)
    example_complete_integration()
    print()
    
    print("=" * 80)
    print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 80)
