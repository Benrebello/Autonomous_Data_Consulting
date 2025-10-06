# Execution Engine Documentation

## Overview

The `ExecutionEngine` class abstracts the complex task execution logic from `app.py`, providing a clean, testable, and reusable component for managing multi-step analysis workflows.

## Architecture

### Location
- **Module**: `optimizations.py`
- **Class**: `ExecutionEngine`

### Responsibilities
1. **Dependency Resolution**: Build and traverse task dependency graphs
2. **Task Execution**: Execute tools with proper parameter resolution
3. **Error Handling**: Retry failed tasks with configurable limits
4. **Cascade Invalidation**: Invalidate dependent tasks when prerequisites fail
5. **Metrics Tracking**: Record execution times and success rates
6. **Context Management**: Store results in shared context

## Usage

### Basic Example

```python
from optimizations import ExecutionEngine, get_metrics_collector

# Initialize engine
engine = ExecutionEngine(
    tool_mapping=pipeline.tool_mapping,
    shared_context=pipeline.shared_context,
    metrics_collector=get_metrics_collector()
)

# Execute plan
result = engine.execute_plan(
    plan=execution_plan,
    resolve_inputs_fn=pipeline._resolve_inputs,
    fill_defaults_fn=pipeline._fill_default_inputs_for_task,
    max_retries=1
)

# Check results
print(f"Completed: {len(result['completed'])}")
print(f"Failed: {len(result['failed'])}")
```

### Integration with app.py

The ExecutionEngine can replace the manual execution loop in `app.py`:

**Before (Manual Loop):**
```python
# Complex while loop with dependency checking
while len(completed_tasks) < len(tasks):
    for task in tasks:
        if dependencies_satisfied(task):
            result = execute_task(task)
            if result.success:
                completed_tasks.append(task)
            else:
                retry_or_fail(task)
```

**After (ExecutionEngine):**
```python
# Clean, declarative execution
engine = ExecutionEngine(self.tool_mapping, self.shared_context, self.metrics)
result = engine.execute_plan(plan, self._resolve_inputs, self._fill_default_inputs_for_task)
```

## Features

### 1. Dependency Graph

Automatically builds and validates task dependencies:

```python
# Task with dependencies
{
    "task_id": "task_3",
    "tool_to_use": "correlation_matrix",
    "dependencies": ["task_1", "task_2"],  # Must complete first
    "inputs": {...}
}
```

### 2. Retry Logic

Configurable retry attempts with error tracking:

```python
# Retry failed task once
result = engine.execute_plan(plan, ..., max_retries=1)

# No retries (fail fast)
result = engine.execute_plan(plan, ..., max_retries=0)
```

### 3. Cascade Invalidation

When a task fails after retries, all dependent tasks are invalidated:

```
Task A (success) → Task B (failed) → Task C (depends on B)
                                   ↓
                          Task C invalidated and removed from completed
```

### 4. Execution Log

Detailed log of all task executions:

```python
{
    'execution_log': [
        {
            'task_id': 'task_1',
            'tool': 'descriptive_stats',
            'status': 'success',
            'execution_time': 0.123,
            'output_variable': 'stats_result'
        },
        {
            'task_id': 'task_2',
            'tool': 'correlation_matrix',
            'status': 'error',
            'error': 'Requires at least 2 numeric columns'
        }
    ]
}
```

## API Reference

### ExecutionEngine

#### Constructor

```python
ExecutionEngine(
    tool_mapping: Dict[str, Callable],
    shared_context: Dict[str, Any],
    metrics_collector: Optional[MetricsCollector] = None
)
```

**Parameters:**
- `tool_mapping`: Dictionary mapping tool names to functions
- `shared_context`: Shared context for storing task results
- `metrics_collector`: Optional metrics collector (creates default if None)

#### Methods

##### execute_plan()

```python
execute_plan(
    plan: Dict[str, Any],
    resolve_inputs_fn: Callable,
    fill_defaults_fn: Callable,
    max_retries: int = 1
) -> Dict[str, Any]
```

Execute a complete execution plan with dependency management.

**Parameters:**
- `plan`: Execution plan dictionary with 'execution_plan' key
- `resolve_inputs_fn`: Function to resolve context references (e.g., `$stats_result`)
- `fill_defaults_fn`: Function to fill missing parameters with defaults
- `max_retries`: Maximum retry attempts per failed task

**Returns:**
```python
{
    'completed': ['task_1', 'task_2'],      # Successfully completed task IDs
    'failed': ['task_3'],                    # Failed task IDs (after retries)
    'results': {                             # Per-task results
        'task_1': {
            'status': 'success',
            'result': {...},
            'execution_time': 0.123
        }
    },
    'execution_log': [...]                   # Detailed execution log
}
```

## Benefits

### 1. Separation of Concerns
- **app.py**: Focuses on UI and high-level orchestration
- **ExecutionEngine**: Handles execution complexity
- **tool_registry.py**: Manages tool metadata and validation

### 2. Testability
```python
def test_execution_engine():
    # Mock dependencies
    tool_mapping = {'test_tool': lambda df: {'result': 'ok'}}
    context = {}
    
    # Create engine
    engine = ExecutionEngine(tool_mapping, context)
    
    # Test execution
    plan = {'execution_plan': [{'task_id': 't1', 'tool_to_use': 'test_tool'}]}
    result = engine.execute_plan(plan, lambda x: x, lambda n, i: i)
    
    assert len(result['completed']) == 1
```

### 3. Reusability
The ExecutionEngine can be used in:
- Web UI (app.py)
- CLI tools
- Batch processing scripts
- API endpoints
- Jupyter notebooks

### 4. Maintainability
- Single source of truth for execution logic
- Easy to add new features (e.g., parallel execution)
- Clear error handling patterns
- Comprehensive logging

## Advanced Features

### Parallel Execution (Future)

The ExecutionEngine can be extended to support parallel execution of independent tasks:

```python
def execute_plan_parallel(self, plan, ...):
    # Identify independent tasks
    batches = self._identify_independent_batches(plan)
    
    # Execute each batch in parallel
    for batch in batches:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._execute_single_task, task) 
                      for task in batch]
            results = [f.result() for f in futures]
```

### Conditional Execution

Support for conditional task execution based on previous results:

```python
{
    "task_id": "task_3",
    "tool_to_use": "advanced_analysis",
    "condition": "$task_1.result.has_outliers == True",
    "dependencies": ["task_1"]
}
```

### Progress Callbacks

Real-time progress updates:

```python
def on_task_complete(task_id, result):
    st.progress(completed / total)
    st.write(f"✅ {task_id} completed")

engine.execute_plan(plan, ..., on_progress=on_task_complete)
```

## Migration Guide

### Migrating from Manual Loop to ExecutionEngine

**Step 1**: Initialize engine in `__init__`:
```python
self.execution_engine = ExecutionEngine(
    self.tool_mapping, 
    self.shared_context, 
    self.metrics
)
```

**Step 2**: Replace execution loop:
```python
# Old: Manual loop (50+ lines)
while len(completed_tasks) < len(tasks):
    # Complex dependency checking
    # Manual retry logic
    # Error handling
    # ...

# New: Single method call
result = self.execution_engine.execute_plan(
    plan, 
    self._resolve_inputs, 
    self._fill_default_inputs_for_task
)
```

**Step 3**: Use results:
```python
completed = result['completed']
failed = result['failed']
execution_log = result['execution_log']
```

## Best Practices

### 1. Error Handling
Always check for errors in the result:
```python
result = engine.execute_plan(...)
if result.get('error'):
    handle_error(result['error'])
elif result['failed']:
    handle_failed_tasks(result['failed'])
```

### 2. Logging
Use the execution log for debugging:
```python
for entry in result['execution_log']:
    if entry['status'] == 'error':
        logger.error(f"Task {entry['task_id']} failed: {entry['error']}")
```

### 3. Metrics
Track execution metrics for optimization:
```python
metrics = engine.metrics
stats = metrics.get_tool_stats('correlation_matrix')
print(f"Success rate: {stats['success_rate']}")
print(f"Avg time: {stats['avg_time']}")
```

## Future Enhancements

- [ ] Parallel execution of independent tasks
- [ ] Conditional task execution
- [ ] Progress callbacks for real-time UI updates
- [ ] Task timeout management
- [ ] Resource usage tracking
- [ ] Execution plan optimization
- [ ] Automatic dependency inference
- [ ] Task result caching

## See Also

- [ARCHITECTURE.md](./ARCHITECTURE.md): Overall system architecture
- [OPTIMIZATIONS_IMPLEMENTED.md](./OPTIMIZATIONS_IMPLEMENTED.md): Performance optimizations
- [TESTING.md](./TESTING.md): Testing strategy
