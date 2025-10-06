# optimizations.py
"""System optimizations: parallelization, metrics, validation, recommendations."""

import pandas as pd
import numpy as np
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
import streamlit as st



# ============================================================================
# 1. PARALELIZAÇÃO DE TAREFAS
# ============================================================================

class ParallelExecutor:
    """Execute independent tasks in parallel."""
    
    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
    
    def identify_independent_tasks(self, tasks: List[Dict]) -> List[List[Dict]]:
        """Group tasks into batches of independent tasks."""
        batches = []
        executed_ids = set()
        
        while len(executed_ids) < len(tasks):
            batch = []
            for task in tasks:
                task_id = task.get('task_id')
                if task_id in executed_ids:
                    continue
                
                # Check if all dependencies are satisfied
                dependencies = task.get('dependencies', [])
                if all(dep in executed_ids for dep in dependencies):
                    batch.append(task)
            
            if not batch:
                # No more independent tasks, break to avoid infinite loop
                break
            
            batches.append(batch)
            executed_ids.update(t.get('task_id') for t in batch)
        
        return batches
    
    def execute_batch_parallel(
        self, 
        batch: List[Dict], 
        executor_func: Callable,
        on_progress: Optional[Callable] = None
    ) -> Dict[int, Any]:
        """Execute a batch of tasks in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=min(len(batch), self.max_workers)) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(executor_func, task): task 
                for task in batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_id = task.get('task_id')
                
                try:
                    result = future.result()
                    results[task_id] = {
                        'success': True,
                        'result': result,
                        'task': task
                    }
                    
                    if on_progress:
                        on_progress(task_id, result, None)
                        
                except Exception as e:
                    results[task_id] = {
                        'success': False,
                        'error': str(e),
                        'task': task
                    }
                    
                    if on_progress:
                        on_progress(task_id, None, str(e))
        
        return results


# ============================================================================
# 2. COMPRESSÃO DE CONTEXTO
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate number of tokens in text (~4 chars per token)."""
    return len(str(text)) // 4


def compress_large_result(result: Any, max_tokens: int = 500) -> Dict[str, Any]:
    """Compress large results to save tokens."""
    result_str = str(result)
    estimated = estimate_tokens(result_str)
    
    if estimated <= max_tokens:
        return {
            'compressed': False,
            'content': result,
            'tokens': estimated
        }
    
    # Extract key information based on type
    if isinstance(result, dict):
        # Keep only most important keys
        summary = {}
        priority_keys = ['summary', 'conclusion', 'top_5', 'significant', 'important']
        
        for key in priority_keys:
            if key in result:
                summary[key] = result[key]
        
        # Add sample of other keys
        other_keys = [k for k in result.keys() if k not in priority_keys][:3]
        for key in other_keys:
            summary[key] = result[key]
        
        summary['_note'] = f'Compressed from {len(result)} keys'
        
    elif isinstance(result, (list, tuple)):
        # Keep first and last items
        if len(result) > 10:
            summary = {
                'first_5': result[:5],
                'last_5': result[-5:],
                '_note': f'Compressed from {len(result)} items'
            }
        else:
            summary = result
    
    else:
        # Truncate string
        summary = result_str[:max_tokens * 4] + '... (truncated)'
    
    return {
        'compressed': True,
        'content': summary,
        'original_tokens': estimated,
        'compressed_tokens': estimate_tokens(str(summary)),
        'savings': estimated - estimate_tokens(str(summary))
    }


def compress_execution_context(execution_results: Dict, max_tokens_per_task: int = 300) -> Dict:
    """Compress entire execution context."""
    compressed = {
        'tasks': [],
        'compression_stats': {
            'original_tokens': 0,
            'compressed_tokens': 0,
            'tasks_compressed': 0
        }
    }
    
    for task_result in execution_results.get('tasks', []):
        result = task_result.get('result')
        
        if result:
            compressed_result = compress_large_result(result, max_tokens_per_task)
            
            compressed['tasks'].append({
                'task_id': task_result.get('task_id'),
                'tool_used': task_result.get('tool_used'),
                'result': compressed_result['content'],
                'compressed': compressed_result['compressed']
            })
            
            if compressed_result['compressed']:
                compressed['compression_stats']['tasks_compressed'] += 1
                compressed['compression_stats']['original_tokens'] += compressed_result['original_tokens']
                compressed['compression_stats']['compressed_tokens'] += compressed_result['compressed_tokens']
        else:
            compressed['tasks'].append(task_result)
    
    return compressed


# ============================================================================
# 3. MÉTRICAS E OBSERVABILIDADE
# ============================================================================

class MetricsCollector:
    """Collect and analyze system metrics."""
    
    def __init__(self):
        self.metrics = []
        self.task_durations = defaultdict(list)
        self.task_errors = defaultdict(int)
        self.task_success = defaultdict(int)
    
    def track_task_execution(
        self, 
        task_name: str, 
        duration: float, 
        success: bool,
        error: Optional[str] = None
    ):
        """Track execution of a task."""
        metric = {
            'task': task_name,
            'duration_ms': duration * 1000,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics.append(metric)
        self.task_durations[task_name].append(duration)
        
        if success:
            self.task_success[task_name] += 1
        else:
            self.task_errors[task_name] += 1
    
    def record_tool_execution(
        self,
        tool_name: str,
        success: bool,
        duration: float,
        inputs: Dict[str, Any],
        error: Optional[str] = None
    ):
        """Record tool execution with detailed parameters.
        
        Args:
            tool_name: Name of the tool executed
            success: Whether execution succeeded
            duration: Execution duration in seconds
            inputs: Input parameters used
            error: Error message if failed
        """
        self.track_task_execution(tool_name, duration, success, error)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        if not self.metrics:
            return {'message': 'No metrics collected yet'}
        
        report = {
            'total_tasks': len(self.metrics),
            'success_rate': sum(1 for m in self.metrics if m['success']) / len(self.metrics),
            'slowest_tasks': [],
            'most_errors': [],
            'average_durations': {}
        }
        
        # Slowest tasks
        avg_durations = {
            task: np.mean(durations) 
            for task, durations in self.task_durations.items()
        }
        report['slowest_tasks'] = sorted(
            avg_durations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Most errors
        report['most_errors'] = sorted(
            self.task_errors.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Average durations
        report['average_durations'] = {
            task: f"{np.mean(durations)*1000:.0f}ms"
            for task, durations in self.task_durations.items()
        }
        
        return report
    
    def display_metrics_sidebar(self):
        """Display metrics in Streamlit sidebar."""
        if not self.metrics:
            return
        
        with st.sidebar.expander("Métricas de Performance", expanded=False):
            report = self.get_performance_report()
            
            st.metric("Taxa de Sucesso", f"{report['success_rate']*100:.1f}%")
            st.metric("Total de Tarefas", report['total_tasks'])
            
            if report['slowest_tasks']:
                st.markdown("**Tarefas Mais Lentas:**")
                for task, duration in report['slowest_tasks'][:3]:
                    st.text(f"  • {task}: {duration*1000:.0f}ms")


# ============================================================================
# 4. VALIDAÇÃO PREDITIVA
# ============================================================================

def validate_query_feasibility(query: str, dataframes: Dict[str, pd.DataFrame]) -> List[str]:
    """Validate if query is feasible with available data."""
    issues = []
    query_lower = query.lower()
    
    # Combine all dataframes for analysis
    if not dataframes:
        return ["Nenhum dataset carregado"]
    
    # Get first dataframe for analysis (or combine if needed)
    df = list(dataframes.values())[0]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Correlation analysis
    if any(word in query_lower for word in ['correlação', 'correlation', 'relação']):
        if len(numeric_cols) < 2:
            issues.append(
                f"Análise de correlação requer pelo menos 2 colunas numéricas. "
                f"Encontradas: {len(numeric_cols)}"
            )
    
    # Time series analysis
    if any(word in query_lower for word in ['temporal', 'série', 'previsão', 'forecast']):
        if len(date_cols) == 0:
            issues.append(
                "Análise temporal requer coluna de data/hora. "
                "Nenhuma coluna datetime detectada."
            )
        if len(df) < 30:
            issues.append(
                f"Análise temporal requer pelo menos 30 observações. "
                f"Dataset tem apenas {len(df)} linhas."
            )
    
    # Machine Learning
    if any(word in query_lower for word in ['modelo', 'predição', 'classificação', 'regressão']):
        if len(df) < 50:
            issues.append(
                f"Modelagem ML requer pelo menos 50 observações. "
                f"Dataset tem apenas {len(df)} linhas."
            )
    
    # Clustering
    if any(word in query_lower for word in ['cluster', 'agrupamento', 'segmentação']):
        if len(numeric_cols) < 2:
            issues.append(
                f"Clustering requer pelo menos 2 variáveis numéricas. "
                f"Encontradas: {len(numeric_cols)}"
            )
    
    # Group comparison
    if any(word in query_lower for word in ['comparar', 'diferença', 'grupos']):
        if len(categorical_cols) == 0:
            issues.append(
                "Comparação de grupos requer variáveis categóricas. "
                "Nenhuma coluna categórica detectada."
            )
    
    return issues


# ============================================================================
# 5. RECOMENDAÇÕES PROATIVAS
# ============================================================================

def suggest_analyses(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Suggest relevant analyses based on data characteristics."""
    suggestions = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Time series analysis
    if date_cols:
        suggestions.append({
            'type': 'time_series',
            'priority': 'high',
            'title': 'Análise Temporal Detectada',
            'description': f'Coluna de data encontrada: {date_cols[0]}',
            'suggested_query': f'Faça uma análise temporal de {numeric_cols[0] if numeric_cols else "dados"} ao longo do tempo',
            'tools': ['decompose_time_series', 'plot_line_chart', 'forecast_arima']
        })
    
    # Correlation analysis
    if len(numeric_cols) >= 2:
        suggestions.append({
            'type': 'correlation',
            'priority': 'high',
            'title': 'Análise de Correlações',
            'description': f'{len(numeric_cols)} variáveis numéricas disponíveis',
            'suggested_query': 'Quais são as correlações entre as variáveis numéricas?',
            'tools': ['correlation_matrix', 'plot_heatmap', 'correlation_tests']
        })
    
    # Group comparison
    if categorical_cols and numeric_cols:
        suggestions.append({
            'type': 'group_comparison',
            'priority': 'medium',
            'title': 'Comparação Entre Grupos',
            'description': f'Comparar {numeric_cols[0]} entre grupos de {categorical_cols[0]}',
            'suggested_query': f'Compare {numeric_cols[0]} entre diferentes {categorical_cols[0]}',
            'tools': ['perform_anova', 'plot_boxplot', 'perform_t_test']
        })
    
    # Outlier detection
    if numeric_cols:
        suggestions.append({
            'type': 'outliers',
            'priority': 'medium',
            'title': 'Detecção de Outliers',
            'description': 'Identificar valores atípicos nos dados',
            'suggested_query': f'Existem outliers em {numeric_cols[0]}?',
            'tools': ['detect_outliers', 'plot_boxplot']
        })
    
    # Clustering
    if len(numeric_cols) >= 2 and len(df) >= 50:
        suggestions.append({
            'type': 'clustering',
            'priority': 'low',
            'title': 'Segmentação de Dados',
            'description': 'Agrupar dados em clusters similares',
            'suggested_query': 'Agrupe os dados em clusters baseado nas características',
            'tools': ['run_kmeans_clustering', 'plot_scatter']
        })
    
    # Data quality
    if df.isna().sum().sum() > 0:
        missing_pct = (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        suggestions.append({
            'type': 'data_quality',
            'priority': 'high',
            'title': 'Análise de Qualidade',
            'description': f'{missing_pct:.1f}% de dados faltantes detectados',
            'suggested_query': 'Analise a qualidade dos dados e dados faltantes',
            'tools': ['data_profiling', 'missing_data_analysis']
        })
    
    return suggestions


def display_recommendations(suggestions: List[Dict]):
    """Display recommendations in Streamlit."""
    if not suggestions:
        return
    
    st.markdown("### Análises Sugeridas")
    st.info("Baseado nos seus dados, sugerimos as seguintes análises:")
    
    for suggestion in suggestions[:3]:  # Show top 3
        priority_emoji = {
            'high': '🔴',
            'medium': '🟡',
            'low': '🟢'
        }
        
        emoji = priority_emoji.get(suggestion['priority'], '⚪')
        
        with st.expander(f"{emoji} {suggestion['title']}", expanded=False):
            st.markdown(f"**Descrição:** {suggestion['description']}")
            st.markdown(f"**Query sugerida:** _{suggestion['suggested_query']}_")
            
            if st.button(f"Usar esta análise", key=f"suggest_{suggestion['type']}"):
                st.session_state['suggested_query'] = suggestion['suggested_query']
                st.rerun()


# ============================================================================
# 6. MODO EXPLAIN
# ============================================================================

TOOL_EXPLANATIONS = {
    'correlation_matrix': 'Escolhido para analisar relações entre variáveis numéricas',
    'perform_t_test': 'Escolhido para comparar médias de dois grupos estatisticamente',
    'perform_anova': 'Escolhido para comparar médias de 3 ou mais grupos',
    'random_forest_classifier': 'Escolhido por ser robusto e lidar bem com não-linearidades',
    'linear_regression': 'Escolhido para modelar relação linear entre variáveis',
    'detect_outliers': 'Escolhido para identificar valores atípicos nos dados',
    'plot_histogram': 'Escolhido para visualizar distribuição de uma variável',
    'plot_scatter': 'Escolhido para visualizar relação entre duas variáveis',
    'run_kmeans_clustering': 'Escolhido para agrupar dados em clusters similares',
    'decompose_time_series': 'Escolhido para separar tendência, sazonalidade e ruído',
    'data_profiling': 'Escolhido para análise completa de qualidade dos dados',
    'missing_data_analysis': 'Escolhido para entender padrões de dados faltantes',
}


def explain_tool_choice(tool_name: str, context: Optional[str] = None) -> str:
    """Explain why a tool was chosen."""
    explanation = TOOL_EXPLANATIONS.get(
        tool_name,
        f'Ferramenta escolhida para executar: {tool_name}'
    )
    
    if context:
        explanation += f"\n\nContexto: {context}"
    
    return explanation


def display_execution_explanations(plan: Dict):
    """Display explanations for tool choices."""
    with st.expander("Por que essas ferramentas?", expanded=False):
        st.markdown("**Explicação das Escolhas:**")
        
        for task in plan.get('execution_plan', [])[:5]:
            tool = task.get('tool_to_use')
            description = task.get('description')
            
            explanation = explain_tool_choice(tool, description)
            st.markdown(f"**{tool}:** {explanation}")


# Global metrics collector instance
_metrics_collector = None

def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# ============================================================================
# EXECUTION ENGINE - Abstracts task execution logic from app.py
# ============================================================================

class ExecutionEngine:
    """Manages task execution with dependency resolution, retries, and error handling.
    
    This class encapsulates the complex execution logic previously embedded in app.py,
    making it reusable, testable, and maintainable.
    """
    
    def __init__(self, tool_mapping: Dict[str, Callable], shared_context: Dict[str, Any],
                 metrics_collector: Optional[MetricsCollector] = None):
        """Initialize execution engine.
        
        Args:
            tool_mapping: Dictionary mapping tool names to functions
            shared_context: Shared context for storing results
            metrics_collector: Optional metrics collector for tracking
        """
        self.tool_mapping = tool_mapping
        self.shared_context = shared_context
        self.metrics = metrics_collector or get_metrics_collector()
        self.execution_log = []
    
    def execute_plan(self, plan: Dict[str, Any], resolve_inputs_fn: Callable,
                     fill_defaults_fn: Callable, max_retries: int = 1) -> Dict[str, Any]:
        """Execute a complete execution plan with dependency management.
        
        Args:
            plan: Execution plan with tasks
            resolve_inputs_fn: Function to resolve input references
            fill_defaults_fn: Function to fill default parameters
            max_retries: Maximum number of retries per task
        
        Returns:
            Dictionary with execution results and statistics
        """
        tasks = plan.get('execution_plan', [])
        if not tasks:
            return {'error': 'No tasks in plan', 'completed': [], 'failed': []}
        
        completed_tasks = []
        failed_tasks = []
        task_results = {}
        retry_count = {}
        
        # Build dependency graph
        dependencies = self._build_dependency_graph(tasks)
        
        # Execute tasks respecting dependencies
        max_iterations = len(tasks) * (max_retries + 1)
        iteration = 0
        
        while len(completed_tasks) + len(failed_tasks) < len(tasks) and iteration < max_iterations:
            iteration += 1
            made_progress = False
            
            for task in tasks:
                task_id = task.get('task_id', f"task_{tasks.index(task)}")
                
                # Skip if already processed
                if task_id in completed_tasks or task_id in failed_tasks:
                    continue
                
                # Check if dependencies are satisfied
                deps = dependencies.get(task_id, [])
                if not all(dep in completed_tasks for dep in deps):
                    continue
                
                # Execute task
                result = self._execute_single_task(
                    task, resolve_inputs_fn, fill_defaults_fn
                )
                
                if result.get('status') == 'success':
                    completed_tasks.append(task_id)
                    task_results[task_id] = result
                    made_progress = True
                else:
                    # Check retry limit
                    retry_count[task_id] = retry_count.get(task_id, 0) + 1
                    if retry_count[task_id] > max_retries:
                        failed_tasks.append(task_id)
                        task_results[task_id] = result
                        made_progress = True
                        
                        # Invalidate dependent tasks
                        self._invalidate_dependents(task_id, dependencies, completed_tasks)
            
            if not made_progress:
                break
        
        return {
            'completed': completed_tasks,
            'failed': failed_tasks,
            'results': task_results,
            'execution_log': self.execution_log
        }
    
    def _build_dependency_graph(self, tasks: List[Dict]) -> Dict[str, List[str]]:
        """Build dependency graph from tasks.
        
        Args:
            tasks: List of task dictionaries
        
        Returns:
            Dictionary mapping task_id to list of dependency task_ids
        """
        graph = {}
        for task in tasks:
            task_id = task.get('task_id', f"task_{tasks.index(task)}")
            deps = task.get('dependencies', [])
            graph[task_id] = deps if isinstance(deps, list) else []
        return graph
    
    def _execute_single_task(self, task: Dict, resolve_inputs_fn: Callable,
                            fill_defaults_fn: Callable) -> Dict[str, Any]:
        """Execute a single task.
        
        Args:
            task: Task dictionary
            resolve_inputs_fn: Function to resolve input references
            fill_defaults_fn: Function to fill defaults
        
        Returns:
            Result dictionary with status and output
        """
        tool_name = task.get('tool_to_use')
        task_id = task.get('task_id', 'unknown')
        
        if not tool_name or tool_name not in self.tool_mapping:
            return {
                'status': 'error',
                'error': f'Tool {tool_name} not found',
                'task_id': task_id
            }
        
        try:
            # Get tool function
            tool_fn = self.tool_mapping[tool_name]
            
            # Fill defaults and resolve inputs
            inputs = task.get('inputs', {})
            inputs = fill_defaults_fn(tool_name, inputs)
            resolved_inputs = resolve_inputs_fn(inputs)
            
            # Execute tool
            start_time = time.time()
            result = tool_fn(**resolved_inputs)
            execution_time = time.time() - start_time
            
            # Store result in shared context
            output_var = task.get('output_variable', f"{tool_name}_result")
            self.shared_context[output_var] = result
            
            # Log execution
            self.execution_log.append({
                'task_id': task_id,
                'tool': tool_name,
                'status': 'success',
                'execution_time': execution_time,
                'output_variable': output_var
            })
            
            # Track metrics
            self.metrics.record_tool_execution(tool_name, True, execution_time, resolved_inputs)
            
            return {
                'status': 'success',
                'result': result,
                'task_id': task_id,
                'output_variable': output_var,
                'execution_time': execution_time
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Log error
            self.execution_log.append({
                'task_id': task_id,
                'tool': tool_name,
                'status': 'error',
                'error': error_msg
            })
            
            # Track metrics
            self.metrics.record_tool_execution(tool_name, False, 0, {}, error_msg)
            
            return {
                'status': 'error',
                'error': error_msg,
                'task_id': task_id,
                'tool': tool_name
            }
    
    def _invalidate_dependents(self, failed_task_id: str, dependencies: Dict[str, List[str]],
                               completed_tasks: List[str]) -> None:
        """Invalidate tasks that depend on a failed task.
        
        Args:
            failed_task_id: ID of the failed task
            dependencies: Dependency graph
            completed_tasks: List of completed task IDs (modified in place)
        """
        # Find all tasks that depend on the failed task
        dependents = [tid for tid, deps in dependencies.items() if failed_task_id in deps]
        
        for dependent_id in dependents:
            if dependent_id in completed_tasks:
                completed_tasks.remove(dependent_id)
                # Recursively invalidate
                self._invalidate_dependents(dependent_id, dependencies, completed_tasks)
