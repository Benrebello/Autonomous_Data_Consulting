# agents.py
import json
import time
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List, Optional, Literal, Callable
import sys
from pathlib import Path

# Ensure local module resolution for 'prompts.py' to avoid shadowing by similarly named external packages
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import (ORCHESTRATOR_PROMPT, TEAM_LEADER_PROMPT, 
                     SYNTHESIS_PROMPT, FINAL_RESPONSE_PROMPT)
from prompt_templates import (get_orchestrator_prompt, get_team_leader_plan_prompt,
                              get_synthesis_prompt, get_final_response_prompt)
from rate_limiter import RateLimiter

# Global for rate limiting (legacy, kept for compatibility)
last_call_time = 0

def clean_json_response(response: str) -> str:
    """Attempt to extract a valid JSON object from a model response.

    Handles cases like:
    - Prefixed text (e.g., "Plano em JSON:")
    - Markdown fences ```json ... ```
    - Trailing commentary after the JSON
    Returns the JSON substring if a balanced object is found; otherwise returns the trimmed original.
    """
    s = response.strip()
    # Remove markdown fences if present anywhere
    if '```' in s:
        parts = s.split('```')
        # Try to find the fenced code part that starts with optional 'json' line
        for i in range(len(parts)-1):
            block = parts[i+1]
            if block.lstrip().startswith('json'):
                candidate = block.lstrip()[4:]  # remove 'json'
                s = candidate.strip()
                break
    # Find first balanced JSON object
    start = s.find('{')
    if start == -1:
        return s
    # Walk and balance braces, taking care of strings
    depth = 0
    in_str = False
    esc = False
    end_index = None
    for idx in range(start, len(s)):
        ch = s[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_index = idx
                    break
    if end_index is not None and end_index >= start:
        return s[start:end_index+1].strip()
    # Fallback: return trimmed string
    return s

class BaseAgent:
    def __init__(self, llm, rpm_limit=10, rate_limiter: Optional[RateLimiter] = None):
        self.llm = llm
        self.rpm_limit = rpm_limit
        self.rate_limiter = rate_limiter or RateLimiter(rpm_limit=rpm_limit)
        self.wait_callback: Optional[Callable[[dict], None]] = None
    
    def set_wait_callback(self, callback: Callable[[dict], None]):
        """Set callback to be called when waiting for rate limit."""
        self.wait_callback = callback


class BriefingModel(BaseModel):
    """Structured representation of the Orchestrator's project briefing."""
    user_query: str
    main_goal: str
    key_questions: List[str]
    main_intent: str
    deliverables: List[str]
    tool: Optional[str] = None


AgentName = Literal[
    "DataArchitectAgent",
    "DataAnalystTechnicalAgent",
    "DataAnalystBusinessAgent",
    "DataScientistAgent",
]

ToolName = Literal[
    "clean_data",
    "descriptive_stats",
    "detect_outliers",
    "correlation_matrix",
    "get_exploratory_analysis",
    "plot_histogram",
    "plot_boxplot",
    "plot_scatter",
    "generate_chart",
    "run_kmeans_clustering",
    "get_data_types",
    "get_central_tendency",
    "get_variability",
    "get_ranges",
    "get_value_counts",
    "get_frequent_values",
    "get_temporal_patterns",
    "get_clusters_summary",
    "get_outliers_summary",
    "get_variable_relations",
    "get_influential_variables",
    "perform_t_test",
    "perform_chi_square",
    "linear_regression",
    "logistic_regression",
    "random_forest_classifier",
    "normalize_data",
    "impute_missing",
    "pca_dimensionality",
    "decompose_time_series",
    "compare_datasets",
    "plot_heatmap",
    "evaluate_model",
    "forecast_arima",
    "perform_anova",
    "check_duplicates",
    "select_features",
    "generate_wordcloud",
    "plot_line_chart",
    "plot_violin_plot",
    "perform_kruskal_wallis",
    "svm_classifier",
    "knn_classifier",
    "sentiment_analysis",
    "plot_geospatial_map",
    "perform_survival_analysis",
    "topic_modeling",
    "perform_bayesian_inference",
    # Extended tools
    "data_profiling",
    "missing_data_analysis",
    "cardinality_analysis",
    "distribution_tests",
    "create_polynomial_features",
    "create_interaction_features",
    "create_binning",
    "create_rolling_features",
    "create_lag_features",
    "correlation_tests",
    "multicollinearity_detection",
    "gradient_boosting_classifier",
    "hyperparameter_tuning",
    "feature_importance_analysis",
    "model_evaluation_detailed",
    "rfm_analysis",
    "ab_test_analysis",
    "export_to_excel",
    "export_analysis_results",
    # Data validation
    "validate_and_clean_dataframe",
    "smart_type_inference",
    "detect_data_quality_issues",
]


class PlanTaskModel(BaseModel):
    """Single execution task in the plan."""
    task_id: int
    description: str
    agent_responsible: AgentName
    tool_to_use: ToolName
    dependencies: List[int] = Field(default_factory=list)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    output_variable: str


class ExecutionPlanModel(BaseModel):
    """Validated execution plan produced by TeamLeaderAgent."""
    execution_plan: List[PlanTaskModel]

class OrchestratorAgent(BaseAgent):
    def run(self, user_query: str) -> dict:
        # Use dynamic prompt with agent profile
        prompt = get_orchestrator_prompt(user_query)
        
        # Execute with rate limiting and retry
        def _execute():
            response = self.llm.invoke(prompt).content
            if not response.strip():
                raise ValueError("Empty response from LLM. Check API key and connectivity.")
            return response
        
        response = self.rate_limiter.execute_with_retry(
            _execute,
            max_retries=3,
            on_wait=self.wait_callback
        )
        response = clean_json_response(response)
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {response[:500]}") from e
        # Validate with Pydantic and return as dict
        try:
            briefing = BriefingModel.model_validate(data)
            result = briefing.model_dump()
            # Heuristic backfill for missing tool
            try:
                if (result.get('tool') is None or str(result.get('tool')).strip() == ''):
                    intent = (result.get('main_intent') or '').lower()
                    if intent in ['simple_analysis', 'statistical_analysis', 'visualization']:
                        text = (result.get('user_query') or '')
                        # Include deliverables text if available to help inference
                        if isinstance(result.get('deliverables'), list):
                            text += ' ' + ' '.join([str(d) for d in result['deliverables']])
                        t = text.lower()
                        # Keyword-to-tool mapping aligned with app.py simple_mappings
                        keyword_map = [
                            ('outlier', 'detect_outliers'),
                            ('correla', 'correlation_matrix'),
                            ('duplic', 'check_duplicates'),
                            ('hist', 'plot_histogram'),
                            ('box', 'plot_boxplot'),
                            ('scatter', 'plot_scatter'),
                            ('kmeans', 'run_kmeans_clustering'),
                            ('tipo', 'get_data_types'),
                            ('média', 'get_central_tendency'),
                            ('media', 'get_central_tendency'),
                            ('mediana', 'get_central_tendency'),
                            ('variân', 'get_variability'),
                            ('varian', 'get_variability'),
                            ('desvio', 'get_variability'),
                            ('intervalo', 'get_ranges'),
                            ('percentil', 'get_ranges'),
                            ('contagem', 'get_value_counts'),
                            ('frequente', 'get_frequent_values'),
                            ('temporal', 'get_temporal_patterns'),
                            ('sazonal', 'get_temporal_patterns'),
                            ('clusters', 'get_clusters_summary'),
                            ('sentimento', 'sentiment_analysis'),
                            ('wordcloud', 'generate_wordcloud'),
                            ('mapa', 'plot_geospatial_map'),
                            ('sobreviv', 'perform_survival_analysis'),
                            ('tópico', 'topic_modeling'),
                            ('topico', 'topic_modeling'),
                            ('bayesian', 'perform_bayesian_inference'),
                            # Model evaluation patterns
                            ('precision', 'evaluate_model'),
                            ('recall', 'evaluate_model'),
                            ('matriz de confus', 'evaluate_model'),
                        ]
                        inferred = None
                        for kw, tool in keyword_map:
                            if kw in t:
                                inferred = tool
                                break
                        result['tool'] = inferred or 'descriptive_stats'
            except Exception:
                # Never fail the flow on heuristic filling
                pass
            return result
        except ValidationError as ve:
            # Provide structured feedback upstream; keep same shape for downstream components.
            raise ValueError(f"Briefing validation error: {ve}")

class TeamLeaderAgent(BaseAgent):
    def create_plan(self, briefing: dict) -> dict:
        # Try up to 2 correction rounds when validation fails
        error_note = None
        for attempt in range(3):
            # Use dynamic prompt with agent profile
            prompt = get_team_leader_plan_prompt(json.dumps(briefing, indent=2))
            if error_note:
                prompt += f"\n\nATENÇÃO: O plano anterior era inválido pelos seguintes motivos de validação. Corrija e retorne APENAS JSON válido no schema exigido.\nErros: {error_note}\n"
            
            # Execute with rate limiting
            def _execute():
                response = self.llm.invoke(prompt).content
                if not response.strip():
                    raise ValueError("Empty response from LLM. Check API key and connectivity.")
                return response
            
            response = self.rate_limiter.execute_with_retry(
                _execute,
                max_retries=2,
                on_wait=self.wait_callback
            )
            response = clean_json_response(response)
            try:
                data = json.loads(response)
            except json.JSONDecodeError as e:
                error_note = f"JSON inválido: {str(e)} | Trecho: {response[:400]}"
                continue
            # Validate with Pydantic
            try:
                plan = ExecutionPlanModel.model_validate(data)
                return plan.model_dump()
            except ValidationError as ve:
                error_note = str(ve)
                continue
        # If we reach here, return last parsed data if any, otherwise raise
        if error_note:
            raise ValueError(f"Plano inválido após tentativas de correção: {error_note}")
        raise ValueError("Falha ao obter plano válido do LLM.")
        
    def synthesize_results(self, execution_results: dict, tools_used: List[str] = None) -> str:
        # Extract tools used from execution results if not provided
        if tools_used is None:
            tools_used = []
            for task_result in execution_results.get('tasks', []):
                tool = task_result.get('tool_used')
                if tool and tool not in tools_used:
                    tools_used.append(tool)
        
        # Use dynamic prompt with agent profile and tool context
        prompt = get_synthesis_prompt(execution_results, tools_used=tools_used)
        
        # Execute with rate limiting
        def _execute():
            return self.llm.invoke(prompt).content
        
        return self.rate_limiter.execute_with_retry(
            _execute,
            max_retries=3,
            on_wait=self.wait_callback
        )

# Specialist agents are simpler since their logic is to execute a tool
class SpecialistAgent(BaseAgent):
    def execute_task(self, tool_function, kwargs) -> Any:
        # The "intelligence" here is simply to call the correct Python function
        return tool_function(**kwargs)

class DataArchitectAgent(SpecialistAgent): pass
class DataAnalystTechnicalAgent(SpecialistAgent): pass
class DataAnalystBusinessAgent(SpecialistAgent):
    def generate_final_response(self, synthesis_report: str, memory_context: str, tools_used: List[str] = None):
        # Use dynamic prompt with agent profile and tool context
        prompt = get_final_response_prompt(synthesis_report, memory_context, tools_used=tools_used)
        
        # Execute with rate limiting (streaming mode)
        def _execute():
            return self.llm.stream(prompt)
        
        return self.rate_limiter.execute_with_retry(
            _execute,
            max_retries=3,
            on_wait=self.wait_callback
        )

class DataScientistAgent(SpecialistAgent): pass
