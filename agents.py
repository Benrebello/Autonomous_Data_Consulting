# agents.py
import json
import time
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List, Optional, Literal
import sys
from pathlib import Path

# Ensure local module resolution for 'prompts.py' to avoid shadowing by similarly named external packages
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompts import (ORCHESTRATOR_PROMPT, TEAM_LEADER_PROMPT, 
                     SYNTHESIS_PROMPT, FINAL_RESPONSE_PROMPT)

# Global for rate limiting
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
    def __init__(self, llm, rpm_limit=10):
        self.llm = llm
        self.rpm_limit = rpm_limit


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
        global last_call_time
        current_time = time.time()
        interval = 60 / self.rpm_limit
        if current_time - last_call_time < interval:
            time.sleep(interval - (current_time - last_call_time))
        last_call_time = time.time()
        
        prompt = ORCHESTRATOR_PROMPT.format(user_query=user_query)
        response = self.llm.invoke(prompt).content
        if not response.strip():
            raise ValueError("Empty response from LLM. Check API key and connectivity.")
        response = clean_json_response(response)
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM response is not valid JSON: {response[:500]}") from e
        # Validate with Pydantic and return as dict
        try:
            briefing = BriefingModel.model_validate(data)
            return briefing.model_dump()
        except ValidationError as ve:
            # Provide structured feedback upstream; keep same shape for downstream components.
            raise ValueError(f"Briefing validation error: {ve}")

class TeamLeaderAgent(BaseAgent):
    def create_plan(self, briefing: dict) -> dict:
        global last_call_time
        current_time = time.time()
        interval = 60 / self.rpm_limit
        if current_time - last_call_time < interval:
            time.sleep(interval - (current_time - last_call_time))
        last_call_time = time.time()
        
        # Try up to 2 correction rounds when validation fails
        error_note = None
        for attempt in range(3):
            prompt = TEAM_LEADER_PROMPT.format(briefing=json.dumps(briefing, indent=2))
            if error_note:
                prompt += f"\n\nATENÇÃO: O plano anterior era inválido pelos seguintes motivos de validação. Corrija e retorne APENAS JSON válido no schema exigido.\nErros: {error_note}\n"
            response = self.llm.invoke(prompt).content
            if not response.strip():
                raise ValueError("Empty response from LLM. Check API key and connectivity.")
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
        
    def synthesize_results(self, execution_results: dict) -> str:
        global last_call_time
        current_time = time.time()
        interval = 60 / self.rpm_limit
        if current_time - last_call_time < interval:
            time.sleep(interval - (current_time - last_call_time))
        last_call_time = time.time()
        
        prompt = SYNTHESIS_PROMPT.format(execution_results=json.dumps(execution_results, default=str, indent=2))
        return self.llm.invoke(prompt).content

# Specialist agents are simpler since their logic is to execute a tool
class SpecialistAgent(BaseAgent):
    def execute_task(self, tool_function, kwargs) -> Any:
        # The "intelligence" here is simply to call the correct Python function
        return tool_function(**kwargs)

class DataArchitectAgent(SpecialistAgent): pass
class DataAnalystTechnicalAgent(SpecialistAgent): pass
class DataAnalystBusinessAgent(SpecialistAgent):
    def generate_final_response(self, synthesis_report: str, memory_context: str):
        global last_call_time
        current_time = time.time()
        interval = 60 / self.rpm_limit
        if current_time - last_call_time < interval:
            time.sleep(interval - (current_time - last_call_time))
        last_call_time = time.time()
        
        prompt = FINAL_RESPONSE_PROMPT.format(synthesis_report=synthesis_report, memory_context=memory_context)
        return self.llm.stream(prompt) # Habilita o streaming

class DataScientistAgent(SpecialistAgent): pass
