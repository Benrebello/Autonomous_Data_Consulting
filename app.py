# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Dict, Any
import os
import glob
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import enums
from io import BytesIO

# Local imports
from config import load_config, obtain_llm
from agents import (OrchestratorAgent, TeamLeaderAgent, DataArchitectAgent, 
                    DataAnalystTechnicalAgent, DataAnalystBusinessAgent, DataScientistAgent)
import tools

def stream_response_to_chat(stream) -> str:
    """Stream chunks to a single chat message, keeping text on screen.

    Aggregates chunk content and updates one placeholder to avoid
    per-chunk newlines and flicker. Returns the full concatenated text.
    """
    placeholder = st.empty()
    full_text = ""
    for chunk in stream:
        chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
        full_text += chunk_text
        # Render progressively in markdown to preserve formatting
        placeholder.markdown(full_text)
    return full_text

def cleanup_old_plot_files(pattern: str = "plot_*.png") -> int:
    """Remove legacy plot files matching the given pattern in the CWD.

    Returns the number of files removed.
    """
    removed = 0
    for path in glob.glob(pattern):
        try:
            os.remove(path)
            removed += 1
        except Exception:
            pass
    return removed

def generate_pdf_report(title: str, user_query: str, synthesis: str, full_response: str, plan: dict, charts: list[bytes]) -> bytes:
    """Generate a PDF report following ABNT-like formatting and Minto Pyramid structure.

    - Title page
    - Executive Summary (Minto: Situation, Complication, Question, Answer)
    - Development (Methods, Results with figures)
    - Conclusion and Recommendations
    - References (placeholder)
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=3*cm, rightMargin=2*cm,
                            topMargin=3*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    # ABNT-like: Times 12, 1.5 spacing (approx using spaceBefore/After)
    normal = ParagraphStyle('ABNT_Normal', parent=styles['Normal'], fontName='Times-Roman', fontSize=12, leading=18)
    h1 = ParagraphStyle('ABNT_H1', parent=styles['Heading1'], fontName='Times-Bold', fontSize=14, spaceAfter=12, alignment=enums.TA_CENTER)
    h2 = ParagraphStyle('ABNT_H2', parent=styles['Heading2'], fontName='Times-Bold', fontSize=12, spaceAfter=8)

    elements = []
    # Title page
    elements.append(Paragraph(title or 'Relatório de Análise de Dados', h1))
    elements.append(Spacer(1, 18))
    elements.append(Paragraph(f"Pergunta do usuário: {user_query}", normal))
    elements.append(Spacer(1, 18))
    elements.append(Paragraph("Autores: Equipe Multiagente (Orchestrator, Team Leader, Data Architect, Data Analyst, Data Scientist)", normal))
    elements.append(PageBreak())

    # Executive Summary - Minto Pyramid
    elements.append(Paragraph("Resumo Executivo (Pirâmide de Minto)", h2))
    elements.append(Paragraph("Situação: Contexto do conjunto de dados e objetivo declarado pelo usuário.", normal))
    elements.append(Paragraph("Complicação: Limitações, qualidade dos dados, volume e restrições levantadas.", normal))
    elements.append(Paragraph("Questão-chave: Qual insight ou decisão a análise precisa apoiar?", normal))
    elements.append(Paragraph("Resposta: Síntese de alto nível dos resultados e implicações.", normal))
    if synthesis:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Síntese do Time Leader:", h2))
        elements.append(Paragraph(full_response[:4000].replace('\n', '<br/>'), normal))

    # Development
    elements.append(PageBreak())
    elements.append(Paragraph("Desenvolvimento", h2))
    elements.append(Paragraph("Método: Plano de execução gerado e ferramentas utilizadas.", normal))
    if plan:
        try:
            plan_brief = str({k: plan[k] for k in plan.keys() if k != 'execution_plan'})
        except Exception:
            plan_brief = 'Plano não disponível.'
        elements.append(Paragraph(f"Plano: {plan_brief}", normal))
        if 'execution_plan' in plan:
            for t in plan['execution_plan'][:10]:
                desc = t.get('description', '')
                tool = t.get('tool_to_use', '')
                elements.append(Paragraph(f"Tarefa {t.get('task_id')}: {desc} (ferramenta: {tool})", normal))

    # Results with charts
    if charts:
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Resultados (Figuras)", h2))
        for ch in charts[:6]:  # limit pages
            try:
                img = Image(BytesIO(ch))
                img._restrictSize(15*cm, 12*cm)
                elements.append(img)
                elements.append(Spacer(1, 12))
            except Exception:
                continue

    # Conclusion
    elements.append(PageBreak())
    elements.append(Paragraph("Conclusões e Recomendações", h2))
    elements.append(Paragraph(full_response.replace('\n', '<br/>')[:8000], normal))

    # References
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Referências (quando aplicável)", h2))
    elements.append(Paragraph("Este relatório segue formatação semelhante às normas ABNT (margens e tipografia) e estrutura de comunicação da Pirâmide de Minto.", normal))

    doc.build(elements)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes

class AnalysisPipeline:
    def __init__(self, llm, dataframes: Dict[str, pd.DataFrame], rpm_limit=10):
        self.dataframes = dataframes
        self.shared_context = {"dataframes": self.dataframes}
        
        # Instancia todos os agentes
        self.orchestrator = OrchestratorAgent(llm, rpm_limit)
        self.team_leader = TeamLeaderAgent(llm, rpm_limit)
        self.agents = {
            "DataArchitectAgent": DataArchitectAgent(llm, rpm_limit),
            "DataAnalystTechnicalAgent": DataAnalystTechnicalAgent(llm, rpm_limit),
            "DataAnalystBusinessAgent": DataAnalystBusinessAgent(llm, rpm_limit),
            "DataScientistAgent": DataScientistAgent(llm, rpm_limit),
        }
        
        # Mapeia nomes de ferramentas para funções
        self.tool_mapping = {
            "join_datasets": tools.join_datasets,
            "join_datasets_on": tools.join_datasets_on,
            "clean_data": tools.clean_data,
            "descriptive_stats": tools.descriptive_stats,
            "detect_outliers": tools.detect_outliers,
            "correlation_matrix": tools.correlation_matrix,
            "get_exploratory_analysis": tools.get_exploratory_analysis,
            "plot_histogram": tools.plot_histogram,
            "plot_boxplot": tools.plot_boxplot,
            "plot_scatter": tools.plot_scatter,
            "generate_chart": tools.generate_chart,
            "run_kmeans_clustering": tools.run_kmeans_clustering,
            "get_data_types": tools.get_data_types,
            "get_central_tendency": tools.get_central_tendency,
            "get_variability": tools.get_variability,
            "get_ranges": tools.get_ranges,
            "calculate_min_max_per_variable": tools.calculate_min_max_per_variable,
            "get_value_counts": tools.get_value_counts,
            "get_frequent_values": tools.get_frequent_values,
            "get_temporal_patterns": tools.get_temporal_patterns,
            "get_clusters_summary": tools.get_clusters_summary,
            "get_outliers_summary": tools.get_outliers_summary,
            "get_variable_relations": tools.get_variable_relations,
            "get_influential_variables": tools.get_influential_variables,
            "perform_t_test": tools.perform_t_test,
            "perform_chi_square": tools.perform_chi_square,
            "linear_regression": tools.linear_regression,
            "logistic_regression": tools.logistic_regression,
            "random_forest_classifier": tools.random_forest_classifier,
            "normalize_data": tools.normalize_data,
            "impute_missing": tools.impute_missing,
            "pca_dimensionality": tools.pca_dimensionality,
            "decompose_time_series": tools.decompose_time_series,
            "compare_datasets": tools.compare_datasets,
            "plot_heatmap": tools.plot_heatmap,
            "evaluate_model": tools.evaluate_model,
            "forecast_arima": tools.forecast_arima,
            "perform_anova": tools.perform_anova,
            "check_duplicates": tools.check_duplicates,
            "select_features": tools.select_features,
            "generate_wordcloud": tools.generate_wordcloud,
            "plot_line_chart": tools.plot_line_chart,
            "plot_violin_plot": tools.plot_violin_plot,
            "perform_kruskal_wallis": tools.perform_kruskal_wallis,
            "svm_classifier": tools.svm_classifier,
            "knn_classifier": tools.knn_classifier,
            "sentiment_analysis": tools.sentiment_analysis,
            "plot_geospatial_map": tools.plot_geospatial_map,
            "perform_survival_analysis": tools.perform_survival_analysis,
            "topic_modeling": tools.topic_modeling,
            "perform_bayesian_inference": tools.perform_bayesian_inference,
            "sort_dataframe": tools.sort_dataframe,
            "group_and_aggregate": tools.group_and_aggregate,
            "create_pivot_table": tools.create_pivot_table,
            "remove_duplicates": tools.remove_duplicates,
            "fill_missing_with_median": tools.fill_missing_with_median,
            "detect_and_remove_outliers": tools.detect_and_remove_outliers,
            "calculate_skewness_kurtosis": tools.calculate_skewness_kurtosis,
            "perform_multiple_regression": tools.perform_multiple_regression,
            "cluster_with_kmeans": tools.cluster_with_kmeans,
            "calculate_growth_rate": tools.calculate_growth_rate,
            "perform_abc_analysis": tools.perform_abc_analysis,
            "forecast_time_series_arima": tools.forecast_time_series_arima,
            "risk_assessment": tools.risk_assessment,
            "sensitivity_analysis": tools.sensitivity_analysis,
            "monte_carlo_simulation": tools.monte_carlo_simulation,
            "validate_and_correct_data_types": tools.validate_and_correct_data_types,
            "perform_causal_inference": tools.perform_causal_inference,
            "perform_named_entity_recognition": tools.perform_named_entity_recognition,
            "text_summarization": tools.text_summarization,
            "add_time_features_from_seconds": tools.add_time_features_from_seconds,
        }

        # Memória para armazenar conclusões anteriores
        self.memory = st.session_state.get('memory', [])

    def _truncate_str(self, s: str, max_len: int = 2000) -> str:
        if not isinstance(s, str):
            s = str(s)
        return s if len(s) <= max_len else s[:max_len] + "... [truncated]"

    def _summarize_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Return a compact summary of a DataFrame to keep token usage low."""
        summary: Dict[str, Any] = {
            "shape": list(df.shape),
            "columns": list(df.columns[:100]),  # limit to first 100
            "dtypes": {c: str(df.dtypes[c]) for c in df.columns[:100]},
        }
        # Add a tiny sample (up to 5 rows, 10 cols) as CSV text truncated
        sample = df.iloc[:5, :10]
        csv_text = sample.to_csv(index=False)
        summary["sample_csv"] = self._truncate_str(csv_text, 1500)
        return summary

    def _compact_shared_context(self) -> Dict[str, Any]:
        """Build a compact context object for LLM synthesis to avoid exceeding token limits."""
        compact: Dict[str, Any] = {}
        for key, val in self.shared_context.items():
            try:
                if isinstance(val, pd.DataFrame):
                    compact[key] = {"type": "dataframe_summary", **self._summarize_dataframe(val)}
                elif isinstance(val, (str, int, float, bool)):
                    compact[key] = val if not isinstance(val, str) else self._truncate_str(val, 1500)
                elif isinstance(val, dict):
                    # For dicts, keep shallow keys and truncate string values
                    compact[key] = {str(k): (self._truncate_str(v, 800) if isinstance(v, str) else v)
                                    for k, v in list(val.items())[:50]}
                else:
                    # Fallback to string representation truncated
                    compact[key] = {"type": type(val).__name__, "repr": self._truncate_str(str(val), 800)}
            except Exception:
                compact[key] = "[unserializable]"
        return compact

    def _set_default_context(self):
        """Derive and store default DataFrame and numeric columns for tools."""
        if not self.dataframes:
            return
        # Prefer user-selected default df if available
        selected_key = st.session_state.get('default_df_key')
        if selected_key and selected_key in self.dataframes:
            default_df = self.dataframes[selected_key]
        else:
            # Fallback to first uploaded dataframe as default
            default_df = next(iter(self.dataframes.values()))
        self.shared_context['df_default'] = default_df
        # Determine numeric columns
        numeric_cols = list(default_df.select_dtypes(include=['number']).columns)
        self.shared_context['numeric_columns'] = numeric_cols

    def _resolve_inputs(self, task_inputs: dict) -> dict:
        """Resolve referências a variáveis do contexto compartilhado."""
        resolved_inputs = {}
        for key, value in task_inputs.items():
            if isinstance(value, str) and value.startswith('@'):
                # É uma referência a um resultado anterior
                var_name = value[1:]
                resolved_inputs[key] = self.shared_context.get(var_name)
            elif isinstance(value, str) and value in self.dataframes:
                 # É uma referência a um dataframe original
                resolved_inputs[key] = self.dataframes[value]
            else:
                resolved_inputs[key] = value
        # Sanitização de DataFrame "impresso" como string
        try:
            self._set_default_context()
            df_default = self.shared_context.get('df_default')
            for k, v in list(resolved_inputs.items()):
                if isinstance(v, str) and len(v) > 200 and ('\n' in v or '... ' in v):
                    # Heurística: parece uma impressão de DataFrame. Substituir por df_default.
                    if isinstance(df_default, pd.DataFrame) and not df_default.empty:
                        resolved_inputs[k] = df_default
        
            # Validação de colunas comuns
            df_for_validation = None
            # Encontrar df nos inputs
            for cand_key in ['df', 'df1', 'df2', 'X']:
                if isinstance(resolved_inputs.get(cand_key), pd.DataFrame):
                    df_for_validation = resolved_inputs[cand_key]
                    break
            if isinstance(df_for_validation, pd.DataFrame):
                cols = set(df_for_validation.columns)
                for col_key in ['column', 'x_column', 'y_column', 'time_column', 'event_column']:
                    val = resolved_inputs.get(col_key)
                    if isinstance(val, str) and val not in cols:
                        # Tentar fallback
                        num_cols = list(df_for_validation.select_dtypes(include=['number']).columns)
                        fallback = num_cols[0] if num_cols else (df_for_validation.columns[0] if len(df_for_validation.columns)>0 else None)
                        if fallback:
                            resolved_inputs[col_key] = fallback
                        else:
                            raise ValueError(f"Coluna '{val}' não encontrada no DataFrame para parâmetro '{col_key}'.")
                # Listas de colunas
                if isinstance(resolved_inputs.get('columns'), list):
                    resolved_inputs['columns'] = [c for c in resolved_inputs['columns'] if c in cols]
                    if not resolved_inputs['columns']:
                        # fallback para primeiras 2 colunas numéricas
                        num_cols = list(df_for_validation.select_dtypes(include=['number']).columns)
                        resolved_inputs['columns'] = num_cols[:2] if len(num_cols) >= 2 else list(df_for_validation.columns[:2])
        except Exception:
            pass
        return resolved_inputs

    def _fill_default_inputs_for_task_join_helper(self, tool: str, inputs: dict) -> dict:
        """Helper mantido por compatibilidade: delega para _fill_default_inputs_for_task."""
        return self._fill_default_inputs_for_task(tool, inputs)

    def _fill_default_inputs_for_task(self, tool, inputs):
        """Fill default inputs for a task based on the tool."""
        df_default = self.dataframes[next(iter(self.dataframes.keys()))] if self.dataframes else None
        if df_default is None or df_default.empty:
            return inputs

        numeric_cols = list(df_default.select_dtypes(include=[np.number]).columns)
        cat_cols = list(df_default.select_dtypes(include=['object', 'category']).columns)
        if not cat_cols and 'class' in df_default.columns:  # for creditcard, class is int but categorical
            cat_cols = ['class']

        if tool == 'descriptive_stats':
            inputs = {'df': df_default}
        elif tool == 'plot_histogram':
            inputs = {'df': df_default, 'column': numeric_cols[0] if numeric_cols else df_default.columns[0]}
        elif tool == 'detect_outliers':
            inputs = {'df': df_default, 'column': numeric_cols[0] if numeric_cols else df_default.columns[0], 'method': 'iqr'}
        elif tool == 'correlation_matrix':
            inputs = {'df': df_default}
        elif tool == 'plot_boxplot':
            inputs = {'df': df_default, 'column': numeric_cols[0] if numeric_cols else df_default.columns[0]}
        elif tool == 'plot_scatter':
            if len(numeric_cols) >= 2:
                inputs = {'df': df_default, 'x_column': numeric_cols[0], 'y_column': numeric_cols[1]}
            else:
                inputs = {'df': df_default, 'x_column': df_default.columns[0], 'y_column': df_default.columns[1] if len(df_default.columns) > 1 else df_default.columns[0]}
        elif tool == 'run_kmeans_clustering':
            if len(numeric_cols) >= 2:
                inputs = {'df': df_default, 'columns': numeric_cols[:2], 'n_clusters': 3}
            else:
                inputs = {'df': df_default, 'columns': df_default.columns[:2], 'n_clusters': 3}
        elif tool == 'clean_data':
            inputs = {'df': df_default, 'column': numeric_cols[0] if numeric_cols else df_default.columns[0], 'strategy': 'median'}
        elif tool == 'generate_chart':
            inputs = {'df': df_default, 'chart_type': 'bar', 'x_column': cat_cols[0] if cat_cols else df_default.columns[0]}
        elif tool == 'get_exploratory_analysis':
            inputs = {'df': df_default}
        elif tool == 'get_data_types':
            inputs = {'df': df_default}
        elif tool == 'get_central_tendency':
            inputs = {'df': df_default}
        elif tool == 'get_variability':
            inputs = {'df': df_default}
        elif tool == 'get_ranges':
            inputs = {'df': df_default}
        elif tool == 'calculate_min_max_per_variable':
            inputs = {'df': df_default}
        elif tool == 'get_value_counts':
            inputs = {'df': df_default, 'column': cat_cols[0] if cat_cols else df_default.columns[0]}
        elif tool == 'get_frequent_values':
            inputs = {'df': df_default, 'column': cat_cols[0] if cat_cols else df_default.columns[0], 'top_n': 10}
        elif tool == 'get_temporal_patterns':
            time_col = [c for c in df_default.columns if 'time' in c.lower()]
            val_col = numeric_cols[0] if numeric_cols else df_default.columns[0]
            inputs = {'df': df_default, 'time_column': time_col[0] if time_col else df_default.columns[0], 'value_column': val_col}
        elif tool == 'get_clusters_summary':
            inputs = {'df': df_default, 'n_clusters': 3}
        elif tool == 'get_outliers_summary':
            inputs = {'df': df_default, 'column': numeric_cols[0] if numeric_cols else df_default.columns[0]}
        elif tool == 'get_variable_relations':
            if len(numeric_cols) >= 2:
                inputs = {'df': df_default, 'x_column': numeric_cols[0], 'y_column': numeric_cols[1]}
            else:
                inputs = {'df': df_default, 'x_column': df_default.columns[0], 'y_column': df_default.columns[1] if len(df_default.columns) > 1 else df_default.columns[0]}
        elif tool == 'get_influential_variables':
            # Prefer a numeric target to avoid conversion issues
            target = None
            if numeric_cols:
                target = numeric_cols[-1]
            else:
                # Fallback to a column named 'class' if numeric, else last column
                if 'class' in df_default.columns and pd.api.types.is_numeric_dtype(df_default['class']):
                    target = 'class'
                else:
                    target = df_default.columns[-1]
            inputs = {'df': df_default, 'target_column': target}
        elif tool == 'perform_t_test':
            inputs = {'df': df_default, 'column': numeric_cols[0] if numeric_cols else df_default.columns[0], 'group_column': cat_cols[0] if cat_cols else df_default.columns[0]}
        elif tool == 'perform_chi_square':
            cat_cols_list = cat_cols[:2] if len(cat_cols) >= 2 else df_default.columns[:2]
            inputs = {'df': df_default, 'column1': cat_cols_list[0], 'column2': cat_cols_list[1]}
        elif tool == 'linear_regression':
            if len(numeric_cols) >= 2:
                inputs = {'df': df_default, 'x_columns': numeric_cols[:-1], 'y_column': numeric_cols[-1]}
            else:
                inputs = {'df': df_default, 'x_columns': df_default.columns[:-1].tolist(), 'y_column': df_default.columns[-1]}
        elif tool == 'logistic_regression':
            target = 'class' if 'class' in df_default.columns else df_default.columns[-1]
            x_cols = [c for c in numeric_cols if c != target][:5]  # limit
            inputs = {'df': df_default, 'x_columns': x_cols, 'y_column': target}
        elif tool == 'random_forest_classifier':
            target = 'class' if 'class' in df_default.columns else df_default.columns[-1]
            x_cols = [c for c in numeric_cols if c != target][:5]
            inputs = {'df': df_default, 'x_columns': x_cols, 'y_column': target}
        elif tool == 'normalize_data':
            inputs = {'df': df_default, 'columns': numeric_cols}
        elif tool == 'impute_missing':
            # Use a safer default that works with mixed types
            inputs = {'df': df_default, 'strategy': 'most_frequent'}
        elif tool == 'pca_dimensionality':
            inputs = {'df': df_default, 'n_components': 2}
        elif tool == 'decompose_time_series':
            time_col = [c for c in df_default.columns if 'time' in c.lower()]
            col = time_col[0] if time_col else numeric_cols[0] if numeric_cols else df_default.columns[0]
            inputs = {'df': df_default, 'column': col, 'period': 12}
        elif tool == 'compare_datasets':
            # For multiple, but here assume default, perhaps need adjustment
            inputs = {'df1': df_default, 'df2': df_default}  # placeholder
        elif tool == 'plot_heatmap':
            inputs = {'df': df_default, 'columns': numeric_cols}
        elif tool == 'evaluate_model':
            # Placeholder, as it needs model, X, y
            inputs = {'model': None, 'X': df_default[numeric_cols], 'y': df_default.get('class', df_default.iloc[:, -1])}
        elif tool == 'forecast_arima':
            time_col = [c for c in df_default.columns if 'time' in c.lower()]
            col = time_col[0] if time_col else numeric_cols[0] if numeric_cols else df_default.columns[0]
            inputs = {'df': df_default, 'column': col, 'order': (1,1,1), 'steps': 10}
        elif tool == 'perform_anova':
            inputs = {'df': df_default, 'column': numeric_cols[0] if numeric_cols else df_default.columns[0], 'group_column': cat_cols[0] if cat_cols else df_default.columns[0]}
        elif tool == 'check_duplicates':
            inputs = {'df': df_default}
        elif tool == 'select_features':
            target = 'class' if 'class' in df_default.columns else df_default.columns[-1]
            inputs = {'df': df_default, 'target_column': target, 'k': 10}
        elif tool == 'generate_wordcloud':
            text_cols = df_default.select_dtypes(include=['object']).columns
            col = text_cols[0] if len(text_cols) > 0 else df_default.columns[0]
            inputs = {'df': df_default, 'text_column': col}
        elif tool == 'plot_line_chart':
            x_col = [c for c in df_default.columns if 'time' in c.lower() or 'date' in c.lower()]
            x_col = x_col[0] if x_col else df_default.columns[0]
            y_col = numeric_cols[0] if numeric_cols else df_default.columns[1] if len(df_default.columns) > 1 else df_default.columns[0]
            inputs = {'df': df_default, 'x_column': x_col, 'y_column': y_col}
        elif tool == 'join_datasets':
            # Use same df and a categorical column as key if available
            key = cat_cols[0] if cat_cols else df_default.columns[0]
            inputs = {'df1': df_default, 'df2': df_default, 'on_column': key}
        elif tool == 'join_datasets_on':
            # Use same key on both sides
            key = cat_cols[0] if cat_cols else df_default.columns[0]
            inputs = {'df1': df_default, 'df2': df_default, 'left_on': key, 'right_on': key, 'how': 'inner'}
        elif tool == 'sort_dataframe':
            by_col = df_default.columns[0]
            inputs = {'df': df_default, 'by': by_col, 'ascending': True}
        elif tool == 'group_and_aggregate':
            grp = cat_cols[0] if cat_cols else df_default.columns[0]
            val = numeric_cols[0] if numeric_cols else df_default.columns[-1]
            inputs = {'df': df_default, 'group_by': [grp], 'agg_dict': {val: 'mean'}}
        elif tool == 'create_pivot_table':
            idx = cat_cols[0] if cat_cols else df_default.columns[0]
            cols = 'class' if 'class' in df_default.columns else (cat_cols[1] if len(cat_cols) > 1 else idx)
            vals = numeric_cols[0] if numeric_cols else df_default.columns[-1]
            inputs = {'df': df_default, 'index': idx, 'columns': cols, 'values': vals, 'aggfunc': 'mean'}
        elif tool == 'fill_missing_with_median':
            cols = numeric_cols if numeric_cols else []
            inputs = {'df': df_default, 'columns': cols}
        elif tool == 'detect_and_remove_outliers':
            col = numeric_cols[0] if numeric_cols else df_default.columns[0]
            inputs = {'df': df_default, 'column': col, 'method': 'iqr', 'threshold': 1.5}
        elif tool == 'calculate_skewness_kurtosis':
            cols = numeric_cols if numeric_cols else []
            inputs = {'df': df_default, 'columns': cols}
        elif tool == 'plot_violin_plot':
            x_col = cat_cols[0] if cat_cols else df_default.columns[0]
            y_col = numeric_cols[0] if numeric_cols else df_default.columns[1] if len(df_default.columns) > 1 else df_default.columns[0]
            inputs = {'df': df_default, 'x_column': x_col, 'y_column': y_col}
        elif tool == 'perform_kruskal_wallis':
            inputs = {'df': df_default, 'column': numeric_cols[0] if numeric_cols else df_default.columns[0], 'group_column': cat_cols[0] if cat_cols else df_default.columns[0]}
        elif tool == 'svm_classifier':
            target = 'class' if 'class' in df_default.columns else df_default.columns[-1]
            x_cols = [c for c in numeric_cols if c != target][:5]
            inputs = {'df': df_default, 'x_columns': x_cols, 'y_column': target}
        elif tool == 'knn_classifier':
            target = 'class' if 'class' in df_default.columns else df_default.columns[-1]
            x_cols = [c for c in numeric_cols if c != target][:5]
            inputs = {'df': df_default, 'x_columns': x_cols, 'y_column': target}
        elif tool == 'sentiment_analysis':
            text_cols = df_default.select_dtypes(include=['object']).columns
            col = text_cols[0] if len(text_cols) > 0 else df_default.columns[0]
            inputs = {'df': df_default, 'text_column': col}
        elif tool == 'perform_multiple_regression':
            # Use numeric columns; ensure at least 2 predictors and 1 target
            if len(numeric_cols) >= 2:
                inputs = {'df': df_default, 'x_columns': numeric_cols[:-1], 'y_column': numeric_cols[-1]}
            else:
                inputs = {'df': df_default, 'x_columns': df_default.columns[:-1].tolist(), 'y_column': df_default.columns[-1]}
        elif tool == 'cluster_with_kmeans':
            cols = numeric_cols[:2] if len(numeric_cols) >= 2 else list(df_default.columns[:2])
            inputs = {'df': df_default, 'columns': cols, 'n_clusters': 3}
        elif tool == 'calculate_growth_rate':
            time_col = 'time' if 'time' in df_default.columns else (df_default.columns[0])
            val_col = numeric_cols[0] if numeric_cols else df_default.columns[0]
            inputs = {'df': df_default, 'value_column': val_col, 'time_column': time_col}
        elif tool == 'perform_abc_analysis':
            val_col = numeric_cols[0] if numeric_cols else df_default.columns[0]
            cat_col = cat_cols[0] if cat_cols else df_default.columns[0]
            inputs = {'df': df_default, 'value_column': val_col, 'category_column': cat_col, 'a_threshold': 0.8, 'b_threshold': 0.95}
        elif tool == 'forecast_time_series_arima':
            col = numeric_cols[0] if numeric_cols else df_default.columns[0]
            inputs = {'df': df_default, 'column': col, 'periods': 10}
        elif tool == 'risk_assessment':
            risk_factors = numeric_cols[:3] if len(numeric_cols) >= 1 else list(df_default.columns[:1])
            inputs = {'df': df_default, 'risk_factors': risk_factors, 'weights': None}
        elif tool == 'sensitivity_analysis':
            # Provide a simple impact function
            impact_fn = (lambda v: float(v))
            variable_changes = {'var': [-0.1, 0.0, 0.1]}
            inputs = {'base_value': 100.0, 'variable_changes': variable_changes, 'impact_function': impact_fn}
        elif tool == 'monte_carlo_simulation':
            variables = {
                'x': {'type': 'normal', 'mean': 0, 'std': 1},
                'y': {'type': 'uniform', 'low': -1, 'high': 1},
            }
            output_fn = (lambda vals: vals['x'] + vals['y'])
            inputs = {'variables': variables, 'n_simulations': 100, 'output_function': output_fn}
        elif tool == 'perform_causal_inference':
            # Choose numeric treatment/outcome to ensure OLS works
            num_cols = numeric_cols if numeric_cols else list(df_default.select_dtypes(include=['number']).columns)
            if len(num_cols) >= 2:
                outcome = num_cols[-1]
                treatment = num_cols[0] if num_cols[0] != outcome else (num_cols[1] if len(num_cols) > 1 else num_cols[0])
                controls = [c for c in num_cols if c not in {treatment, outcome}][:3]
            else:
                # Fallback: use first two columns assuming they are numeric-like
                outcome = df_default.columns[-1]
                treatment = df_default.columns[0]
                controls = [c for c in df_default.columns if c not in {treatment, outcome}][:3]
            inputs = {'df': df_default, 'treatment': treatment, 'outcome': outcome, 'controls': controls}
        elif tool == 'perform_named_entity_recognition':
            text_cols = df_default.select_dtypes(include=['object']).columns
            col = text_cols[0] if len(text_cols) > 0 else df_default.columns[0]
            inputs = {'df': df_default, 'text_column': col}
        elif tool == 'text_summarization':
            # Use a small synthetic text
            sample_text = "This is a simple example. It demonstrates summarization. The function should return a concise text."
            inputs = {'text': sample_text, 'max_sentences': 2}
        elif tool == 'plot_geospatial_map':
            lat_col = [c for c in df_default.columns if 'lat' in c.lower()]
            lon_col = [c for c in df_default.columns if 'lon' in c.lower() or 'lng' in c.lower()]
            lat_col = lat_col[0] if lat_col else 'latitude' if 'latitude' in df_default.columns else None
            lon_col = lon_col[0] if lon_col else 'longitude' if 'longitude' in df_default.columns else None
            if lat_col and lon_col:
                inputs = {'df': df_default, 'lat_column': lat_col, 'lon_column': lon_col}
            else:
                inputs = {'df': df_default, 'lat_column': None, 'lon_column': None}
        elif tool == 'perform_survival_analysis':
            time_col = [c for c in df_default.columns if 'time' in c.lower()]
            event_col = [c for c in df_default.columns if 'event' in c.lower() or 'status' in c.lower()]
            time_col = time_col[0] if time_col else df_default.columns[0]
            event_col = event_col[0] if event_col else df_default.columns[1] if len(df_default.columns) > 1 else df_default.columns[0]
            inputs = {'df': df_default, 'time_column': time_col, 'event_column': event_col}
        elif tool == 'topic_modeling':
            text_cols = df_default.select_dtypes(include=['object']).columns
            col = text_cols[0] if len(text_cols) > 0 else df_default.columns[0]
            inputs = {'df': df_default, 'text_column': col, 'num_topics': 5}
        elif tool == 'perform_bayesian_inference':
            inputs = {'df': df_default, 'column': numeric_cols[0] if numeric_cols else df_default.columns[0], 'prior_mean': 0, 'prior_std': 1}
        elif tool == 'add_time_features_from_seconds':
            # Prefer a column named 'time' if exists; fallback to first numeric
            time_col = 'time' if 'time' in df_default.columns else (numeric_cols[0] if numeric_cols else df_default.columns[0])
            inputs = {'df': df_default, 'time_column': time_col, 'origin': '2000-01-01'}
        else:
            # Fallback
            inputs = {'df': df_default}

        return inputs

    def run(self, user_query: str):
        # Etapa 1: Orquestrador cria o briefing
        st.write("1. **Orquestrador:** Analisando sua solicitação...")
        
        # Verificar se é pergunta simples baseada em palavras-chave
        query_lower = user_query.lower()
        simple_mappings = {
            "outlier": "detect_outliers",
            "correla": "correlation_matrix",
            "duplicata": "check_duplicates",
            "média": "get_central_tendency",
            "mediana": "get_central_tendency",
            "moda": "get_central_tendency",
            "variância": "get_variability",
            "desvio": "get_variability",
            "quartil": "get_ranges",
            "percentil": "get_ranges",
            "contagem": "get_value_counts",
            "frequente": "get_frequent_values",
            "temporal": "get_temporal_patterns",
            "clusters": "get_clusters_summary",
            "sentimento": "sentiment_analysis",
            "wordcloud": "generate_wordcloud",
            "mapa": "plot_geospatial_map",
            "sobrevivência": "perform_survival_analysis",
            "tópico": "topic_modeling",
            "bayesian": "perform_bayesian_inference",
        }
        tool_name = None
        for keyword, tool in simple_mappings.items():
            if keyword in query_lower:
                tool_name = tool
                break
        
        if tool_name:
            briefing = {
                "main_intent": "simple_analysis",
                "tool": tool_name,
                "user_query": user_query,
                "main_goal": f"Responder à pergunta simples sobre {tool_name}",
                "key_questions": [user_query],
                "deliverables": ["Resultado direto da ferramenta"]
            }
        else:
            briefing = self.orchestrator.run(user_query)
        
        st.json(briefing)

        if briefing.get('main_intent') == 'simple_analysis':
            # Resposta direta para perguntas simples sobre aspectos específicos
            st.write("3. **Análise Direta:** Executando ferramenta e interpretando...")
            tool_name = briefing.get('tool')
            if tool_name in self.tool_mapping:
                inputs = self._fill_default_inputs_for_task(tool_name, {})
                resolved_inputs = self._resolve_inputs(inputs)
                result = self.tool_mapping[tool_name](**resolved_inputs)
                
                # Criar síntese técnica do resultado
                if isinstance(result, dict):
                    synthesis = f"Resultado da ferramenta {tool_name}: " + ", ".join([f"{k}: {v}" for k, v in result.items()])
                else:
                    synthesis = f"Resultado da ferramenta {tool_name}: {str(result)}"
                
                # Obter contexto de memória
                memory_context = "\n".join(self.memory[-5:])  # Últimas 5 memórias
                
                # Gerar resposta final pelo analista de negócios
                final_response_stream = self.agents["DataAnalystBusinessAgent"].generate_final_response(synthesis, memory_context)
                
                # Exibir resposta
                with st.chat_message("assistant"):
                    st.markdown("**Resposta à sua pergunta:**")
                    full_response = stream_response_to_chat(final_response_stream)
                    if isinstance(result, bytes):  # Para gráficos
                        st.image(result)
                
                # Armazenar resposta na memória
                self.memory.append(f"**Pergunta:** {user_query}\n**Resposta:** {full_response}")
                st.session_state['memory'] = self.memory
                
                # Gerar PDF simples
                charts_bytes = [result] if isinstance(result, bytes) else []
                pdf = generate_pdf_report(
                    title=f"Análise: {tool_name}",
                    user_query=user_query,
                    synthesis=synthesis,
                    full_response=full_response,
                    plan={"tool": tool_name},
                    charts=charts_bytes,
                )
                st.download_button(
                    label="Baixar análise em PDF",
                    data=pdf,
                    file_name=f"analise_{tool_name}.pdf",
                    mime="application/pdf",
                )
            else:
                st.error(f"Ferramenta '{tool_name}' não encontrada.")
            return  # Sair sem executar o fluxo completo

        if briefing.get('main_intent') == 'simple_data_description':
            # Resposta direta para perguntas simples sobre descrição de dados
            st.write("2. **Resposta Direta:** Gerando descrição simples dos dados...")
            if not self.dataframes:
                full_response = "Nenhum dado foi carregado ainda. Por favor, faça upload de arquivos de dados."
                synthesis_report = ""
                plan = {}
            else:
                summaries = []
                for name, df in self.dataframes.items():
                    shape = df.shape
                    columns = list(df.columns)
                    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
                    summaries.append(f"**Planilha '{name}':**\n- {shape[0]} linhas, {shape[1]} colunas.\n- Colunas: {', '.join(columns)}.\n- Tipos de dados: {', '.join([f'{k}: {v}' for k, v in dtypes.items()])}")
                full_response = "Aqui está um resumo dos dados disponíveis:\n\n" + "\n\n".join(summaries)
                synthesis_report = full_response
                plan = {}

            # Armazenar resposta na memória
            self.memory.append(f"**Pergunta:** {user_query}\n**Resposta:** {full_response}")
            st.session_state['memory'] = self.memory

            # Gerar PDF simples
            charts_bytes = []
            pdf = generate_pdf_report(
                title="Resumo de Dados",
                user_query=user_query,
                synthesis=synthesis_report,
                full_response=full_response,
                plan=plan,
                charts=charts_bytes,
            )
            st.download_button(
                label="Baixar resumo em PDF",
                data=pdf,
                file_name="resumo_dados.pdf",
                mime="application/pdf",
            )

            # Exibir resposta diretamente
            with st.chat_message("assistant"):
                st.markdown(full_response)
            return  # Sair sem executar o fluxo completo

        # Etapa 2: Team Leader cria o plano
        st.write("2. **Líder de Equipe:** Criando plano de execução...")
        plan = self.team_leader.create_plan(briefing)
        
        # Normalizar estrutura do plano se necessário
        if 'execution_plan' not in plan:
            if 'tarefas' in plan:
                plan['execution_plan'] = []
                for i, tarefa in enumerate(plan['tarefas'], 1):
                    desc = tarefa['descricao'].lower()
                    agent = tarefa['responsável']
                    if 'limpeza' in desc or 'juntar' in desc:
                        tool = 'clean_data'
                    elif 'estatísticas' in desc or 'descritivas' in desc:
                        tool = 'descriptive_stats'
                    elif 'gráficos' in desc or 'visualizar' in desc:
                        tool = 'plot_histogram'
                    elif 'outliers' in desc or 'atípicos' in desc:
                        tool = 'detect_outliers'
                    elif 'correlações' in desc:
                        tool = 'correlation_matrix'
                    elif 'clusters' in desc or 'clusterização' in desc:
                        tool = 'run_kmeans_clustering'
                    else:
                        tool = 'get_exploratory_analysis'
                    
                    inputs = {}
                    # Defaults by tool
                    inputs = self._fill_default_inputs_for_task(tool, inputs)
                    new_task = {
                        'task_id': i,
                        'description': tarefa['descricao'],
                        'agent_responsible': agent,
                        'tool_to_use': tool,
                        'dependencies': [],
                        'inputs': inputs,
                        'output_variable': f'result_{i}'
                    }
                    plan['execution_plan'].append(new_task)
            elif 'projeto' in plan and isinstance(plan['projeto'], list):
                # Variante: modelo retorna a lista de tarefas em 'projeto'
                plan['execution_plan'] = []
                for i, tarefa in enumerate(plan['projeto'], 1):
                    desc = str(tarefa.get('descricao') or tarefa.get('tarefa') or '').lower()
                    agent = tarefa.get('responsavel') or tarefa.get('responsável') or 'DataAnalystTechnicalAgent'
                    if 'limpeza' in desc or 'juntar' in desc or 'prepara' in desc:
                        tool = 'clean_data'
                    elif 'estat' in desc or 'descritiv' in desc:
                        tool = 'descriptive_stats'
                    elif 'histograma' in desc or 'gráf' in desc or 'density' in desc or 'visualiz' in desc:
                        tool = 'plot_histogram'
                    elif 'boxplot' in desc or 'outlier' in desc or 'atípic' in desc:
                        tool = 'plot_boxplot'
                    elif 'correla' in desc:
                        tool = 'correlation_matrix'
                    elif 'cluster' in desc:
                        tool = 'run_kmeans_clustering'
                    else:
                        tool = 'get_exploratory_analysis'

                    inputs = self._fill_default_inputs_for_task(tool, {})
                    new_task = {
                        'task_id': i,
                        'description': tarefa.get('descricao') or tarefa.get('tarefa') or '',
                        'agent_responsible': agent,
                        'tool_to_use': tool,
                        'dependencies': [],
                        'inputs': inputs,
                        'output_variable': f'result_{i}'
                    }
                    plan['execution_plan'].append(new_task)
            elif 'plano_de_execucao' in plan:
                plan['execution_plan'] = []
                for i, tarefa in enumerate(plan['plano_de_execucao'], 1):
                    desc = tarefa['descricao'].lower()
                    agent = tarefa['responsavel']
                    if 'limpeza' in desc or 'junção' in desc:
                        tool = 'clean_data'
                    elif 'estatísticas' in desc or 'descritivas' in desc:
                        tool = 'descriptive_stats'
                    elif 'histogramas' in desc or 'distribuições' in desc:
                        tool = 'plot_histogram'
                    elif 'boxplots' in desc or 'outliers' in desc:
                        tool = 'plot_boxplot'
                    elif 'correlação' in desc:
                        tool = 'correlation_matrix'
                    elif 'scatter' in desc or 'relações' in desc:
                        tool = 'plot_scatter'
                    elif 'conclusões' in desc or 'relatório' in desc:
                        tool = 'get_exploratory_analysis'  # or something
                    else:
                        tool = 'get_exploratory_analysis'
                    
                    inputs = self._fill_default_inputs_for_task(tool, {})
                    new_task = {
                        'task_id': i,
                        'description': tarefa['descricao'],
                        'agent_responsible': agent,
                        'tool_to_use': tool,
                        'dependencies': [],
                        'inputs': inputs,
                        'output_variable': f'result_{i}'
                    }
                    plan['execution_plan'].append(new_task)
        
        # Fallback: tentar detectar lista de tarefas sob qualquer chave conhecida
        if isinstance(plan, dict) and 'execution_plan' not in plan:
            for k, v in list(plan.items()):
                if isinstance(v, list) and v and isinstance(v[0], dict) and (('descricao' in v[0]) or ('tarefa' in v[0])):
                    plan['execution_plan'] = []
                    for i, tarefa in enumerate(v, 1):
                        desc = str(tarefa.get('descricao') or tarefa.get('tarefa') or '').lower()
                        agent = tarefa.get('responsavel') or tarefa.get('responsável') or 'DataAnalystTechnicalAgent'
                        if 'limpeza' in desc or 'juntar' in desc or 'prepara' in desc:
                            tool = 'clean_data'
                        elif 'estat' in desc or 'descritiv' in desc:
                            tool = 'descriptive_stats'
                        elif 'histograma' in desc or 'gráf' in desc or 'density' in desc or 'visualiz' in desc:
                            tool = 'plot_histogram'
                        elif 'boxplot' in desc or 'outlier' in desc or 'atípic' in desc:
                            tool = 'plot_boxplot'
                        elif 'correla' in desc:
                            tool = 'correlation_matrix'
                        elif 'cluster' in desc:
                            tool = 'run_kmeans_clustering'
                        else:
                            tool = 'get_exploratory_analysis'
                        inputs = self._fill_default_inputs_for_task(tool, {})
                        plan['execution_plan'].append({
                            'task_id': i,
                            'description': tarefa.get('descricao') or tarefa.get('tarefa') or '',
                            'agent_responsible': agent,
                            'tool_to_use': tool,
                            'dependencies': [],
                            'inputs': inputs,
                            'output_variable': f'result_{i}'
                        })
                    break

        # Ensure every task has the minimum required defaults
        for task in plan.get('execution_plan', []):
            task['inputs'] = self._fill_default_inputs_for_task(task.get('tool_to_use', ''), task.get('inputs', {}))

        if not isinstance(plan, dict) or 'execution_plan' not in plan:
            st.warning("Plano retornado pelo LLM não segue a estrutura padrão. Gerando plano simples para prosseguir.")
            plan = {
                'execution_plan': [
                    {
                        'task_id': 1,
                        'description': 'Análise exploratória básica',
                        'agent_responsible': 'DataAnalystTechnicalAgent',
                        'tool_to_use': 'get_exploratory_analysis',
                        'dependencies': [],
                        'inputs': self._fill_default_inputs_for_task('get_exploratory_analysis', {}),
                        'output_variable': 'result_1'
                    }
                ]
            }
        if not isinstance(plan['execution_plan'], list):
            raise ValueError(f"'execution_plan' deve ser uma lista. Plano: {plan}")
        st.json(plan)
        
        # Etapa 3: Execução do plano (com paralelização segura)
        st.write("3. **Esquadrão de Dados:** Executando as tarefas...")
        # Reinicia a lista de gráficos desta execução para evitar paths antigos
        st.session_state.charts = []
        tasks = plan['execution_plan']
        completed_task_ids = set()

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        def run_task(task):
            """Worker sem chamadas de UI: executa a ferramenta e retorna (task_id, result, error, duration_s)."""
            start = time.perf_counter()
            try:
                agent = self.agents[task['agent_responsible']]
                tool = self.tool_mapping[task['tool_to_use']]
                kwargs = self._resolve_inputs(task['inputs'])
                result = agent.execute_task(tool, kwargs)
                duration = time.perf_counter() - start
                return task['task_id'], result, None, duration
            except Exception as e:
                duration = time.perf_counter() - start
                return task['task_id'], None, e, duration

        while len(completed_task_ids) < len(tasks):
            # Seleciona tarefas prontas (dependências satisfeitas) e ainda não executadas
            ready_tasks = [t for t in tasks if t['task_id'] not in completed_task_ids and all(d in completed_task_ids for d in t.get('dependencies', []))]
            if not ready_tasks:
                st.error("Erro: Dependências circulares ou tarefas não executáveis detectadas.")
                break

            # Configuração dinâmica de paralelismo e autosplit por RPM
            user_parallel = int(st.session_state.get('max_parallel_tasks', 4))
            max_parallel = user_parallel
            # Adaptação dinâmica sob quota: reduzir paralelismo quando houver erro de quota recente
            api_status = st.session_state.get('api_status', {})
            if api_status.get('reason') == 'quota':
                max_parallel = max(1, max_parallel // 2)
            elif api_status.get('ok'):
                # Reexpandir quando janela for liberada
                max_parallel = user_parallel
            rpm_cfg = int(st.session_state.get('rpm_limit', 10))
            auto_split = bool(st.session_state.get('auto_split', True))

            # Badge de alta demanda
            if auto_split and len(ready_tasks) > rpm_cfg:
                st.sidebar.info(f"⚠️ Alta demanda: {len(ready_tasks)} tarefas prontas > RPM {rpm_cfg}. Dividindo em lotes.")

            # Número de tarefas a processar neste lote
            if auto_split:
                chunk_size = max(1, min(len(ready_tasks), max_parallel, rpm_cfg))
            else:
                chunk_size = max(1, min(len(ready_tasks), max_parallel))

            batch_tasks = ready_tasks[:chunk_size]

            # Executa em paralelo as tarefas do lote
            max_workers = min(max_parallel, len(batch_tasks))
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for t in batch_tasks:
                    st.info(f"Executando Tarefa {t['task_id']}: {t['description']}")
                    futures[executor.submit(run_task, t)] = t
                for fut in as_completed(futures):
                    t = futures[fut]
                    task_id = t['task_id']
                    result = None
                    error = None
                    try:
                        task_id_ret, result, error, duration = fut.result()
                    except Exception as e:
                        error = e
                        duration = None

                    if error is None:
                        # Atualiza contexto e UI no thread principal
                        self.shared_context[t['output_variable']] = result
                        if duration is not None:
                            st.success(f"Tarefa {task_id} concluída. Tempo: {duration:.2f}s")
                        else:
                            st.success(f"Tarefa {task_id} concluída.")
                        completed_task_ids.add(task_id)

            # Determinar se o lote envolve chamadas de LLM (por ora, ferramentas são locais; mantenha lista vazia)
            def _is_llm_tool(name: str) -> bool:
                llm_tools = set()  # Placeholder para futuras ferramentas LLM-based
                return name in llm_tools

            llm_in_batch = any(_is_llm_tool(t['tool_to_use']) for t in batch_tasks)

            # Se autosplit e houver LLM no lote, respeitar janela de RPM entre lotes (best-effort)
            if auto_split and llm_in_batch and chunk_size >= rpm_cfg and rpm_cfg > 0:
                import time as _t
                wait_s = max(1, int(60 / rpm_cfg))
                st.info(f"Aguardando {wait_s}s para respeitar o limite de {rpm_cfg} req/min antes do próximo lote (LLM)...")
                _t.sleep(wait_s)

            # Auto-correção: chamar TeamLeader para revisar plano (best-effort)
            if error is not None:
                error_briefing = briefing.copy()
                error_briefing['error'] = f"Tarefa {task_id} falhou: {str(error)}. Revise o plano."
                try:
                    _ = self.team_leader.create_plan(error_briefing)
                    st.warning("Plano revisado gerado devido a erro. Pulando tarefa com erro e seguindo.")
                except Exception:
                    st.error("Falha ao revisar plano. Pulando tarefa.")
                completed_task_ids.add(task_id)

        # Etapa 4: Síntese e Resposta Final
        st.write("4. **Líder de Equipe:** Sintetizando resultados...")
        compact_context = self._compact_shared_context()
        synthesis_report = self.team_leader.synthesize_results(compact_context)
        
        # Contexto de memória
        memory_context = "\n".join(self.memory[-3:])  # Últimas 3 memórias
        
        st.write("5. **Analista de Negócios:** Gerando insights e resposta final...")
        final_response_stream = self.agents["DataAnalystBusinessAgent"].generate_final_response(synthesis_report, memory_context)
        
        # Coletar e renderizar a resposta completa em um único componente de chat
        full_response = stream_response_to_chat(final_response_stream)
        
        # Armazenar resposta completa na memória (sem truncar)
        self.memory.append(f"**Pergunta:** {user_query}\n**Resposta:** {full_response}")
        st.session_state['memory'] = self.memory

        # Botão para baixar relatório em PDF (ABNT + Pirâmide de Minto) - apenas se solicitado
        if 'relatório' in user_query.lower() or 'report' in user_query.lower():
            charts_bytes = []
            if 'charts' in st.session_state and st.session_state.charts:
                for ci in st.session_state.charts:
                    if isinstance(ci, dict) and 'bytes' in ci:
                        charts_bytes.append(ci['bytes'])
            pdf = generate_pdf_report(
                title="Relatório de Análise de Dados",
                user_query=user_query,
                synthesis=synthesis_report,
                full_response=full_response,
                plan=plan,
                charts=charts_bytes,
            )
            st.download_button(
                label="Baixar relatório em PDF (ABNT + Pirâmide de Minto)",
                data=pdf,
                file_name="analysis_report.pdf",
                mime="application/pdf",
            )

# --- Interface Streamlit ---
def main():
    st.set_page_config(page_title="Autonomous Data Consulting", layout="wide")
    st.title("🤖 Autonomous Data Consulting")

    with st.sidebar:
        st.header("LLM Settings")
        default_config = load_config()
        provider = st.selectbox("Provider", ["groq", "openai", "google"], index=["groq", "openai", "google"].index(default_config.get('provider', 'groq')))
        
        # Model options per provider
        models = {
            "google": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
            "openai": ["gpt-5", "gpt-5-nano", "gpt-5-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"],
            "groq": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "meta-llama/llama-guard-4-12b", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "deepseek-r1-distill-llama-70b", "meta-llama/llama-4-maverick-17b-128e-instruct"]
        }
        model_options = models.get(provider, [])
        default_model = default_config.get('model', model_options[0] if model_options else 'llama-3.1-8b-instant')
        index = model_options.index(default_model) if default_model in model_options else 0
        model = st.selectbox("Model", model_options, index=index)
        
        # Do NOT prefill API key from file for security; keep only in session during runtime
        api_key = st.text_input("API Key", value=st.session_state.get('api_key', ''), type="password")
        st.session_state['api_key'] = api_key
        if api_key == "your_api_key_here" or not api_key.strip():
            st.warning("Please configure a valid API key for the selected provider. The system will not work with the placeholder key.")

        rpm_limit = st.slider("Max Requests per Minute (RPM)", 1, 60, value=default_config.get('rpm_limit', 10))
        st.info(f"RPM Limit set to: {rpm_limit}")
        # persist rpm in session for executor awareness
        st.session_state['rpm_limit'] = rpm_limit

        # API status indicator
        if not api_key.strip():
            st.error("API Key ausente. Informe sua chave para executar análises.")
            st.session_state['api_status'] = {'ok': False, 'reason': 'missing_api_key'}
        else:
            st.success("API Key carregada para esta sessão.")
            st.session_state['api_status'] = {'ok': True}

        # Controls to optimize experience under high demand
        st.subheader("Execução e Desempenho")
        max_parallel = st.slider("Tarefas paralelas (máx)", 1, 8, value=st.session_state.get('max_parallel_tasks', 4))
        st.session_state['max_parallel_tasks'] = max_parallel
        auto_split = st.checkbox("Auto-dividir plano quando demanda alta (respeitar RPM)", value=st.session_state.get('auto_split', True))
        st.session_state['auto_split'] = auto_split
        # Quota/Rate-limit signaling
        retry_at = st.session_state.get('retry_at')
        if retry_at:
            import time as _t
            remaining = max(0, int(retry_at - _t.time()))
            total_wait = int(st.session_state.get('wait_window_s', max(1, int(60 / max(1, st.session_state.get('rpm_limit', 10))))))
            if remaining > 0:
                st.warning(f"Aguardando janela de quota. Tente novamente em ~{remaining}s.")
                # Progress visual
                progress_val = 1 - (remaining / max(1, total_wait))
                st.progress(min(max(progress_val, 0.0), 1.0))
                # Botão de retentativa (só efetivo quando tempo zerar)
                if st.button("Tentar novamente agora"):
                    if remaining <= 0:
                        st.session_state.pop('retry_at', None)
                        st.session_state['api_status'] = {'ok': True}
                        st.experimental_rerun()
                    else:
                        st.info("Ainda aguardando janela de quota.")
            else:
                # limpar quando passar
                st.session_state.pop('retry_at', None)

        if st.button("Save Configurations"):
            # Persist only non-sensitive settings; API key remains in session only
            config = {
                'provider': provider,
                'model': model,
                'rpm_limit': rpm_limit
            }
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
            st.success("Configurations (without API key) saved! API key is kept only for this session.")

    # File uploader
    uploaded_files = st.file_uploader("Upload your data", accept_multiple_files=True, type=['csv', 'xlsx', 'xls', 'ods', 'odt'])

    if uploaded_files:
        dataframes = {}
        normalize_cols = st.sidebar.checkbox("Normalize column names (snake_case)", value=True)
        for file in uploaded_files:
            fname = file.name.lower()
            if fname.endswith('.csv'):
                df = pd.read_csv(file)
                if normalize_cols:
                    df = tools.normalize_dataframe_columns(df)
                corrected_df, report = tools.validate_and_correct_data_types(df)
                df = corrected_df
                corrections = {k: v for k, v in report.items() if 'Converted' in v}
                if corrections:
                    st.info(f"Correções de tipos de dados para {file.name}: {corrections}")
                dataframes[file.name] = df
            elif fname.endswith('.xlsx'):
                df = pd.read_excel(file, engine='openpyxl')
                if normalize_cols:
                    df = tools.normalize_dataframe_columns(df)
                corrected_df, report = tools.validate_and_correct_data_types(df)
                df = corrected_df
                corrections = {k: v for k, v in report.items() if 'Converted' in v}
                if corrections:
                    st.info(f"Correções de tipos de dados para {file.name}: {corrections}")
                dataframes[file.name] = df
            elif fname.endswith('.xls'):
                # Requires xlrd
                df = pd.read_excel(file, engine='xlrd')
                if normalize_cols:
                    df = tools.normalize_dataframe_columns(df)
                corrected_df, report = tools.validate_and_correct_data_types(df)
                df = corrected_df
                corrections = {k: v for k, v in report.items() if 'Converted' in v}
                if corrections:
                    st.info(f"Correções de tipos de dados para {file.name}: {corrections}")
                dataframes[file.name] = df
            elif fname.endswith('.ods'):
                df = pd.read_excel(file, engine='odf')
                if normalize_cols:
                    df = tools.normalize_dataframe_columns(df)
                corrected_df, report = tools.validate_and_correct_data_types(df)
                df = corrected_df
                corrections = {k: v for k, v in report.items() if 'Converted' in v}
                if corrections:
                    st.info(f"Correções de tipos de dados para {file.name}: {corrections}")
                dataframes[file.name] = df
            elif fname.endswith('.odt'):
                try:
                    tables = tools.read_odt_tables(file)
                    if tables:
                        for tname, tdf in tables.items():
                            if normalize_cols:
                                tdf = tools.normalize_dataframe_columns(tdf)
                            corrected_tdf, report = tools.validate_and_correct_data_types(tdf)
                            tdf = corrected_tdf
                            corrections = {k: v for k, v in report.items() if 'Converted' in v}
                            if corrections:
                                st.info(f"Correções de tipos de dados para {file.name}::{tname}: {corrections}")
                            dataframes[f"{file.name}::{tname}"] = tdf
                    else:
                        st.warning(f"No tables found in {file.name}. ODT only supports tables.")
                except Exception as e:
                    st.error(f"Failed to read {file.name} (ODT): {e}")
        st.session_state.dataframes = dataframes
        st.write("Data loaded:", list(dataframes.keys()))

    # If there are dataframes, allow selecting the default
    default_df_key = None
    if 'dataframes' in st.session_state and st.session_state.dataframes:
        keys = list(st.session_state.dataframes.keys())
        default_df_key = st.sidebar.selectbox("Default DataFrame", keys, index=0)
        st.session_state['default_df_key'] = default_df_key

        # If multiple files, ask about relation and allow defining keys
        if len(keys) >= 2:
            related = st.sidebar.checkbox("Are the datasets related?", value=False)
            if related:
                left_df_key = st.sidebar.selectbox("Left dataset (left)", keys, index=0, key="left_df_key")
                right_df_key = st.sidebar.selectbox("Right dataset (right)", keys, index=1, key="right_df_key")
                left_cols = list(st.session_state.dataframes[left_df_key].columns)
                right_cols = list(st.session_state.dataframes[right_df_key].columns)
                # Try to suggest common key
                commons = [c for c in left_cols if c in right_cols]
                use_same_key = st.sidebar.checkbox("Same key column in both?", value=bool(commons))
                if use_same_key and commons:
                    same_key = st.sidebar.selectbox("Key column (common)", commons, index=0)
                    st.session_state['join_spec'] = {
                        'left_df_key': left_df_key,
                        'right_df_key': right_df_key,
                        'left_on': same_key,
                        'right_on': same_key,
                        'how': st.sidebar.selectbox("Join type", ["inner", "left", "right", "outer"], index=0)
                    }
                else:
                    left_on = st.sidebar.selectbox("Left dataset key (left_on)", left_cols, index=0)
                    right_on = st.sidebar.selectbox("Right dataset key (right_on)", right_cols, index=0)
                    st.session_state['join_spec'] = {
                        'left_df_key': left_df_key,
                        'right_df_key': right_df_key,
                        'left_on': left_on,
                        'right_on': right_on,
                        'how': st.sidebar.selectbox("Join type", ["inner", "left", "right", "outer"], index=0, key="how_join")
                    }

        # Option to clean old chart files
        with st.sidebar.expander("Maintenance"):
            if st.button("Clean old chart files (plot_*.png)"):
                removed = cleanup_old_plot_files()
                st.success(f"Files removed: {removed}")

        # Preview of default DataFrame: header, 4 rows, types and potential keys
        with st.expander("Preview of Default DataFrame (header, 4 rows, types and potential keys)"):
            df_preview = st.session_state.dataframes[default_df_key]
            st.markdown("**Header (column names):**")
            st.write(list(df_preview.columns))
            st.markdown("**Sample (4 rows):**")
            st.dataframe(df_preview.head(4))
            st.markdown("**Inferred types per column:**")
            dtypes_df = pd.DataFrame({
                'coluna': df_preview.columns,
                'dtype': df_preview.dtypes.astype(str).values,
            })
            st.dataframe(dtypes_df, hide_index=True)

            # Uniqueness metrics to suggest keys
            st.markdown("**Uniqueness per column (candidates for keys):**")
            total_rows = max(1, len(df_preview))
            nunique = df_preview.nunique(dropna=True)
            unique_ratio = (nunique / total_rows).fillna(0).round(4)
            keys_df = pd.DataFrame({
                'coluna': df_preview.columns,
                'valores_unicos': nunique.values,
                'linhas_total': total_rows,
                'unicidade': unique_ratio.values,
            })
            # Suggestion: uniqueness > 0.9
            candidates = keys_df[keys_df['unicidade'] > 0.9]['coluna'].tolist()
            st.dataframe(keys_df.sort_values('unicidade', ascending=False), hide_index=True)
            if candidates:
                st.info(f"Possible keys (uniqueness > 0.9): {candidates}")

        # Button to test the configured join (if any)
        if 'join_spec' in st.session_state:
            st.subheader("Test Configured Join")
            if st.button("Test Join"):
                js = st.session_state['join_spec']
                left_key = js.get('left_df_key')
                right_key = js.get('right_df_key')
                if left_key in st.session_state.dataframes and right_key in st.session_state.dataframes:
                    df1 = st.session_state.dataframes[left_key]
                    df2 = st.session_state.dataframes[right_key]
                    left_on = js.get('left_on')
                    right_on = js.get('right_on')
                    how = js.get('how', 'inner')
                    try:
                        if left_on == right_on:
                            preview_df = tools.join_datasets(df1, df2, on_column=left_on)
                        else:
                            preview_df = tools.join_datasets_on(df1, df2, left_on=left_on, right_on=right_on, how=how)
                        st.success(f"Join successful. Shape: {preview_df.shape}")
                        st.dataframe(preview_df.head(10))
                    except Exception as e:
                        st.error(f"Failed to test join: {e}")

    # Display memory (complete responses)
    with st.expander("Memory of Previous Analyses (complete responses)"):
        if 'memory' in st.session_state and st.session_state['memory']:
            for i, mem in enumerate(st.session_state['memory'], 1):
                st.markdown(f"**{i}.**")
                st.write(mem)
        else:
            st.write("No previous analysis stored.")

    if 'dataframes' in st.session_state and st.session_state.dataframes:
        if prompt := st.chat_input("Faça uma pergunta sobre seus dados (ex.: 'Execute uma EDA completa no dataset')..."):
            st.chat_message("user").markdown(prompt)
            
            config = {'provider': provider, 'model': model, 'api_key': api_key}
            try:
                llm = obtain_llm(config)
            except Exception as e:
                st.session_state['api_status'] = {'ok': False, 'reason': 'llm_init_error', 'error': str(e)}
                st.error(f"Falha ao inicializar LLM: {e}")
                st.stop()

            pipeline = AnalysisPipeline(llm, st.session_state.dataframes, rpm_limit)
            
            with st.chat_message("assistant"):
                try:
                    pipeline.run(prompt)
                except Exception as e:
                    msg = str(e).lower()
                    import time as _t
                    if any(k in msg for k in ["rate limit", "quota", "429", "exceeded", "too many requests"]):
                        # Set retry after based on RPM (best-effort): wait window of 60/rpm seconds
                        wait_s = max(1, int(60 / max(1, rpm_limit)))
                        st.session_state['retry_at'] = _t.time() + wait_s
                        st.session_state['wait_window_s'] = wait_s
                        st.session_state['api_status'] = {'ok': False, 'reason': 'quota', 'wait_s': wait_s}
                        st.warning(f"Cota/Rate limit atingido. Sugerimos reduzir tarefas paralelas e aguardar {wait_s}s antes de tentar novamente.")
                        st.info("Dica: habilite 'Auto-dividir plano' e diminua 'Tarefas paralelas (máx)' na barra lateral.")
                    elif any(k in msg for k in ["api key", "invalid key", "authentication"]):
                        st.session_state['api_status'] = {'ok': False, 'reason': 'auth', 'error': str(e)}
                        st.error("Problema de autenticação com a API Key. Verifique a chave informada.")
                    else:
                        st.error(f"Falha na execução: {e}")

        # Renders saved charts (bytes in memory or old paths)
                if 'charts' in st.session_state and st.session_state.charts:
                    for idx, chart_item in enumerate(st.session_state.charts, start=1):
                        if isinstance(chart_item, dict) and 'bytes' in chart_item:
                            st.image(chart_item['bytes'], caption=chart_item.get('caption'))
                            st.download_button(
                                label=f"Download chart {idx}",
                                data=chart_item['bytes'],
                                file_name=f"chart_{idx}.png",
                                mime="image/png",
                            )
                        else:
                            # Legacy path: check existence before trying to render
                            if isinstance(chart_item, str) and os.path.exists(chart_item):
                                with open(chart_item, 'rb') as f:
                                    img_bytes = f.read()
                                st.image(img_bytes)
                                st.download_button(
                                    label=f"Download chart {idx}",
                                    data=img_bytes,
                                    file_name=os.path.basename(chart_item),
                                    mime="image/png",
                                )
                            else:
                                st.warning(f"Image not found (ignored): {chart_item}")
                    del st.session_state.charts  # Clear for next execution

if __name__ == "__main__":
    main()
