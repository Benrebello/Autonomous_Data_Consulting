# prompt_templates.py
"""Dynamic prompt templates based on tools used in analysis and agent profiles."""

from typing import List, Dict, Any, Optional

# Tool-specific interpretation guidelines
TOOL_INTERPRETATION_GUIDES = {
    # Correlation tools
    "correlation_matrix": """
**Interpretação de Correlações:**
- |r| < 0.1: NEGLIGÍVEL - não mencione como tendência
- 0.1 ≤ |r| < 0.3: FRACA - mencione com extrema cautela
- 0.3 ≤ |r| < 0.5: MODERADA
- 0.5 ≤ |r| < 0.7: FORTE
- |r| ≥ 0.7: MUITO FORTE
- Sempre verifique p-value (p < 0.05 = significativo)
- Correlação ≠ Causalidade

**Exemplos Few-Shot:**

❌ INCORRETO:
Query: "Qual a correlação entre ano e nota?"
Resultado: r=0.05, p=0.6
Resposta ERRADA: "Há uma tendência positiva de que filmes mais recentes têm notas melhores"
Problema: Correlação negligível sendo interpretada como tendência

❌ INCORRETO:
Query: "Filmes mais longos têm mais votos?"
Resultado: r=-0.21, p=0.04
Resposta ERRADA: "Sim, filmes mais longos recebem significativamente menos votos"
Problema: Correlação fraca sendo interpretada como relação forte

✅ CORRETO:
Query: "Qual a correlação entre ano e nota?"
Resultado: r=0.05, p=0.6
Resposta CORRETA: "Não há correlação significativa entre ano de lançamento e nota (r=0.05, p=0.6). 
A correlação é negligível e não estatisticamente significativa. O ano não é um preditor relevante da nota."

✅ CORRETO:
Query: "Filmes mais longos têm mais votos?"
Resultado: r=-0.21, p=0.04
Resposta CORRETA: "Há uma correlação negativa fraca mas estatisticamente significativa (r=-0.21, p=0.04). 
Embora exista uma tendência de filmes mais longos receberem menos votos, a correlação é fraca, 
indicando que outros fatores (qualidade, gênero, marketing) são muito mais importantes que a duração."
""",
    
    "correlation_tests": """
**Testes de Correlação Múltiplos:**
- Pearson: mede relação LINEAR
- Spearman: mede relação MONOTÔNICA (não necessariamente linear)
- Kendall: robusto a outliers
- Use Spearman/Kendall se dados não forem normalmente distribuídos
- Sempre reporte qual teste foi usado
""",
    
    # Outlier detection
    "detect_outliers": """
**Detecção de Outliers:**
- IQR: identifica valores extremos (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
- Z-score: identifica valores > 3 desvios padrão da média
- Outliers NÃO são necessariamente erros - podem ser insights valiosos
- Sempre investigue o CONTEXTO antes de remover outliers
- Mencione quantos outliers foram encontrados e sua % do total
""",
    
    "detect_and_remove_outliers": """
**Remoção de Outliers:**
- CUIDADO: remoção pode distorcer análises
- Só remova se houver justificativa de negócio
- Sempre reporte quantos foram removidos
- Considere análise com e sem outliers
""",
    
    # Statistical tests
    "perform_t_test": """
**Teste t:**
- Compara MÉDIAS de dois grupos
- Pressupõe distribuição normal
- p < 0.05: diferença é estatisticamente significativa
- Sempre reporte: t-statistic, p-value, tamanhos dos grupos
- Significância estatística ≠ significância prática
""",
    
    "perform_chi_square": """
**Teste Qui-Quadrado:**
- Testa INDEPENDÊNCIA entre variáveis categóricas
- p < 0.05: variáveis são dependentes (relacionadas)
- Requer frequências esperadas ≥ 5 em cada célula
- Sempre reporte graus de liberdade
""",
    
    "perform_anova": """
**ANOVA:**
- Compara médias de 3+ grupos
- p < 0.05: pelo menos um grupo difere significativamente
- NÃO indica QUAL grupo difere (use post-hoc tests)
- Pressupõe: normalidade, homogeneidade de variâncias
""",
    
    # Machine Learning
    "linear_regression": """
**Regressão Linear:**
- R²: % de variância explicada (0-1, maior = melhor)
- R² < 0.3: modelo fraco
- R² 0.3-0.5: modelo moderado
- R² > 0.7: modelo forte
- Sempre verifique resíduos para validar pressupostos
- Coeficientes indicam direção e magnitude do efeito
""",
    
    "logistic_regression": """
**Regressão Logística:**
- Para variáveis dependentes BINÁRIAS (0/1, sim/não)
- Accuracy: % de predições corretas
- Precision: % de positivos preditos que são realmente positivos
- Recall: % de positivos reais que foram identificados
- F1-score: média harmônica de precision e recall
""",
    
    "random_forest_classifier": """
**Random Forest:**
- Ensemble de árvores de decisão
- Robusto a outliers e não-linearidades
- Feature importance: indica variáveis mais importantes
- Accuracy > 0.8: geralmente bom, mas depende do contexto
- Cuidado com overfitting em datasets pequenos
""",
    
    "gradient_boosting_classifier": """
**Gradient Boosting:**
- Modelo avançado, geralmente alta performance
- Feature importance: quais variáveis mais contribuem
- Mais propenso a overfit que Random Forest
- Requer tuning cuidadoso de hiperparâmetros
""",
    
    # Clustering
    "run_kmeans_clustering": """
**K-Means Clustering:**
- Agrupa dados em K clusters
- Escolha de K: use elbow method ou silhouette score
- Clusters NÃO têm significado pré-definido - você deve interpretá-los
- Sempre descreva características de cada cluster
- Sensível a escala - normalize dados primeiro
""",
    
    # Time Series
    "decompose_time_series": """
**Decomposição de Séries Temporais:**
- Trend: tendência de longo prazo
- Seasonal: padrões que se repetem em intervalos fixos
- Residual: variação não explicada
- Aditivo: componentes somam (Y = T + S + R)
- Multiplicativo: componentes multiplicam (Y = T * S * R)
""",
    
    "forecast_arima": """
**ARIMA Forecasting:**
- AR (AutoRegressive): usa valores passados
- I (Integrated): diferenciação para tornar série estacionária
- MA (Moving Average): usa erros passados
- Sempre reporte intervalo de confiança das previsões
- Previsões de longo prazo são menos confiáveis
""",
    
    # Data Quality
    "data_profiling": """
**Profiling de Dados:**
- Quality score: 0-100 (baseado em dados faltantes)
- Score > 90: excelente qualidade
- Score 70-90: boa qualidade
- Score < 70: problemas significativos
- Sempre mencione colunas com >20% de dados faltantes
""",
    
    "missing_data_analysis": """
**Análise de Dados Faltantes:**
- MCAR: Missing Completely At Random (melhor caso)
- MAR: Missing At Random (padrão identificável)
- MNAR: Missing Not At Random (pior caso)
- >50% faltante: considere remover coluna
- 20-50%: use imputação avançada
- <20%: imputação simples (média/mediana/moda)
""",
    
    "distribution_tests": """
**Testes de Normalidade:**
- Shapiro-Wilk: mais poderoso para n < 5000
- Kolmogorov-Smirnov: para amostras maiores
- p < 0.05: dados NÃO são normalmente distribuídos
- Skewness: assimetria (0 = simétrico, >0 = cauda direita, <0 = cauda esquerda)
- Kurtosis: "peso" das caudas (0 = normal, >0 = caudas pesadas, <0 = caudas leves)
""",
    
    # Business Analytics
    "rfm_analysis": """
**Análise RFM:**
- Recency: quão recente foi a última compra (menor = melhor)
- Frequency: quantas vezes comprou (maior = melhor)
- Monetary: quanto gastou (maior = melhor)
- Scores 1-5 para cada dimensão
- Champions: 555, 554, 544, 545 (melhores clientes)
- At Risk: clientes com R baixo mas F/M alto
""",
    
    "ab_test_analysis": """
**Teste A/B:**
- p-value: probabilidade de diferença ser por acaso
- p < 0.05: diferença estatisticamente significativa
- Cohen's d: tamanho do efeito
  - d < 0.2: efeito negligível
  - 0.2 ≤ d < 0.5: efeito pequeno
  - 0.5 ≤ d < 0.8: efeito médio
  - d ≥ 0.8: efeito grande
- Significância estatística ≠ significância prática
""",
    
    # Feature Engineering
    "create_polynomial_features": """
**Features Polinomiais:**
- Captura relações não-lineares
- Degree 2: adiciona x², x*y
- Degree 3: adiciona x³, x²*y, x*y²
- CUIDADO: aumenta muito o número de features
- Sempre normalize após criar features polinomiais
""",
    
    "multicollinearity_detection": """
**Detecção de Multicolinearidade (VIF):**
- VIF < 5: baixa multicolinearidade (OK)
- 5 ≤ VIF < 10: moderada (investigar)
- VIF ≥ 10: alta (remover variável ou usar PCA)
- Multicolinearidade infla erros padrão dos coeficientes
- Não afeta predições, mas dificulta interpretação
""",
}


def get_tool_context(tool_name: str) -> str:
    """Get interpretation guidelines for a specific tool."""
    return TOOL_INTERPRETATION_GUIDES.get(tool_name, "")


def get_combined_context(tools_used: List[str]) -> str:
    """Get combined interpretation guidelines for multiple tools."""
    contexts = []
    for tool in tools_used:
        context = get_tool_context(tool)
        if context:
            contexts.append(f"**{tool}:**{context}")
    
    if not contexts:
        return ""
    
    return "\n".join(contexts)


def enhance_synthesis_prompt(base_prompt: str, tools_used: List[str]) -> str:
    """Enhance synthesis prompt with tool-specific guidelines."""
    tool_context = get_combined_context(tools_used)
    
    if not tool_context:
        return base_prompt
    
    # Insert tool context before execution results
    enhanced = base_prompt.replace(
        "**Resultados da Execução:**",
        f"**Diretrizes Específicas das Ferramentas Usadas:**\n{tool_context}\n\n**Resultados da Execução:**"
    )
    
    return enhanced


def enhance_final_response_prompt(base_prompt: str, tools_used: List[str]) -> str:
    """Enhance final response prompt with tool-specific guidelines."""
    tool_context = get_combined_context(tools_used)
    
    if not tool_context:
        return base_prompt
    
    # Insert tool context before technical draft
    enhanced = base_prompt.replace(
        "**Rascunho Técnico:**",
        f"**Contexto das Ferramentas Usadas:**\n{tool_context}\n\n**Rascunho Técnico:**"
    )
    
    return enhanced


# Category-based guidelines for when specific tool is not found
CATEGORY_GUIDELINES = {
    "statistical_tests": """
**Testes Estatísticos em Geral:**
- p-value < 0.05: resultado estatisticamente significativo
- Sempre reporte: estatística do teste, p-value, tamanho da amostra
- Verifique pressupostos do teste (normalidade, homogeneidade, etc.)
- Significância estatística ≠ significância prática
- Considere poder do teste e tamanho do efeito
""",
    
    "machine_learning": """
**Machine Learning em Geral:**
- Sempre divida dados em treino/teste
- Reporte múltiplas métricas (accuracy, precision, recall, F1)
- Cuidado com overfitting (modelo muito complexo)
- Cuidado com underfitting (modelo muito simples)
- Feature importance: quais variáveis mais contribuem
- Valide pressupostos e interprete resultados no contexto do negócio
""",
    
    "visualization": """
**Visualizações:**
- Escolha o gráfico apropriado para o tipo de dado
- Histograma: distribuição de uma variável contínua
- Boxplot: identificar outliers e quartis
- Scatter: relação entre duas variáveis contínuas
- Heatmap: matriz de correlações
- Sempre adicione títulos e labels claros
""",
    
    "time_series": """
**Análise de Séries Temporais:**
- Verifique estacionariedade (ADF test)
- Identifique tendência, sazonalidade, ciclos
- Autocorrelação: valores correlacionados com seus lags
- Previsões: sempre inclua intervalos de confiança
- Quanto mais distante no futuro, menos confiável a previsão
""",
}


def get_category_for_tool(tool_name: str) -> str:
    """Determine category for a tool."""
    statistical_tests = ['perform_t_test', 'perform_chi_square', 'perform_anova', 
                         'perform_kruskal_wallis', 'perform_bayesian_inference']
    ml_tools = ['linear_regression', 'logistic_regression', 'random_forest_classifier',
                'svm_classifier', 'knn_classifier', 'gradient_boosting_classifier']
    viz_tools = ['plot_histogram', 'plot_boxplot', 'plot_scatter', 'plot_heatmap',
                 'plot_line_chart', 'plot_violin_plot', 'generate_chart']
    ts_tools = ['decompose_time_series', 'forecast_arima', 'forecast_time_series_arima',
                'get_temporal_patterns', 'add_time_features_from_seconds']
    
    if tool_name in statistical_tests:
        return "statistical_tests"
    elif tool_name in ml_tools:
        return "machine_learning"
    elif tool_name in viz_tools:
        return "visualization"
    elif tool_name in ts_tools:
        return "time_series"
    
    return None


def get_fallback_context(tools_used: List[str]) -> str:
    """Get fallback category-based context for tools without specific guidelines."""
    categories_found = set()
    
    for tool in tools_used:
        if tool not in TOOL_INTERPRETATION_GUIDES:
            category = get_category_for_tool(tool)
            if category:
                categories_found.add(category)
    
    contexts = []
    for category in categories_found:
        if category in CATEGORY_GUIDELINES:
            contexts.append(CATEGORY_GUIDELINES[category])
    
    return "\n".join(contexts) if contexts else ""


# ============================================================================
# AGENT PROFILES - System Prompts Dinâmicos
# ============================================================================

AGENT_PROFILES = {
    "OrchestratorAgent": {
        "role": "Orquestrador de IA",
        "specialty": "Comunicação e entendimento de negócios",
        "system_prompt": """Você é um Orquestrador de IA, o principal ponto de contato com o usuário.
Sua especialidade é comunicação e entendimento de negócios.
Sua missão é traduzir perguntas vagas em briefings estruturados.""",
        "guidelines": [
            "Classifique a intenção do usuário (eda, simple_analysis, etc.)",
            "Identifique arquivos e perguntas-chave",
            "Retorne APENAS JSON válido, sem markdown",
        ]
    },
    
    "TeamLeaderAgent": {
        "role": "Líder de Equipe de IA",
        "specialty": "Gerenciamento de projetos de dados",
        "system_prompt": """Você é o Líder de Equipe de IA, um gerente de projetos de dados sênior.
Você recebe briefings e cria planos de execução detalhados.
Você também sintetiza resultados em relatórios técnicos coesos.""",
        "guidelines": [
            "Decomponha tarefas complexas em atômicas",
            "Delegue para o agente especialista apropriado",
            "Sintetize resultados com rigor técnico",
            "Sempre verifique pressupostos estatísticos",
        ]
    },
    
    "DataArchitectAgent": {
        "role": "Arquiteto de Dados",
        "specialty": "Limpeza, transformação e integração de dados",
        "system_prompt": """Você é um Arquiteto de Dados especializado em ETL e qualidade de dados.
Sua missão é preparar dados para análise.""",
        "guidelines": [
            "Priorize qualidade e consistência dos dados",
            "Documente todas as transformações aplicadas",
            "Identifique e trate dados faltantes apropriadamente",
            "Normalize e padronize quando necessário",
        ]
    },
    
    "DataAnalystTechnicalAgent": {
        "role": "Analista de Dados Técnico",
        "specialty": "Análise estatística e EDA profunda",
        "system_prompt": """Você é um Analista de Dados Técnico com forte background estatístico.
Você realiza análises exploratórias rigorosas e testes estatísticos.""",
        "guidelines": [
            "Sempre verifique pressupostos antes de aplicar testes",
            "Reporte p-values e intervalos de confiança",
            "Diferencie significância estatística de prática",
            "Use visualizações para validar análises",
        ]
    },
    
    "DataAnalystBusinessAgent": {
        "role": "Analista de Dados de Negócios",
        "specialty": "Tradução de dados em insights acionáveis",
        "system_prompt": """Você é um Analista de Dados focado em Negócios.
Você traduz análises técnicas em insights claros e acionáveis para stakeholders.""",
        "guidelines": [
            "Fale em linguagem de negócios, não técnica",
            "Foque em insights acionáveis",
            "Use analogias e exemplos práticos",
            "Destaque impacto e recomendações",
            "Seja direto e objetivo",
        ]
    },
    
    "DataScientistAgent": {
        "role": "Cientista de Dados",
        "specialty": "Machine Learning e modelagem preditiva",
        "system_prompt": """Você é um Cientista de Dados especializado em ML e modelagem preditiva.
Você aplica algoritmos avançados para descobrir padrões e fazer previsões.""",
        "guidelines": [
            "Valide modelos com train/test split",
            "Reporte múltiplas métricas de performance",
            "Explique feature importance",
            "Cuidado com overfitting e underfitting",
            "Interprete resultados no contexto do negócio",
        ]
    },
}


def get_agent_system_prompt(agent_name: str, tools_used: Optional[List[str]] = None) -> str:
    """Get dynamic system prompt for an agent based on their profile and tools used.
    
    Args:
        agent_name: Name of the agent (e.g., 'TeamLeaderAgent')
        tools_used: List of tools that will be used (optional)
    
    Returns:
        Complete system prompt with agent profile + tool guidelines
    """
    profile = AGENT_PROFILES.get(agent_name, {})
    
    if not profile:
        return ""
    
    # Base system prompt
    system_prompt = f"{profile['system_prompt']}\n\n"
    
    # Add role and specialty
    system_prompt += f"**Seu Papel:** {profile['role']}\n"
    system_prompt += f"**Sua Especialidade:** {profile['specialty']}\n\n"
    
    # Add general guidelines
    if profile.get('guidelines'):
        system_prompt += "**Diretrizes Gerais:**\n"
        for guideline in profile['guidelines']:
            system_prompt += f"- {guideline}\n"
        system_prompt += "\n"
    
    # Add tool-specific context if tools are provided
    if tools_used:
        tool_context = get_combined_context(tools_used)
        if tool_context:
            system_prompt += "**Diretrizes Específicas das Ferramentas:**\n"
            system_prompt += tool_context + "\n"
        
        # Add fallback category context
        fallback = get_fallback_context(tools_used)
        if fallback:
            system_prompt += fallback + "\n"
    
    return system_prompt


def build_prompt_with_context(
    agent_name: str,
    base_template: str,
    tools_used: Optional[List[str]] = None,
    **kwargs
) -> str:
    """Build a complete prompt with agent context and tool guidelines.
    
    Args:
        agent_name: Name of the agent
        base_template: Base prompt template with placeholders
        tools_used: List of tools used
        **kwargs: Variables to format into the template
    
    Returns:
        Complete formatted prompt
    """
    # Get system prompt for agent
    system_prompt = get_agent_system_prompt(agent_name, tools_used)
    
    # Inject system prompt at the beginning of base template
    if system_prompt and not base_template.startswith(system_prompt):
        full_template = system_prompt + "\n" + base_template
    else:
        full_template = base_template
    
    # Format with provided variables
    try:
        return full_template.format(**kwargs)
    except KeyError as e:
        # If formatting fails, return template with system prompt
        return full_template


# Convenience functions for specific agents
def get_orchestrator_prompt(user_query: str) -> str:
    """Get orchestrator prompt with agent profile."""
    from prompts import ORCHESTRATOR_PROMPT
    return build_prompt_with_context(
        "OrchestratorAgent",
        ORCHESTRATOR_PROMPT,
        user_query=user_query
    )


def get_team_leader_plan_prompt(briefing: str) -> str:
    """Get team leader planning prompt with agent profile."""
    from prompts import TEAM_LEADER_PROMPT
    return build_prompt_with_context(
        "TeamLeaderAgent",
        TEAM_LEADER_PROMPT,
        briefing=briefing
    )


def get_synthesis_prompt(execution_results: dict, tools_used: Optional[List[str]] = None) -> str:
    """Get synthesis prompt with agent profile and tool context."""
    from prompts import SYNTHESIS_PROMPT
    import json
    return build_prompt_with_context(
        "TeamLeaderAgent",
        SYNTHESIS_PROMPT,
        tools_used=tools_used,
        execution_results=json.dumps(execution_results, default=str, indent=2)
    )


def get_final_response_prompt(
    synthesis_report: str,
    memory_context: str,
    tools_used: Optional[List[str]] = None
) -> str:
    """Get final response prompt with agent profile and tool context."""
    from prompts import FINAL_RESPONSE_PROMPT
    return build_prompt_with_context(
        "DataAnalystBusinessAgent",
        FINAL_RESPONSE_PROMPT,
        tools_used=tools_used,
        synthesis_report=synthesis_report,
        memory_context=memory_context
    )
