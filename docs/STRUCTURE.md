# Project Structure Diagram

```mermaid
graph TD
    A[app.py - Streamlit Web Interface] --> B[agents.py - Agent Definitions]
    A --> C[tools.py - Specialized Tools]
    A --> D[prompts.py - Prompt Templates]
    A --> E[config.py - LLM Configuration]

    B --> F[OrchestratorAgent - Translates questions into briefings]
    B --> G[TeamLeaderAgent - Creates execution plans]
    B --> H[DataArchitectAgent - Cleans and joins datasets]
    B --> I[DataAnalystTechnicalAgent - Statistical analyses]
    B --> J[DataAnalystBusinessAgent - Generates graphs and insights]
    B --> K[DataScientistAgent - Machine learning]

    C --> L[Engineering Tools: join_datasets, clean_data]
    C --> M[EDA Tools: descriptive_stats, detect_outliers]
    C --> N[Visualization Tools: plot_histogram, generate_chart]
    C --> O[ML Tools: run_kmeans_clustering]
    C --> P[Utilities: read_odt_tables, normalize_dataframe_columns]

    D --> Q[Templates for briefing, plan, synthesis and QA]

    E --> R[LLM Integration: Groq, OpenAI, Gemini]

    S[config.json - Config file] --> E
    V[plan_cache (session)] --> A
    T[requirements.txt - Dependencies] --> U[streamlit, pandas, langchain, etc.]
```
