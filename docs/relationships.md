# Relationships Diagram between Agents and Tools

```mermaid
flowchart TD
    subgraph "Orchestration Agents"
        A[OrchestratorAgent]
        B[TeamLeaderAgent]
    end

    subgraph "Specialized Agents"
        C[DataArchitectAgent]
        D[DataAnalystTechnicalAgent]
        E[DataAnalystBusinessAgent]
        F[DataScientistAgent]
    end

    subgraph "Engineering Tools"
        G[join_datasets]
        H[clean_data]
        I[normalize_dataframe_columns]
    end

    subgraph "EDA Tools"
        J[descriptive_stats]
        K[detect_outliers]
        L[correlation_matrix]
        M[get_exploratory_analysis]
    end

    subgraph "Visualization Tools"
        N[plot_histogram]
        O[plot_boxplot]
        P[plot_scatter]
        Q[generate_chart]
    end

    subgraph "ML Tools"
        R[run_kmeans_clustering]
    end

    subgraph "Utilities"
        S[read_odt_tables]
    end

    A --> B
    B --> C
    B --> D
    B --> E
    B --> F

    C --> G
    C --> H
    C --> I

    D --> J
    D --> K
    D --> L
    D --> M

    E --> N
    E --> O
    E --> P
    E --> Q

    F --> R

    C --> S
    D --> S
    E --> S
    F --> S

    T[app.py - Interface and Execution] --> A
    T --> B
    T --> C
    T --> D
    T --> E
    T --> F

    U[tools.py - Tools Library] --> G
    U --> H
    U --> I
    U --> J
    U --> K
    U --> L
    U --> M
    U --> N
    U --> O
    U --> P
    U --> Q
    U --> R
    U --> S
```

## Update Notes

- QA Review: while not shown as a separate agent in this diagram, the QA step is conducted via `prompts.py` (QA Review Prompt) between the `TeamLeaderAgent` synthesis and the `DataAnalystBusinessAgent` final response.
- Selective retry + cascading re-execution: when a tool fails, the `TeamLeaderAgent` is asked to correct the plan. The task is retried once and, if outputs change, already-completed dependents are reprocessed (logic in `app.py`).
- Success plan cache: error-free plans may be reused according to the Sidebar configuration (implemented in `app.py` using `st.session_state['plan_cache']`).
