# Data Processing Diagram

```mermaid
flowchart TD
    A[User uploads datasets] --> B[Preprocessing: Column normalization, Default DataFrame selection]
    B --> C[Relationship and join configuration: Keys, join type, join test]
    C --> D[User asks question in chat]
    D --> E[OrchestratorAgent: Strict JSON briefing + Pydantic validation and auto-correction]
    E --> F[TeamLeaderAgent: Execution plan (JSON) + Pydantic validation and auto-correction]
    F --> G[Plan normalization in app.py (variations like tasks/execution_plan/project)]
    G --> H[Expander: Column selection/confirmation for chart tasks]
    H --> I[Execution: Agents call tools; selective retry and cascading re-execution if outputs change]
    I --> J[Results stored in shared_context]
    J --> K[Team Leader synthesizes results (compact context)]
    K --> L[QA Review: suggestions incorporated into context]
    L --> M[DataAnalystBusinessAgent: Final response with conclusions]
    M --> N[Full response stored in session memory]
    N --> O[In-memory charts + download]
    O --> P[PDF: ABNT-like + Minto Pyramid (lazy reportlab)]
    P --> Q[Analytics: success/error by tool, mean duration, frequent error inputs]
    Q --> R[Logs: optional JSON save to logs/]
    R --> S[End of processing]
```
