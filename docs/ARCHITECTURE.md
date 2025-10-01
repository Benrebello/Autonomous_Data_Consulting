# System Architecture and Design Rationale

This document describes the architecture, design decisions, and processing flow of the autonomous multi-agent EDA application. It explains how inputs flow through the system, how each layer collaborates, and why specific technical choices were made.

## High-Level Overview

- The application is a Streamlit web app (`app.py`) that orchestrates a team of specialized AI agents (`agents.py`) using structured prompts (`prompts.py`) and domain tools (`tools.py`).
- Users upload datasets (CSV, XLSX/XLS, ODS, ODT), configure relationships/joins, and ask for analyses via chat.
- A multi-step pipeline runs:
  1. Orchestrator converts the user query into a structured "Project Brief" (JSON) and it is validated with Pydantic; auto-correction loop is triggered on schema errors.
  2. Team Leader converts the brief into an "Execution Plan" (JSON), validated with Pydantic; auto-correction loop is triggered on schema errors.
  3. Agents execute the plan by calling functions in `tools.py`, with selective retry on failure and cascading re-execution of dependents when a retried task changes outputs.
  4. Team Leader synthesizes results; then a QA Review step critically evaluates the draft and suggestions are appended to memory context.
  5. Business Analyst returns the final answer; the app stores the full answer in memory and offers PDF export (ABNT-like + Minto Pyramid).

## Modules and Responsibilities

- `app.py` (UI + Orchestration)
  - Manages file uploads, sidebar configuration, and chat UI.
  - Normalizes the plan structure and executes tasks using the mapped tools.
  - Column-selection UI expander for chart tasks (confirm X/Y/column and chart type when applicable).
  - Selective retry with auto-correction and cascading re-execution of dependents.
  - Execution log (structured) with optional JSON persistence and analytics panel (success/error rate by tool, mean duration, frequent error inputs).
  - Aggregates results, handles chart rendering, and produces a downloadable PDF report.
  - Stores prior full analyses in session memory.

- `agents.py` (LLM Agent Layer)
  - Defines agents and their responsibilities: Orchestrator, Team Leader, Technical Analyst, Business Analyst, Data Scientist.
  - Pydantic models enforce schemas for Briefing and Execution Plan with an auto-correction loop on validation failure.
  - Provides `clean_json_response()` to robustly extract valid JSON from the LLM outputs (handles markdown fences, preambles, extra text, and balanced braces).

- `prompts.py` (Prompt Templates)
  - Strict JSON-only prompts to reduce parsing failures.
  - Team Leader prompt specifies an explicit schema for `execution_plan` to ensure consistency.
  - QA Review prompt to critically assess the synthesis and suggest improvements.

- `tools.py` (Domain Tools)
  - Data Engineering: `join_datasets`, `join_datasets_on`, `clean_data`.
  - EDA: `descriptive_stats`, `detect_outliers`, `correlation_matrix`, `get_exploratory_analysis`.
  - Visualization: `plot_histogram`, `plot_boxplot`, `plot_scatter`, `generate_chart`.
  - Data Utilities: `read_odt_tables` (extract tables from ODT), `normalize_dataframe_columns` (snake_case ASCII columns).
  - In-memory chart storage (BytesIO) to avoid file path issues and enable direct rendering/download in UI.

## Data Flow and Execution Pipeline

1. File Upload and Preprocessing
   - Supported formats: CSV, XLSX, XLS (xlrd), ODS (odf), ODT tables (odfpy via temporary extraction).
   - Optional column normalization to snake_case ASCII to stabilize joins and plotting.
   - The default DataFrame is selected for automatic tool parameter filling.

2. Relationship and Join Configuration
   - Sidebar controls allow picking left/right datasets, join keys (same or different), and join type (inner/left/right/outer).
   - A quick preview test executes the chosen join to validate the configuration.
   - The main preview shows headers, sample rows, dtypes, and candidate key columns based on uniqueness.

3. Orchestration (LLM Agents)
   - Orchestrator transforms the chat query into a structured brief (strict JSON) and it is validated by Pydantic. Validation errors are fed back to the LLM for auto-correction (up to 3 attempts).
   - Team Leader transforms the brief into an `execution_plan` (strict JSON with schema) and it is validated by Pydantic, with the same auto-correction loop.
   - Plan normalization in `app.py` handles variations (e.g., `tarefas`, `plano_de_execucao`, or a generic list under another key like `projeto`). If all else fails, a minimal fallback plan is created.

4. Task Execution and Tools Mapping
   - Each task specifies `tool_to_use`. A mapping from tool name to function in `tools.py` is used to invoke the function.
   - `_fill_default_inputs_for_task()` fills missing parameters based on the default DataFrame and numeric columns, or based on join spec for join tools.
   - On failure, traceback is captured and a corrected plan is requested. A selective retry is attempted once; if successful, dependents already completed are invalidated and re-queued (cascading re-execution).
   - Results are stored back into `shared_context` for subsequent tasks or synthesis.

5. Synthesis, QA, and Final Response
   - To avoid token overrun (TPM) with Groq models, a compact shared context is built:
     - DataFrames summarized by shape, first columns/dtypes, and a small CSV sample.
     - Long strings truncated.
   - The Team Leader synthesizes results, then a QA Review step produces critical suggestions, which are appended to the memory context.
   - The Business Analyst renders the final user-facing narrative, incorporating QA insights.
   - The full assistant response is streamed to a single message and stored untruncated in session memory.

6. Charts Rendering and Download
   - Plots are captured as PNG bytes in-memory (BytesIO) and displayed immediately.
   - Each chart provides a download button for exporting the image.
   - A maintenance action can remove legacy `plot_*.png` files.

7. PDF Export (ABNT-like + Minto Pyramid)
   - Generated using ReportLab.
   - Includes: cover page, executive summary (Minto: Situation, Complication, Question, Answer), development (methods, plan summary, tasks), results with figures (limited), conclusions/recommendations, and references note.
   - Uses ABNT-like margins and typography approximation (Times 12, 1.5 line spacing) for better readability.

## Error Handling and Robustness

- JSON Parsing Resilience
  - `clean_json_response()` extracts the first balanced JSON object even if the LLM returns text wrappers or markdown.
  - Prompts enforce JSON-only responses to minimize failures.

- Plan Structure Normalization
  - Specific mappers for `tarefas` and `plano_de_execucao` keys.
  - Generic fallback mapping for any list of task-like dicts under other keys (e.g., `projeto`).
  - Minimal default plan if everything else fails, preventing user-facing crashes.

- Token Limits (Groq TPM 413)
  - Shared context compaction (summarize DataFrames, truncate long strings) drastically reduces token usage for synthesis.
  - Optional strategies (not enabled by default) could further reduce payload: ultra-lean summaries, column filtering.

- Media File Storage Errors
  - Optional/lazy imports: PDF generation (ReportLab) loaded on demand; app remains functional without it. Sentiment analysis (TextBlob) is lazy-imported and returns a friendly message if missing.

- Validation and Auto-Correction with Pydantic
  - Briefing and Execution Plan are validated against Pydantic models.
  - Validation errors are summarized and fed back to the LLM with a strict instruction to correct and return JSON-only.

- Selective Retry and Cascading Re-execution
  - On tool errors, traceback is captured and the Team Leader is asked for a corrected plan.
  - The failing task is retried once with suggested corrections; successful retries invalidate already-completed dependents to ensure consistency.

- Analytics and Logging
  - Structured execution log stored in memory with optional JSON persistence.
  - Analytics panel summarizes success/error rates by tool, mean duration, and frequent error inputs.
  - All plots are stored as in-memory bytes; the UI checks existence when encountering legacy paths and warns instead of failing.

## Rationale Behind Key Decisions

- Streamlit UI for Rapid Iteration
  - Quick development, reactive sidebar controls, and smooth chat display with streaming.

- Strict JSON Prompts
  - Greatly reduce runtime parsing issues and branchy error handling.

- Plan Normalization and Fallbacks
  - Accepts real-world variance in LLM outputs across providers/models while keeping the execution stable.

- In-memory Charts
  - Avoids temp-path fragility, simplifies download, and keeps UX consistent across environments.

- ABNT + Minto PDF
  - Lazy import to avoid hard dependency when PDF is not used.
  - Combines a familiar academic/enterprise formatting style with a communication structure tailored for clarity and decision-making.

## Future Improvements

- Pagination and table-of-contents in PDF, with page numbering and footers.
- Configurable synthesis compression (ultra-lean mode) in the sidebar.
- More robust join diagnostics (duplicates, unmatched counts, column collision report).
- Persist plan cache to disk and introduce a ranking for best plans by success rate and latency.

