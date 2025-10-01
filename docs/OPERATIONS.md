# Operations Guide

## Environments
- Local development: filesystem persistence.
- Cloud/ephemeral (Streamlit Cloud, containers): use external storage for persistence.

## Storage Strategy
- Execution logs (optional): `./logs/execution_log_<timestamp>.json`
- Plan cache (session): `st.session_state['plan_cache']`
- For durable storage, use one of:
  - S3 / GCS / Azure Blob
  - Database (Postgres/Supabase) or Redis

## Configuration
- Use `st.secrets` for cloud credentials.
- Sidebar toggles:
  - Reuse success plan
  - Save execution log (JSON)

## Deployment Tips
- Ensure `REPORTLAB` and `TEXTBLOB` are optional: the app runs without them.
- Configure API provider/model/key via sidebar or `config.json`.

## Observability
- Enable JSON logs to analyze execution after runs.
- Use the in-app Analytics expander for quick insights.
