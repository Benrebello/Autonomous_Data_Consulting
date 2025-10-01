# Execution Analytics

## Execution Log Schema
Each task appends an item with fields like:
- `task_id`: integer
- `description`: string
- `tool`: string
- `inputs_keys`: list of strings
- `status`: `success` | `error` | `retrying` | `cascade_invalidation`
- `duration_s`: float (optional)
- `error`: string (optional)
- `traceback`: string (optional)

## Built-in Analytics (UI)
- Success/error counts by tool
- Mean duration by tool (success only)
- Most frequent input keys among errors

## Exporting Logs
- If enabled in the sidebar, logs are written to `./logs/execution_log_<timestamp>.json`.
- Example quickload in a notebook:
  ```python
  import json, pandas as pd
  with open('logs/execution_log_YYYYMMDD_HHMMSS.json') as f:
      data = json.load(f)
  df = pd.DataFrame(data)
  ```
