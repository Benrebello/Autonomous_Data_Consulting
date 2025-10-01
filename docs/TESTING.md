# Testing Strategy

## Run Tests
```bash
pytest -q
```

## Scope
- Unit tests for tools (`tools.py`).
- Integration tests for mapping tool names to functions.
- Time-series and feature engineering tests where applicable.

## Notes
- Optional dependencies are lazy-imported (ReportLab/TextBlob) to avoid breaking collection.
- Prefer small, deterministic datasets in tests.
