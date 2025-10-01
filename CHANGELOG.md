# Changelog

## [Unreleased]
### Added
- Column selection expander for chart tasks.
- Pydantic validation + auto-correction cycles for Briefing and Execution Plan.
- Task execution selective retry + cascading re-execution of dependents.
- QA Review step between synthesis and final response.
- Success plan cache (optional reuse).
- Execution Analytics expander and optional JSON log persistence.

### Changed
- PDF generation is lazily imported (ReportLab) to avoid hard dependency at import.
- TextBlob is lazy-imported within the sentiment tool.

### Fixed
- Improved robustness of plan normalization and error handling during execution.
