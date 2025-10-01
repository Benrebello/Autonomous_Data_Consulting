# Troubleshooting

## Invalid or malformed JSON
- Briefing/Plan are validated with Pydantic; the app triggers an auto-correction loop.
- If it still fails, a minimal fallback plan is generated.

## Task execution failures
- Traceback is captured; Team Leader is asked to correct the plan.
- A selective retry is attempted. If outputs change, dependents may be re-executed.

## Rate limits / RPM windows
- The app adapts batch size and shows sidebar warnings.

## File reading issues (XLS/ODS/ODT)
- Ensure `openpyxl`, `xlrd`, `odfpy` are installed.

## Missing optional dependencies
- PDF export (ReportLab) and TextBlob sentiment are optional and loaded lazily.
