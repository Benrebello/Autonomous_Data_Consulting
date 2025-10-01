# Security Guidelines

## Secrets and Credentials
- Never commit API keys to the repository.
- Use `st.secrets` or environment variables for provider credentials.

## File Uploads
- Supported formats: CSV, XLSX, XLS, ODS, ODT (tables only).
- Consider size limits and sanitize column names (normalization option available).

## Logging and Privacy
- Execution logs are optional; avoid logging sensitive data.
- If exporting logs, store them securely (local disk with restricted access or external storage with IAM).

## Dependencies
- Some optional dependencies (ReportLab/TextBlob) are loaded lazily; ensure your environment pins versions via `requirements.txt`.

## Hosting
- In cloud environments, use managed secrets, encrypted storage, and HTTPS endpoints.
