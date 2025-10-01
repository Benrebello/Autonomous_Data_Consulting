# Contributing Guide

Thank you for your interest in contributing!

## Local Setup
- Use Python 3.10+.
- Create a virtual environment:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Run the app:
  ```bash
  streamlit run app.py
  ```
- Run tests:
  ```bash
  pytest -q
  ```

## Coding Standards
- Follow existing architecture and style conventions.
- Keep public functions and classes documented with docstrings.
- Prefer reusing existing tools/services over adding new ones.
- Use Conventional Commits for messages:
  - feat, fix, refactor, docs, chore, test, perf, ci, build, style, revert
  - Example: `feat(ui): add column selection expander for charts`

## PR Process
- Fork and create a feature branch: `feat/<short-name>`
- Keep PRs focused and under ~300 lines when possible.
- Include tests when you change behavior or add features.
- Update docs/README when you change UX or flows.

## Tests
- Run `pytest -q` locally before submitting.
- Prefer deterministic tests with small data.

## Security & Secrets
- Never commit API keys or secrets.
- Use `st.secrets` or environment variables.

## Discussions
- Open an issue for large refactors or design changes before implementing.
