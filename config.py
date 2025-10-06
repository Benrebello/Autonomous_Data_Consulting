"""
Configuration module for LLM setup.
"""

import json
import os
from langchain_openai import OpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

def load_config() -> dict:
    """
    Load configuration from config.json and allow environment variables to override
    (PROVIDER, MODEL, API_KEY, RPM_LIMIT). This is friendly to Streamlit Cloud
    where secrets are usually provided via env vars.
    """
    # Base from file (optional)
    try:
        with open('config.json', 'r') as f:
            file_cfg = json.load(f)
    except FileNotFoundError:
        file_cfg = {}

    # Defaults
    cfg = {
        'provider': 'groq',
        'model': 'llama-3.1-8b-instant',
        'api_key': '',
        'rpm_limit': 10,
    }

    # Merge file config
    cfg.update({k: v for k, v in file_cfg.items() if v is not None})

    # Env overrides
    provider_env = os.getenv('PROVIDER')
    model_env = os.getenv('MODEL')
    api_key_env = os.getenv('API_KEY')
    rpm_env = os.getenv('RPM_LIMIT')

    if provider_env:
        cfg['provider'] = provider_env.strip().lower()
    if model_env:
        cfg['model'] = model_env.strip()
    if api_key_env:
        cfg['api_key'] = api_key_env.strip()
    if rpm_env:
        try:
            cfg['rpm_limit'] = int(rpm_env)
        except ValueError:
            pass

    return cfg

def obtain_llm(config: dict):
    """
    Initialize LLM based on config.
    """
    provider = (config.get('provider') or 'groq').lower()
    model = config.get('model') or 'llama-3.1-8b-instant'
    # Prefer explicit config, fallback to env var
    api_key = config.get('api_key') or os.getenv('API_KEY') or ''
    if provider == 'openai':
        return OpenAI(model=model, api_key=api_key)
    elif provider == 'groq':
        return ChatGroq(model=model, api_key=api_key)
    elif provider == 'google':
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    else:
        raise ValueError("Unsupported provider")
