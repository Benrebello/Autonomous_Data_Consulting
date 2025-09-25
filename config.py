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
    Load configuration from config.json.
    """
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    config.setdefault('provider', 'groq')
    config.setdefault('model', 'llama-3.1-8b-instant')
    config.setdefault('api_key', '')
    config.setdefault('rpm_limit', 10)
    return config

def obtain_llm(config: dict):
    """
    Initialize LLM based on config.
    """
    provider = config['provider']
    model = config['model']
    api_key = config.get('api_key') or os.getenv('API_KEY')
    if provider == 'openai':
        return OpenAI(model=model, api_key=api_key)
    elif provider == 'groq':
        return ChatGroq(model=model, api_key=api_key)
    elif provider == 'google':
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    else:
        raise ValueError("Unsupported provider")
