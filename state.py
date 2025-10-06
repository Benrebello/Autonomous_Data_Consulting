# state.py
"""Centralized application state management with Pydantic validation."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import streamlit as st


class AppState(BaseModel):
    """Centralized application state with type validation.
    
    This replaces the scattered st.session_state variables with a single,
    validated, and type-safe state object.
    """
    
    # Discovery flow
    discovery_active: bool = False
    discovery_step: int = 0
    discovery_questions: List[str] = Field(default_factory=list)
    discovery_answers: List[str] = Field(default_factory=list)
    discovery_summary: Optional[str] = None
    discovery_confirm_pending: bool = False
    discovery_edit_mode: bool = False
    discovery_edit_index: Optional[int] = None
    first_user_query: Optional[str] = None
    
    # Analysis context
    clarify_pending: bool = False
    intent_mode: Optional[str] = None
    last_tool: Optional[str] = None
    last_objective: Optional[str] = None
    last_deliverables: Optional[List[str]] = None
    last_result: Optional[Any] = None
    last_topic: Optional[str] = None
    last_narrative: Optional[str] = None
    last_summary_signature: Optional[str] = None
    
    # UI preferences
    response_style: str = 'Padrão'  # Concisa | Padrão | Executiva
    
    # Memory and history
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    conversation: List[str] = Field(default_factory=list)
    chat_history: List[Dict[str, Any]] = Field(default_factory=list)
    reports: List[Dict[str, Any]] = Field(default_factory=list)
    analyses_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Charts
    charts: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Default DataFrame selection
    default_df_key: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def get(cls) -> 'AppState':
        """Load state from Streamlit session or create new instance.
        
        Returns:
            AppState instance from session or newly created
        """
        if 'app_state' not in st.session_state:
            st.session_state['app_state'] = cls()
        return st.session_state['app_state']
    
    def save(self):
        """Persist state back to Streamlit session."""
        st.session_state['app_state'] = self
    
    def reset_discovery(self):
        """Reset discovery flow state."""
        self.discovery_active = False
        self.discovery_step = 0
        self.discovery_questions = []
        self.discovery_answers = []
        self.discovery_summary = None
        self.discovery_confirm_pending = False
        self.discovery_edit_mode = False
        self.discovery_edit_index = None
        self.first_user_query = None
    
    def add_to_memory(self, item: Dict[str, Any]):
        """Add item to memory."""
        self.memory.append(item)
    
    def add_to_chat_history(self, role: str, content: str):
        """Add message to chat history."""
        self.chat_history.append({
            'role': role,
            'content': content
        })
    
    def add_chart(self, chart_bytes: bytes, caption: str):
        """Add chart to charts list."""
        self.charts.append({
            'bytes': chart_bytes,
            'caption': caption
        })
    
    def clear_charts(self):
        """Clear all charts."""
        self.charts = []


# Backward compatibility helper functions
def migrate_session_state_to_app_state():
    """Migrate old session_state variables to new AppState.
    
    This function helps transition from the old scattered state management
    to the new centralized approach.
    """
    state = AppState.get()
    
    # Map old keys to new state attributes
    mappings = {
        'discovery_active': 'discovery_active',
        'discovery_step': 'discovery_step',
        'discovery_questions': 'discovery_questions',
        'discovery_answers': 'discovery_answers',
        'discovery_summary': 'discovery_summary',
        'clarify_pending': 'clarify_pending',
        'intent_mode': 'intent_mode',
        'last_tool': 'last_tool',
        'last_objective': 'last_objective',
        'last_deliverables': 'last_deliverables',
        'last_result': 'last_result',
        'last_topic': 'last_topic',
        'last_narrative': 'last_narrative',
        'last_summary_signature': 'last_summary_signature',
        'response_style': 'response_style',
        'memory': 'memory',
        'conversation': 'conversation',
        'chat_history': 'chat_history',
        'reports': 'reports',
        'analyses_history': 'analyses_history',
        'charts': 'charts',
    }
    
    for old_key, new_attr in mappings.items():
        if old_key in st.session_state and old_key != 'app_state':
            setattr(state, new_attr, st.session_state[old_key])
    
    state.save()
    return state
