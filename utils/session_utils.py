"""
Utilitaires pour la gestion de session Streamlit.
Initialisation et manipulation de st.session_state.
"""

import streamlit as st
from typing import Any, Dict, Optional
from datetime import datetime

def init_session_state(defaults: Dict[str, Any]):
    """
    Initialise les variables de session avec des valeurs par défaut.
    
    Args:
        defaults: Dictionnaire de valeurs par défaut
    """
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_session_value(key: str, default: Any = None) -> Any:
    """
    Récupère une valeur de session de manière sécurisée.
    
    Args:
        key: Clé à récupérer
        default: Valeur par défaut si la clé n'existe pas
        
    Returns:
        La valeur ou default
    """
    return st.session_state.get(key, default)

def set_session_value(key: str, value: Any):
    """
    Définit une valeur de session.
    
    Args:
        key: Clé à définir
        value: Valeur à stocker
    """
    st.session_state[key] = value

def delete_session_value(key: str) -> bool:
    """
    Supprime une valeur de session.
    
    Args:
        key: Clé à supprimer
        
    Returns:
        True si la clé existait et a été supprimée
    """
    if key in st.session_state:
        del st.session_state[key]
        return True
    return False

def clear_session():
    """Efface toutes les variables de session."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def session_has_key(key: str) -> bool:
    """
    Vérifie si une clé existe dans la session.
    
    Args:
        key: Clé à vérifier
        
    Returns:
        True si la clé existe
    """
    return key in st.session_state

def increment_counter(key: str, increment: int = 1) -> int:
    """
    Incrémente un compteur dans la session.
    
    Args:
        key: Clé du compteur
        increment: Valeur d'incrémentation
        
    Returns:
        Nouvelle valeur du compteur
    """
    current = st.session_state.get(key, 0)
    new_value = current + increment
    st.session_state[key] = new_value
    return new_value

def toggle_boolean(key: str) -> bool:
    """
    Inverse une valeur booléenne dans la session.
    
    Args:
        key: Clé à inverser
        
    Returns:
        Nouvelle valeur
    """
    current = st.session_state.get(key, False)
    new_value = not current
    st.session_state[key] = new_value
    return new_value

def log_session_event(event_name: str, data: Optional[Dict[str, Any]] = None):
    """
    Enregistre un événement dans l'historique de session.
    
    Args:
        event_name: Nom de l'événement
        data: Données associées
    """
    if 'session_logs' not in st.session_state:
        st.session_state.session_logs = []
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'event': event_name,
        'data': data or {}
    }
    
    st.session_state.session_logs.append(log_entry)

def get_session_logs() -> list:
    """
    Récupère tous les logs de session.
    
    Returns:
        Liste des logs
    """
    return st.session_state.get('session_logs', [])

def initialize_chat_session():
    """Initialise spécifiquement les variables pour le chat."""
    defaults = {
        'messages': [],
        'message_count': 0,
        'documents_loaded': False,
        'enable_tts': False,
        'stream_response': True,
        'show_latex': True,
        'default_level': 'Seconde',
        'nb_docs': 5,
    }
    init_session_state(defaults)