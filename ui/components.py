"""
Composants réutilisables pour l'interface Streamlit.
Widgets personnalisés et fonctions d'affichage.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import time

def display_message(role: str, content: str, avatar: Optional[str] = None):
    """Affiche un message dans le chat avec avatar."""
    avatars = {
        'user': '👨‍🎓',
        'assistant': '👨‍🏫'
    }
    avatar = avatar or avatars.get(role, '💬')
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

def create_topic_selector(key: str = "topic_selector") -> str:
    """
    Crée un sélecteur de sujets mathématiques.
    
    Args:
        key: Clé unique pour le widget Streamlit
        
    Returns:
        Le sujet sélectionné
    """
    topics = {
        "📊 Fonctions": "fonctions",
        "📈 Dérivées": "dérivées",
        "📉 Intégrales": "intégrales",
        "🔢 Suites": "suites",
        "🎲 Probabilités": "probabilités",
        "📐 Géométrie": "géométrie",
        "🌀 Trigonométrie": "trigonométrie",
        "⚖️ Équations": "équations",
        "∞ Limites": "limites",
        "➡️ Vecteurs": "vecteurs",
        "🔺 Nombres complexes": "nombres complexes",
        "📊 Statistiques": "statistiques"
    }

    selected = st.selectbox(
        "Choisir un sujet",
        options=list(topics.keys()),
        key=key
    )

    return topics[selected]

def create_level_selector(key: str = "level_selector") -> str:
    """
    Crée un sélecteur de niveau scolaire.
    
    Args:
        key: Clé unique pour le widget Streamlit
        
    Returns:
        Le niveau sélectionné
    """
    levels = ["Seconde", "Première", "Terminale"]

    return st.selectbox(
        "Niveau",
        options=levels,
        key=key
    )


class ChatHistory:
    """Gestionnaire d'historique de chat avec fonctionnalités avancées"""

    @staticmethod
    def initialize():
        """Initialise l'historique dans session_state"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "message_count" not in st.session_state:
            st.session_state.message_count = 0

    @staticmethod
    def add_message(role: str, content: str):
        """Ajoute un message à l'historique"""
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        st.session_state.message_count += 1

    @staticmethod
    def get_messages() -> List[Dict[str, Any]]:
        """Récupère tous les messages"""
        return st.session_state.messages

    @staticmethod
    def clear_history():
        """Efface l'historique"""
        st.session_state.messages = []
        st.session_state.message_count = 0

    @staticmethod
    def display_history():
        """Affiche tout l'historique"""
        for msg in st.session_state.messages:
            display_message(msg["role"], msg["content"])

    @staticmethod
    def export_history() -> str:
        """
        Exporte l'historique en format texte.
        
        Returns:
            Historique formaté en texte
        """
        export_text = "# Historique de conversation\n\n"
        for i, msg in enumerate(st.session_state.messages, 1):
            role_name = "Élève" if msg["role"] == "user" else "Professeur"
            export_text += f"## Message {i} - {role_name}\n\n"
            export_text += f"{msg['content']}\n\n"
            export_text += "---\n\n"
        return export_text


