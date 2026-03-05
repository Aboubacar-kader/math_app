"""
Composants réutilisables pour l'interface Streamlit.
Widgets personnalisés et fonctions d'affichage.
"""

import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
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

def display_latex_formula(formula: str, block: bool = True):
    """Affiche une formule LaTeX."""
    if block:
        st.latex(formula)
    else:
        st.markdown(f"${formula}$")

def create_progress_bar(text: str, duration: float = 2.0):
    """Crée une barre de progression avec texte."""
    progress_text = st.empty()
    progress_bar = st.progress(0)
    steps = 100
    for i in range(steps + 1):
        progress_bar.progress(i / steps)
        progress_text.text(f"{text}... {i}%")
        time.sleep(duration / steps)
    progress_bar.empty()
    progress_text.empty()

def create_collapsible_section(title: str, content: str, expanded: bool = False):
    """Crée une section pliable/dépliable."""
    with st.expander(title, expanded=expanded):
        st.markdown(content)

def create_info_box(title: str, content: str, icon: str = "ℹ️"):
    """Crée une boîte d'information stylisée."""
    st.info(f"**{icon} {title}**\n\n{content}")

def create_warning_box(title: str, content: str):
    """Crée une boîte d'avertissement."""
    st.warning(f"**⚠️ {title}**\n\n{content}")

def create_success_box(title: str, content: str):
    """Crée une boîte de succès."""
    st.success(f"**✅ {title}**\n\n{content}")

def create_error_box(title: str, content: str):
    """Crée une boîte d'erreur."""
    st.error(f"**❌ {title}**\n\n{content}")

def display_document_card(filename: str, file_type: str, size: int):
    """
    Affiche une carte pour un document uploadé.
    
    Args:
        filename: Nom du fichier
        file_type: Type MIME du fichier
        size: Taille en bytes
    """
    # Icônes par type de fichier
    if 'pdf' in file_type:
        icon = '📕'
    elif 'word' in file_type:
        icon = '📘'
    elif 'text' in file_type:
        icon = '📄'
    elif 'image' in file_type:
        icon = '🖼️'
    else:
        icon = '📎'

    # Convertir la taille en MB
    size_mb = size / (1024 * 1024)

    st.markdown(f"""
    <div style="
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 5px 0;
        border-left: 4px solid #FF4B4B;
    ">
        {icon} <strong>{filename}</strong><br>
        <small>Taille: {size_mb:.2f} MB</small>
    </div>
    """, unsafe_allow_html=True)

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

def display_statistics(stats: Dict[str, Any]):
    """
    Affiche des statistiques dans des colonnes.
    
    Args:
        stats: Dictionnaire de statistiques
    """
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats.items()):
        with col:
            st.metric(label=label, value=value)

def create_quick_action_buttons() -> Optional[str]:
    """
    Crée des boutons d'action rapide.
    
    Returns:
        L'action sélectionnée ou None
    """
    st.markdown("### 🚀 Actions rapides")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📖 Expliquer un cours", use_container_width=True):
            return "cours"
    with col2:
        if st.button("✏️ Résoudre un exercice", use_container_width=True):
            return "exercice"
    with col3:
        if st.button("📚 Chercher une définition", use_container_width=True):
            return "definition"

    return None

def create_math_keyboard() -> Optional[str]:
    """
    Crée un clavier de symboles mathématiques.
    
    Returns:
        Le symbole cliqué ou None
    """
    symbols = [
        ['∫', '∑', '∏', '√', '∞'],
        ['α', 'β', 'γ', 'θ', 'π'],
        ['≤', '≥', '≠', '≈', '±'],
        ['∈', '∉', '⊂', '∪', '∩']
    ]

    st.markdown("**Symboles mathématiques :**")

    for row in symbols:
        cols = st.columns(len(row))
        for col, symbol in zip(cols, row):
            with col:
                if st.button(symbol, key=f"symbol_{symbol}"):
                    return symbol

    return None


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


# ════════════════════════════════════════════════════════════════════════════
# BARRE DE SAISIE AMÉLIORÉE
# Ajouté le 23/02/2026 - Retour à la ligne automatique + raccourcis clavier
# ════════════════════════════════════════════════════════════════════════════


def render_input_bar(placeholder, section_key, height=120):
    """Barre sans bouton visible"""
    
    # CSS pour cacher le bouton
    st.markdown("""
    <style>
    button[kind="primaryFormSubmit"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.caption("⏎ Ctrl+Enter pour envoyer")
    
    with st.form(key=f"form_{section_key}", clear_on_submit=True):
        user_input = st.text_area(
            "Message",
            placeholder=placeholder,
            height=height,
            key=f"input_{section_key}",
            label_visibility="collapsed"
        )
        
        submitted = st.form_submit_button("Envoyer")
        
        if submitted and user_input and user_input.strip():
            return user_input.strip(), False
    
    return None, False
