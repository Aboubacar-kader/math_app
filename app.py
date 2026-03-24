"""
IntelliMath - Application principale COMPLÈTE
Historique des conversations + Analyse de fichiers + Chargement simultané
"""

import os
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st
from datetime import datetime
from typing import List, Dict
import json
import uuid

# Configuration de la page
st.set_page_config(
    page_title="IntelliMath",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports des composants
from ui.sidebar import render_footer, render_sidebar
from ui.tabs import render_tabs
from services.rag_service import rag_service
from services.document_processor import document_processor
from core.vectorstore_manager import vectorstore_manager


# ============================================================
# INITIALISATION BASE DE CONNAISSANCES (une seule fois)
# ============================================================

@st.cache_resource(show_spinner=False)
def _run_knowledge_base_init():
    """
    Lance l'indexation en arrière-plan sans bloquer l'interface.
    - Une seule fois par session serveur (cache_resource)
    - Thread daemon : ne bloque pas l'UI, s'arrête avec l'appli
    - Seuls les fichiers nouveaux ou modifiés sont traités
    """
    import threading
    from core.knowledge_base_init import init_knowledge_base

    thread = threading.Thread(
        target=init_knowledge_base,
        daemon=True,
        name="kb-indexer"
    )
    thread.start()


# ============================================================
# GESTION DE L'HISTORIQUE
# ============================================================

def init_session_state():
    """Initialise toutes les variables de session"""
    if 'conversations' not in st.session_state:
        st.session_state.conversations = []
    
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'indexed_files' not in st.session_state:
        st.session_state.indexed_files = []
    
    if 'conversation_counter' not in st.session_state:
        st.session_state.conversation_counter = 0  # conservé pour compatibilité affichage
    
    if 'show_uploader' not in st.session_state:
        st.session_state.show_uploader = False

    # Paramètres persistants (ne jamais écraser si déjà définis)
    if 'setting_voice_mode' not in st.session_state:
        st.session_state.setting_voice_mode = False


def create_new_conversation(first_message: str = None):
    """Crée une nouvelle conversation"""
    st.session_state.conversation_counter += 1

    conversation = {
        'id': uuid.uuid4().hex,
        'title': first_message[:50] + "..." if first_message and len(first_message) > 50 else first_message or f"Conversation {st.session_state.conversation_counter}",
        'created_at': datetime.now().isoformat(),
        'messages': []
    }
    
    st.session_state.conversations.append(conversation)
    st.session_state.current_conversation = conversation['id']
    st.session_state.messages = []

    # Effacer le contexte de conversation des onglets (chat_history_*, figure_detection_*, etc.)
    keys_to_remove = [k for k in st.session_state.keys()
                      if k.startswith('chat_history_')
                      or k.startswith('figure_detection_')
                      or k.startswith('current_conv_id_')]
    for k in keys_to_remove:
        del st.session_state[k]
    st.session_state.uploaded_files_names = []
    st.session_state.uploaded_documents_content = []

    return conversation


def add_message_to_history(role: str, content: str, sources: List[Dict] = None, files: List[str] = None):
    """Ajoute un message à l'historique de la conversation courante"""
    message = {
        'role': role,
        'content': content,
        'timestamp': datetime.now().isoformat(),
        'sources': sources,
        'files': files
    }
    
    st.session_state.messages.append(message)
    
    # Mettre à jour la conversation dans la liste
    if st.session_state.current_conversation:
        for conv in st.session_state.conversations:
            if conv['id'] == st.session_state.current_conversation:
                conv['messages'] = st.session_state.messages
                break


def load_conversation(conversation_id: int):
    """Charge une conversation existante"""
    for conv in st.session_state.conversations:
        if conv['id'] == conversation_id:
            st.session_state.current_conversation = conversation_id
            st.session_state.messages = conv['messages'].copy()
            st.rerun()
            break


def delete_conversation(conversation_id: int):
    """Supprime une conversation"""
    st.session_state.conversations = [
        conv for conv in st.session_state.conversations 
        if conv['id'] != conversation_id
    ]
    
    if st.session_state.current_conversation == conversation_id:
        st.session_state.current_conversation = None
        st.session_state.messages = []
    
    st.rerun()


# ============================================================
# ANALYSE DE FICHIERS
# ============================================================

def process_uploaded_files(uploaded_files):
    """Traite et indexe les fichiers uploadés"""
    if not uploaded_files:
        return None, None
    
    try:
        with st.spinner("📚 Analyse des fichiers en cours..."):
            # Traiter les fichiers
            processed_docs = document_processor.process_files(uploaded_files)
            
            # Extraire textes et métadonnées
            texts = [doc['text'] for doc in processed_docs]
            metadatas = [doc['metadata'] for doc in processed_docs]
            
            # Indexer dans Qdrant
            vectorstore_manager.add_documents(texts, metadatas)
            
            # Sauvegarder les noms
            file_names = [file.name for file in uploaded_files]
            for name in file_names:
                if name not in st.session_state.indexed_files:
                    st.session_state.indexed_files.append(name)
            
            # Créer un résumé du contenu
            summary = f"J'ai analysé {len(uploaded_files)} fichier(s) : {', '.join(file_names)}. "
            
            # Extraire les concepts principaux
            all_text = " ".join(texts[:3])  # Prendre les 3 premiers pour ne pas surcharger
            concept_prompt = f"Liste brièvement les concepts mathématiques principaux dans ce texte : {all_text[:1000]}"
            
            concepts = rag_service.query(concept_prompt, top_k=3)
            
            return summary, concepts
            
    except Exception as e:
        st.error(f"❌ Erreur lors de l'analyse : {str(e)}")
        return None, None


# ============================================================
# INTERFACE PRINCIPALE
# ============================================================

def render_conversation_history():
    """Affiche l'historique des conversations dans la sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💬 Historique des conversations")
    
    if not st.session_state.conversations:
        st.sidebar.info("Aucune conversation pour le moment.")
        return
    
    # Bouton pour nouvelle conversation
    if st.sidebar.button("➕ Nouvelle conversation", use_container_width="Stretch", type="primary"):
        create_new_conversation()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Liste des conversations (plus récentes en premier)
    for conv in reversed(st.session_state.conversations):
        is_current = conv['id'] == st.session_state.current_conversation
        
        col1, col2 = st.sidebar.columns([4, 1])
        
        with col1:
            button_type = "primary" if is_current else "secondary"
            if st.button(
                f"{'📌 ' if is_current else '💬 '}{conv['title'][:30]}...",
                key=f"conv_{conv['id']}",
                use_container_width="Stretch",
                type=button_type
            ):
                load_conversation(conv['id'])
        
        with col2:
            if st.button("🗑️", key=f"del_{conv['id']}", help="Supprimer"):
                delete_conversation(conv['id'])


def render_chat_messages():
    """Affiche les messages de la conversation courante"""
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
            # Afficher les fichiers associés
            if message.get('files'):
                st.caption(f"📎 Fichiers : {', '.join(message['files'])}")
            
            # Afficher les sources
            if message.get('sources'):
                with st.expander("📚 Sources", expanded=False):
                    for idx, source in enumerate(message['sources'], 1):
                        st.markdown(f"**{idx}. {source.get('filename', 'Document')}**")
                        if source.get('page'):
                            st.caption(f"📖 Page {source['page']}")


def first_page_render():
    """Hero header — IntelliMath design moderne"""

    st.markdown("""
    <div style="
        text-align: center;
        padding: 28px 24px 20px;
        margin-bottom: 4px;
    ">
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        ">
            <span style="
                font-size: 36px;
                line-height: 1;
            ">🧮</span>
            <h1 style="
                margin: 0;
                font-family: 'Inter', sans-serif;
                font-size: 34px;
                font-weight: 800;
                letter-spacing: -0.04em;
                color: #0F172A;
            "><span style="
                background: linear-gradient(135deg, #FF8C00 0%, #FF6B2B 55%, #E55A1C 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            ">Intelli</span>Math</h1>
        </div>
        <p style="
            margin: 0;
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            color: #64748B;
            font-weight: 400;
            letter-spacing: 0.01em;
        ">Apprends tes cours &nbsp;·&nbsp; Résous tes exercices &nbsp;·&nbsp; Seconde → Terminale</p>
    </div>
    """, unsafe_allow_html=True)


def render_chat_input():
    """Barre de saisie style GPT avec boutons 📎 🎤 ➤"""

    st.markdown("""
    <style>
    div[data-testid="stForm"] {
        border-radius: 40px !important;
        border: 1px solid #e5e5e5 !important;
        padding: 8px 15px !important;
        background: white !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
    }

    .stTextArea textarea {
        border: none !important;
        outline: none !important;
        resize: none !important;
        box-shadow: none !important;
        font-size: 15px !important;
        background: transparent !important;
    }

    button[kind="secondaryFormSubmit"] {
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        padding: 0 !important;
        font-size: 18px !important;
        border: none !important;
        background: transparent !important;
    }

    button[kind="secondaryFormSubmit"]:hover {
        background: #f2f2f2 !important;
    }

    button[kind="primaryFormSubmit"] {
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        padding: 0 !important;
        font-size: 18px !important;
        background: black !important;
        color: white !important;
        border: none !important;
    }

    button[kind="primaryFormSubmit"]:hover {
        opacity: 0.85 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded_files = None
    user_input = None
    send_clicked = False

    with st.form("chat_form", clear_on_submit=True):

        col1, col2, col3, col4 = st.columns([1, 10, 1, 1])

        with col1:
            upload_clicked = st.form_submit_button("📎", help="Ajouter un fichier")

        with col2:
            user_input = st.text_area(
                "",
                placeholder="Pose une question…",
                height=40,
                label_visibility="collapsed"
            )

        with col3:
            st.form_submit_button("🎤", help="Mode vocal")

        with col4:
            send_clicked = st.form_submit_button("➤", type="primary", help="Envoyer")

        # Activer l'uploader via le bouton 📎
        if upload_clicked:
            st.session_state.show_uploader = True

        # Fermer l'uploader après envoi
        if send_clicked:
            st.session_state.show_uploader = False

        # Afficher l'uploader si actif
        if st.session_state.show_uploader:
            uploaded_files = st.file_uploader(
                "Sélectionne un ou plusieurs fichiers",
                type=['pdf', 'docx', 'txt', 'png', 'jpg'],
                accept_multiple_files=True,
                key="chat_uploader"
            )

    return user_input, uploaded_files, send_clicked


# ============================================================
# FONCTION PRINCIPALE
# ============================================================

def main():
    """Fonction principale - Chargement simultané"""

    # Initialiser l'état de session
    init_session_state()

    # Indexation de la base de connaissances (une seule fois au démarrage)
    _run_knowledge_base_init()
    
    # Charger les styles CSS
    with open('assets/styles.css', 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # SIDEBAR (se charge en même temps que le contenu)
    with st.sidebar:
        st.image('./assets/logo_Kaydan tech.png', width="stretch")
        
        # Navigation
        navigation = st.radio(
            "Navigation",
            options=["💬 Historique", "⚙️ Paramètres"],
            label_visibility="hidden"
        )
        
        if navigation == "💬 Historique":
            render_conversation_history()
        
        else:  # Paramètres
            st.markdown("---")
            st.markdown("### ⚙️ Paramètres")
            st.checkbox("Afficher les sources", value=True, key="show_sources")

            # Mode vocal — valeur persistée dans setting_voice_mode (séparée du widget)
            def _on_voice_mode_change():
                st.session_state['setting_voice_mode'] = st.session_state['_cb_voice_mode']

            st.checkbox(
                "Mode vocal",
                value=st.session_state.get('setting_voice_mode', False),
                key="_cb_voice_mode",
                on_change=_on_voice_mode_change,
                help="Répond par voix + micro activé",
            )
            st.checkbox("Sauvegarder automatiquement", value=True, key="auto_save")
        
        # Footer
        render_footer()
    
    # CONTENU PRINCIPAL
    first_page_render()
    
    # Afficher les onglets
    render_tabs()
    
    # Zone de chat en bas (si conversation active)
    if st.session_state.current_conversation:
        st.markdown("---")
        st.markdown("### 💬 Conversation en cours")
        
        # Afficher les messages
        render_chat_messages()
        
        # ── Barre de saisie style GPT ──
        user_input, uploaded_files, send_clicked = render_chat_input()
        
        # Traitement de la saisie utilisateur
        if send_clicked and (user_input.strip() or uploaded_files):
            
            # Créer une conversation si nécessaire
            if not st.session_state.current_conversation:
                create_new_conversation(user_input or "Analyse de fichier")
            
            # Traiter les fichiers uploadés
            file_names = None
            if uploaded_files:
                summary, concepts = process_uploaded_files(uploaded_files)
                file_names = [f.name for f in uploaded_files]
                
                if summary:
                    add_message_to_history(
                        "assistant",
                        f"{summary}\n\n**Concepts détectés :**\n{concepts}",
                        files=file_names
                    )
                    st.rerun()
            
            # Traiter la question de l'utilisateur
            if user_input.strip():
                add_message_to_history("user", user_input, files=file_names)
                
                with st.spinner("🤔 Réflexion en cours..."):
                    response, sources = rag_service.query_with_sources(
                        user_input,
                        top_k=5
                    )
                
                add_message_to_history("assistant", response, sources=sources)
                st.rerun()


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == "__main__":
    main()