"""
Interface de chat - VERSION CORRIGÉE
Les fichiers uploadés ne sont PAS vectorisés, ils sont analysés directement
"""

import streamlit as st
from typing import Optional
from services.rag_service import rag_service
from services.voice_service import voice_service
from services.math_solver import math_solver
from ui.components import display_message, ChatHistory
from utils.pdf_utils import markdown_to_pdf
from core.llm_manager import llm_manager
import re


# ============================================================
# NETTOYAGE DES RÉPONSES
# ============================================================

def clean_rag_response(response: str) -> str:
    """Supprime les références aux documents dans les réponses"""
    if not response:
        return response
    
    response = re.sub(r'Dans un document \[.*?\]\s*\(Pertinence\s*:\s*\d+%\)', '', response, flags=re.IGNORECASE)
    response = re.sub(r'\[[\w\-_\.]+\.pdf\]', '', response)
    response = re.sub(r'\(Pertinence\s*:\s*\d+%\)', '', response)
    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
    response = re.sub(r'  +', ' ', response)
    
    return response.strip()


# ============================================================
# CSS
# ============================================================

SEARCH_BAR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');

.stChatInput { display: none !important; }

button[kind="primaryFormSubmit"],
button[kind="secondaryFormSubmit"] {
    display: none !important;
    visibility: hidden !important;
}

.file-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #FFFFFF;
    color: #FF6B35 !important;
    border: 2px solid #FF6B35;
    border-radius: 20px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px;
    box-shadow: 0 2px 8px rgba(255, 107, 53, 0.15);
    transition: all 0.3s ease;
}

.file-tag:hover {
    background: #FF6B35;
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 107, 53, 0.25);
}
</style>
"""


# ============================================================
# GESTION DU CONTEXTE
# ============================================================

def get_conversation_context(max_messages: int = 40) -> str:
    """Récupère le contexte des derniers messages"""
    if 'messages' not in st.session_state or not st.session_state.messages:
        return ""
    
    recent_messages = st.session_state.messages[-max_messages:]
    context_parts = []
    
    for msg in recent_messages:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:4000] + "..." if len(msg["content"]) > 4000 else msg["content"]
        context_parts.append(f"{role}: {content}")
    
    return "\n".join(context_parts)


def build_query_with_context(question: str) -> str:
    """Construit une requête incluant le contexte"""
    context = get_conversation_context(max_messages=40)
    
    if context:
        return f"""Contexte de la conversation précédente :
{context}

Question actuelle : {question}

Réponds à la question actuelle en tenant compte du contexte."""
    else:
        return question


# ============================================================
# GESTION DES DOCUMENTS UPLOADÉS (SANS VECTORISATION)
# ============================================================

def get_uploaded_documents_context() -> str:
    """
    Récupère le contenu des documents uploadés comme contexte
    SANS les vectoriser
    """
    if 'uploaded_documents_content' not in st.session_state:
        return ""
    
    if not st.session_state.uploaded_documents_content:
        return ""
    
    context_parts = []
    for doc in st.session_state.uploaded_documents_content:
        filename = doc.get('filename', 'Document')
        text = doc.get('text', '')
        
        # Limiter la taille du contexte
        text_truncated = text[:5000] if len(text) > 5000 else text
        
        context_parts.append(f"""
📄 Document uploadé : {filename}

Contenu :
{text_truncated}
""")
    
    return "\n\n".join(context_parts)


# ============================================================
# FONCTIONS PRINCIPALES
# ============================================================

def render_chat_interface():
    """Rend l'interface de chat complète"""

    ChatHistory.initialize()
    
    if 'uploaded_documents_content' not in st.session_state:
        st.session_state.uploaded_documents_content = []
    
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = []
    
    if 'show_uploader' not in st.session_state:
        st.session_state.show_uploader = False
    
    if 'show_voice' not in st.session_state:
        st.session_state.show_voice = False
    
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = ''

    st.markdown(SEARCH_BAR_CSS, unsafe_allow_html=True)

    # En-tête
    st.markdown("""
    <div style='text-align:center; padding: 24px 0 16px 0;'>
        <div style='font-size: 56px; margin-bottom: 12px;'>🧮</div>
        <h1 style='
            font-size: 48px;
            font-weight: 900;
            color: #2D2D2D;
            margin: 0 0 12px 0;
            letter-spacing: -1.5px;
            font-family: Google Sans, sans-serif;
        '>
            Intelli<span style='color: #FF6B35;'>Math</span>
        </h1>
        <p style='
            color: #6C757D;
            font-size: 16px;
            margin: 0 0 8px 0;
            font-weight: 500;
        '>
            Pose tes questions · Uploade tes documents 📎
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Boutons
    if st.session_state.get('messages'):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("💾 Sauvegarder", use_container_width=True, type="secondary"):
                from ui.sidebar import save_current_conversation
                save_current_conversation()
                st.success("✅ Conversation sauvegardée !")
                st.balloons()
        
        with col2:
            if st.button("🗑️ Nouvelle conversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.message_count = 0
                st.session_state.uploaded_documents_content = []
                st.session_state.uploaded_files_names = []
                st.rerun()

    # Afficher l'historique
    if st.session_state.messages:
        st.markdown("---")
        for msg in st.session_state.messages:
            display_message(msg["role"], msg["content"])
        st.markdown("---")
    else:
        render_welcome_screen()

    st.divider()

    # Barre de saisie
    render_search_bar()


def render_welcome_screen():
    """Écran d'accueil"""
    st.markdown("""
    <div style='text-align:center; padding: 32px 0 24px 0;'>
        <p style='
            color: #6C757D;
            font-size: 18px;
            font-weight: 500;
        '>
            Comment puis-je t'aider aujourd'hui ?
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**💡 Essaie par exemple :**")

    col1, col2 = st.columns(2)

    suggestions = [
        ("📐", "Théorème de Pythagore",  "Explique-moi le théorème de Pythagore"),
        ("📈", "Calculer une dérivée",    "Comment dériver f(x) = x³ + 2x² - 5 ?"),
        ("🔢", "Suite arithmétique",      "C'est quoi une suite arithmétique ?"),
        ("⚖️", "Résoudre une équation",   "Résous : 2x² - 5x + 3 = 0"),
        ("∫",  "Calculer une intégrale",  "Comment calculer l'intégrale de x² ?"),
        ("🎲", "Loi binomiale",           "Explique la loi binomiale"),
    ]

    with col1:
        for icon, label, full_question in suggestions[:3]:
            if st.button(f"{icon} {label}", key=f"sugg_{label}", use_container_width=True):
                st.session_state.pending_question = full_question
                st.rerun()

    with col2:
        for icon, label, full_question in suggestions[3:]:
            if st.button(f"{icon} {label}", key=f"sugg_{label}", use_container_width=True):
                st.session_state.pending_question = full_question
                st.rerun()


def render_search_bar():
    """Barre de saisie style ChatGPT"""

    # =========================
    # DOCUMENTS TAGS
    # =========================
    if st.session_state.get('uploaded_files_names'):
        tags_html = ""
        for fname in st.session_state.uploaded_files_names:
            tags_html += f'<span class="file-tag">📄 {fname}</span>'
        st.markdown(tags_html, unsafe_allow_html=True)

    # =========================
    # CONTEXTE INFO
    # =========================
    if st.session_state.get('messages'):
        nb_messages = len(st.session_state.messages)
        st.caption(f"💬 {nb_messages} message(s) • Mémoire active")

    # =========================
    # FORM CHAT
    # =========================
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

    with st.form("chat_form", clear_on_submit=True):

        col1, col2, col3, col4 = st.columns([1, 10, 1, 1])

        # 📎 Upload
        with col1:
            upload_clicked = st.form_submit_button("📎", help="Ajouter")

        # Zone texte
        with col2:
            user_input = st.text_area(
                "",
                placeholder="Poser une question",
                height=40,
                key="user_input_area",
                label_visibility="collapsed"
            )

        # 🎤 Micro
        with col3:
            voice_clicked = st.form_submit_button("🎤")

        # ➤ Envoyer
        with col4:
            send_clicked = st.form_submit_button("➤", type="primary")

        # =========================
        # ACTIONS
        # =========================
        if upload_clicked:
            st.session_state.show_uploader = True

        if voice_clicked:
            st.session_state.show_voice = True

        if send_clicked and user_input and user_input.strip():
            process_question(user_input.strip())

    # =========================
    # PENDING QUESTION
    # =========================
    if st.session_state.get('pending_question'):
        question = st.session_state.pending_question
        st.session_state.pending_question = ''
        process_question(question)

    # =========================
    # PANELS
    # =========================
    if st.session_state.get('show_uploader'):
        render_inline_uploader()

    if st.session_state.get('show_voice'):
        voice_question = render_voice_panel()
        if voice_question:
            st.session_state.show_voice = False
            process_question(voice_question)


def render_inline_uploader():
    """Zone d'upload - SANS vectorisation"""

    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("### 📤 Uploader des documents pour analyse")
        st.info("💡 Ces documents seront **analysés directement** par le LLM, ils ne seront **PAS ajoutés** à la base de connaissances.")
    
    with col2:
        if st.button("✖ Fermer", key="close_upload", use_container_width=True):
            st.session_state.show_uploader = False
            st.rerun()

    uploaded_files = st.file_uploader(
        "Glissez vos fichiers ici",
        type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key="inline_file_uploader",
        help="PDF, Word, TXT, Images — Analyse directe sans indexation"
    )

    if uploaded_files:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**{len(uploaded_files)} fichier(s) sélectionné(s) :**")
            for f in uploaded_files:
                size_mb = f.size / (1024 * 1024)
                st.markdown(f"- 📄 `{f.name}` ({size_mb:.2f} MB)")

        with col2:
            if st.button("📖 Charger", type="primary", use_container_width=True, key="btn_load_files"):
                load_uploaded_files_for_analysis(uploaded_files)

    st.markdown("---")


def render_voice_panel() -> Optional[str]:
    """Panel vocal"""
    st.markdown("---")
    
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown("### 🎤 Enregistrement vocal")
        st.caption("Appuie sur le micro, parle, puis arrête l'enregistrement.")

    with col2:
        if st.button("✖ Fermer", key="close_voice", use_container_width=True):
            st.session_state.show_voice = False
            st.rerun()

    transcribed_text = voice_service.render_audio_input()

    if transcribed_text:
        st.success(f"✅ Reconnu : **{transcribed_text}**")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ Modifier avant d'envoyer", use_container_width=True, key="edit_voice"):
                st.session_state.pending_question = transcribed_text
                st.session_state.show_voice = False
                st.rerun()

        with col2:
            if st.button("➤ Envoyer directement", type="primary", use_container_width=True, key="send_voice"):
                return transcribed_text

    st.markdown("---")
    return None


def load_uploaded_files_for_analysis(uploaded_files):
    """
    Charge les fichiers pour analyse SANS les vectoriser
    """
    
    from services.document_processor import document_processor

    with st.spinner("📖 Chargement des documents pour analyse..."):
        processed_docs = document_processor.process_files(uploaded_files)

        if processed_docs:
            # Stocker en session_state SANS vectoriser
            for doc in processed_docs:
                st.session_state.uploaded_documents_content.append({
                    'filename': doc['metadata'].get('filename', 'Document'),
                    'text': doc['text']
                })
                
                filename = doc['metadata'].get('filename', 'Inconnu')
                if filename not in st.session_state.uploaded_files_names:
                    st.session_state.uploaded_files_names.append(filename)
            
            st.session_state.show_uploader = False
            
            st.success(f"✅ {len(processed_docs)} document(s) chargé(s) pour analyse !")
            st.info("💡 Ces documents sont maintenant disponibles pour l'analyse. Le LLM va chercher les réponses dans la **base de connaissances** pour répondre aux questions contenues dans ces documents.")
            st.rerun()
        else:
            st.error("❌ Aucun document n'a pu être traité.")


def process_question(question: str):
    """Traite la question avec LLM Router"""
    
    # Classification
    with st.spinner("🤔 Analyse de ta question..."):
        category, reasoning = llm_manager.classify_query(question)
    
    # Ajouter à l'historique
    ChatHistory.add_message("user", question)
    
    # Afficher la question
    with st.chat_message("user", avatar="👨‍🎓"):
        st.markdown(question)
    
    # Router
    if category == "CONVERSATION":
        response = llm_manager.get_conversation_response(question)
        
        with st.chat_message("assistant", avatar="👨‍🏫"):
            st.markdown(response)
            ChatHistory.add_message("assistant", response)
        
        st.rerun()
        return
    
    elif category == "OUT_OF_SCOPE":
        response = llm_manager.get_out_of_scope_response()
        
        with st.chat_message("assistant", avatar="👨‍🏫"):
            st.markdown(response)
            ChatHistory.add_message("assistant", response)
        
        st.rerun()
        return
    
    # MATH_RAG - Chercher dans knowledge base
    topic        = math_solver.detect_topic(question)
    level        = st.session_state.get('default_level', 'Lycée')
    is_exercise  = math_solver.is_exercise(question)
    is_def, term = math_solver.is_definition_request(question)

    with st.status("🔍 Recherche dans la base de connaissances...", expanded=False) as status:
        st.write(f"🤖 Classification : **{category}**")
        st.write(f"📚 Sujet : **{topic.value}**")
        st.write(f"🎓 Niveau : **{level}**")
        
        # Afficher le contexte
        context = get_conversation_context(max_messages=40)
        if context:
            st.write("💬 **Contexte conversation activé**")
        
        # Afficher documents uploadés
        uploaded_context = get_uploaded_documents_context()
        if uploaded_context:
            st.write(f"📄 **{len(st.session_state.uploaded_documents_content)} document(s) uploadé(s) analysé(s)**")
        
        if is_exercise:
            st.write("✏️ Type : Exercice")
        elif is_def and term:
            st.write(f"📖 Type : Définition de **{term}**")
        
        status.update(label="✅ Recherche terminée", state="complete")

    # Construire le prompt avec documents uploadés
    query_with_context = build_query_with_context(question)
    
    if uploaded_context:
        query_with_context = f"""{uploaded_context}

---

Question de l'utilisateur : {query_with_context}

**Instructions :**
- Analyse le contenu des documents uploadés ci-dessus
- Cherche les réponses dans ta base de connaissances (knowledge base)
- Si tu n'as PAS d'information sur le sujet dans ta base de connaissances, réponds : "Je ne dispose pas d'information sur ce sujet dans ma base de connaissances."
- Ne te base PAS uniquement sur les documents uploadés pour répondre, utilise ta knowledge base
"""

    # Générer réponse
    with st.chat_message("assistant", avatar="👨‍🏫"):
        try:
            use_streaming = st.session_state.get('stream_response', True)
            nb_docs       = st.session_state.get('nb_docs', 5)
            sources = []

            if is_def and term:
                response, sources = rag_service.get_definition_with_sources(term, level)
                response = clean_rag_response(response)
                st.markdown(response)

            elif is_exercise:
                response, sources = rag_service.solve_exercise_with_sources(query_with_context, level)
                response = clean_rag_response(response)
                st.markdown(response)

            else:
                if use_streaming:
                    placeholder    = st.empty()
                    full_response  = ""
                    for chunk in rag_service.query(query_with_context, top_k=nb_docs, use_streaming=True):
                        full_response += chunk
                        placeholder.markdown(full_response + "▌")
                    full_response = clean_rag_response(full_response)
                    placeholder.markdown(full_response)
                    response = full_response
                    sources = rag_service.get_last_sources()
                else:
                    response, sources = rag_service.query_with_sources(query_with_context, top_k=nb_docs)
                    response = clean_rag_response(response)
                    st.markdown(response)

            # Afficher les sources (de la knowledge base)
            if sources:
                render_sources(sources)

            ChatHistory.add_message("assistant", response)

            # Boutons
            col1, col2, col3 = st.columns([1, 1, 8])

            with col1:
                voice_service.render_tts_button(response, key=f"tts_{len(st.session_state.messages)}")

            with col2:
                pdf_data = markdown_to_pdf(response, "Réponse IntelliMath")
                st.download_button(
                    label="💾",
                    data=pdf_data,
                    file_name="reponse_intellimath.pdf",
                    mime="application/pdf",
                    key=f"dl_{len(st.session_state.messages)}",
                    help="Télécharger en PDF"
                )

            from ui.sidebar import auto_save_conversation
            auto_save_conversation()

        except Exception as e:
            error_msg = f"❌ Une erreur s'est produite : {str(e)}"
            st.error(error_msg)
            ChatHistory.add_message("assistant", error_msg)

    st.rerun()


def render_sources(sources):
    """Affiche les sources de la knowledge base"""
    
    with st.expander("📚 Sources utilisées (Knowledge Base)", expanded=False):
        st.markdown("Les informations proviennent de la base de connaissances :")
        
        for idx, source in enumerate(sources, 1):
            filename = source.get('filename', 'Document inconnu')
            page = source.get('page', 'N/A')
            level = source.get('level', '')
            excerpt = source.get('excerpt', '')
            score = source.get('score', 0)
            
            st.markdown(f"**{idx}. 📄 {filename}**")
            
            cols = st.columns([1, 1, 1])
            with cols[0]:
                if page != 'N/A':
                    st.caption(f"📖 Page {page}")
            with cols[1]:
                if level:
                    st.caption(f"🎓 {level}")
            with cols[2]:
                st.caption(f"🎯 Pertinence: {score:.0%}")
            
            if excerpt:
                excerpt_short = excerpt[:200] + "..." if len(excerpt) > 200 else excerpt
                st.markdown(f"> {excerpt_short}")
            
            if idx < len(sources):
                st.divider()
