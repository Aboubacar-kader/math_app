"""
Sidebar avec historique des conversations
"""

import streamlit as st
from datetime import datetime


def render_sidebar():
    """Affiche la sidebar complète avec historique"""
    
    st.image("logo_Kaydan tech.png", use_container_width='stretch')
    st.sidebar.markdown("### ⚙️ Panneau de contrôle")
    
    # Navigation via boutons pleine largeur natifs
    if 'sidebar_nav' not in st.session_state:
        st.session_state.sidebar_nav = "Historique"

    if st.sidebar.button("💬 Historique", use_container_width=True,
                         type="primary" if st.session_state.sidebar_nav == "Historique" else "secondary"):
        st.session_state.sidebar_nav = "Historique"
        st.rerun()

    if st.sidebar.button("⚙️ Paramètres", use_container_width=True,
                         type="primary" if st.session_state.sidebar_nav == "Paramètres" else "secondary"):
        st.session_state.sidebar_nav = "Paramètres"
        st.rerun()

    st.sidebar.markdown("---")

    if st.session_state.sidebar_nav == "Historique":
        render_history_section()
    else:
        render_settings_section()
    
    # Footer
    render_footer()



def render_footer():
    """Footer avec branding Kaydan - HTML corrigé"""
    
    # Version simple sans HTML complexe
    st.markdown("---")
    
    # IntelliMath
    st.markdown(
        "<h3 style='text-align:center; font-size:17px; color:#111111; margin:0;'>🧮 <span style='color:white; font-size:17px'>Intelli</span>Math</h3>",
        unsafe_allow_html=True
    )
    
    # Sous-titre
    st.markdown(
        "<p style='text-align:center; color:#111111; font-size:17px; margin:8px 0;'>Lycée : Seconde → Terminale</p>",
        unsafe_allow_html=True
    )
    
    # Badge Kaydan
    st.markdown(
        "<div style='text-align:center; margin:12px 0;'><span style='background:white; color:#FF4500; padding:4px 12px; border-radius:12px; font-size:15px; font-weight:700; letter-spacing:0.5px;'>POWERED BY KAYDAN TECHNOLOGY</span></div>",
        unsafe_allow_html=True 
    )
    


def render_history_section():
    """Section historique des conversations"""
    st.sidebar.markdown("### 💬 Historique")

    # Bouton Nouvelle conversation (toujours visible)
    if st.sidebar.button("✏️ Nouvelle conversation", use_container_width=True, type="primary"):
        _start_new_conversation()

    conversations = st.session_state.get('conversations', [])

    if not conversations:
        st.sidebar.info("Aucune conversation pour le moment.")
        return

    st.sidebar.markdown("---")

    # Liste des conversations, plus récentes d'abord
    for conv in reversed(conversations):
        section_key = conv.get('section_key', '')
        conv_id_key = f"current_conv_id_{section_key}"
        is_current = st.session_state.get(conv_id_key) == conv['id']

        with st.sidebar.container():
            col1, col2 = st.columns([5, 1])

            with col1:
                label = f"{'📌 ' if is_current else '💬 '}{conv['title']}"
                if st.button(label, key=f"conv_{conv['id']}",
                             use_container_width=True,
                             type="primary" if is_current else "secondary"):
                    load_conversation(conv['id'])

            with col2:
                if st.button("🗑️", key=f"del_{conv['id']}", help="Supprimer"):
                    delete_conversation(conv['id'])

            updated = conv.get('updated_at', conv.get('created_at', ''))
            try:
                dt = datetime.fromisoformat(updated)
                nb = len(conv['messages'])
                st.sidebar.caption(f"🕐 {dt.strftime('%d/%m %H:%M')}  •  {nb} msg")
            except Exception:
                pass


def _start_new_conversation():
    """Réinitialise le chat actif pour démarrer une nouvelle conversation."""
    # On efface le lien conv_id pour toutes les sections connues
    for key in list(st.session_state.keys()):
        if key.startswith("current_conv_id_"):
            del st.session_state[key]
        if key.startswith("chat_history_"):
            st.session_state[key] = []
    st.rerun()


def render_settings_section():
    """Section paramètres"""
    st.sidebar.markdown("### ⚙️ Paramètres")
    
    # Paramètres d'affichage
    st.sidebar.markdown("**🎨 Affichage**")
    st.sidebar.checkbox(
        "Afficher les sources",
        value=True,
        key="setting_show_sources",
        help="Afficher les sources utilisées par le RAG"
    )
    
    st.sidebar.checkbox(
        "Mode compact",
        value=False,
        key="setting_compact_mode",
        help="Affichage plus compact des messages"
    )
    
    st.sidebar.markdown("---")
    
    # Paramètres audio
    st.sidebar.markdown("**🔊 Audio**")
    st.sidebar.checkbox(
        "Mode vocal",
        value=False,
        key="setting_voice_mode",
        help="Activer la reconnaissance vocale"
    )
    
    st.sidebar.checkbox(
        "Text-to-Speech",
        value=True,
        key="setting_tts",
        help="Lire les réponses à voix haute"
    )
    
    st.sidebar.markdown("---")
    
    # Paramètres de sauvegarde
    st.sidebar.markdown("**💾 Sauvegarde**")
    st.sidebar.checkbox(
        "Sauvegarde automatique",
        value=True,
        key="setting_auto_save",
        help="Sauvegarder automatiquement les conversations"
    )
    
    if st.sidebar.button("💾 Exporter l'historique", use_container_width='stretch'):
        export_conversations()
    
    st.sidebar.markdown("---")
    
    # Paramètres RAG
    st.sidebar.markdown("**🤖 RAG**")
    st.sidebar.slider(
        "Nombre de sources",
        min_value=1,
        max_value=10,
        value=5,
        key="setting_rag_sources",
        help="Nombre de sources à utiliser pour la réponse"
    )
    
    st.sidebar.markdown("---")
    
    # Actions dangereuses
    st.sidebar.markdown("**⚠️ Zone dangereuse**")
    if st.sidebar.button("🗑️ Effacer tout l'historique", type="secondary"):
        if st.sidebar.checkbox("Je confirme la suppression", key="confirm_delete"):
            st.session_state.conversations = []
            st.session_state.messages = []
            st.session_state.current_conversation = None
            st.success("✅ Historique effacé")
            st.rerun()


def load_conversation(conversation_id: int):
    """Restaure une conversation dans le chat de la section correspondante."""
    for conv in st.session_state.get('conversations', []):
        if conv['id'] == conversation_id:
            section_key = conv.get('section_key', '')
            # Restaurer l'historique de la bonne section
            st.session_state[f"chat_history_{section_key}"] = list(conv['messages'])
            # Marquer comme conversation active
            st.session_state[f"current_conv_id_{section_key}"] = conversation_id
            st.rerun()
            break


def delete_conversation(conversation_id: int):
    """Supprime une conversation et nettoie le chat si c'était l'active."""
    target = next((c for c in st.session_state.get('conversations', [])
                   if c['id'] == conversation_id), None)

    st.session_state.conversations = [
        c for c in st.session_state.get('conversations', [])
        if c['id'] != conversation_id
    ]

    if target:
        section_key = target.get('section_key', '')
        conv_id_key = f"current_conv_id_{section_key}"
        if st.session_state.get(conv_id_key) == conversation_id:
            del st.session_state[conv_id_key]
            st.session_state[f"chat_history_{section_key}"] = []

    st.rerun()


def export_conversations():
    """Exporte les conversations en JSON"""
    import json
    
    if not st.session_state.conversations:
        st.sidebar.warning("Aucune conversation à exporter")
        return
    
    data = {
        'export_date': datetime.now().isoformat(),
        'conversations': st.session_state.conversations
    }
    
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    st.sidebar.download_button(
        label="📥 Télécharger JSON",
        data=json_str,
        file_name=f"intellimath_historique_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
