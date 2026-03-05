"""
Module assets - Chargement des ressources statiques.
Styles CSS personnalisés aux couleurs Kaydan Technology.
"""

import streamlit as st
from pathlib import Path


def load_custom_css():
    """
    Charge les styles CSS personnalisés Kaydan Technology.
    Couleurs : Noir (#2D2D2D) + Orange (#FF6B35)
    """
    css_file = Path(__file__).parent / "styles.css"
    
    if css_file.exists():
        with open(css_file, "r", encoding="utf-8") as f:
            css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Fichier styles.css introuvable dans assets/")


def render_kaydan_logo():
    """
    Affiche le logo Kaydan Technology dans la sidebar.
    """
    st.markdown("""
    <div class="kaydan-logo">
        <div style="font-size: 32px; font-weight: 900; letter-spacing: 3px; color: white;">
            <span class="kaydan-accent">KAYDAN</span>
        </div>
        <div style="font-size: 12px; color: #FF6B35; letter-spacing: 4px; margin-top: 4px;">
            TECHNOLOGY
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_intellimath_header():
    """
    En-tête IntelliMath avec branding Kaydan.
    """
    st.markdown("""
    <div style="text-align: center; padding: 24px 0;">
        <div style="font-size: 48px; margin-bottom: 8px;">🧮</div>
        <h1 style="
            font-size: 42px;
            font-weight: 900;
            color: #2D2D2D;
            margin: 0;
            letter-spacing: -1px;
        ">
            Intelli<span style="color: #FF6B35;">Math</span>
        </h1>
        <p style="
            color: #6C757D;
            font-size: 16px;
            margin-top: 8px;
        ">
            Powered by <span style="color: #FF6B35; font-weight: 600;">Kaydan Technology</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
