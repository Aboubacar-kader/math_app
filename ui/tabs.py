"""
Tabs avec LLM ROUTER INTELLIGENT + NETTOYAGE
Le LLM analyse chaque question et décide de la meilleure action
Les références aux documents sont automatiquement supprimées
"""

import html as _html
import streamlit as st
import streamlit.components.v1 as st_components
from datetime import datetime
from services.rag_service import rag_service
from services.voice_service import voice_service
from core.llm_manager import llm_manager
from utils.pdf_utils import markdown_to_pdf
from utils.geometry_plotter import detect_figure_needed, auto_draw_figure
from ui.components import (
    create_topic_selector,
    create_level_selector,
)
import re


# ============================================================
# NETTOYAGE DES RÉPONSES
# ============================================================

def clean_rag_response(response: str) -> str:
    """Supprime les références aux documents dans les réponses"""
    if not response:
        return response
    
    # Pattern 1 : "Dans un document [filename] (Pertinence : XX%)"
    # [^\]]{0,300} — pas de quantificateur avide sur données non maîtrisées (ReDoS)
    response = re.sub(
        r'Dans un document \[[^\]]{0,300}\]\s*\(Pertinence\s*:\s*\d+%\)',
        '',
        response,
        flags=re.IGNORECASE
    )
    
    # Pattern 2 : "[filename.pdf]" ou "[filename.docx]"
    response = re.sub(r'\[[\w\-_\.]+\.(pdf|docx|txt|png|jpg|jpeg)\]', '', response, flags=re.IGNORECASE)

    # Pattern 3 : "(Pertinence : XX%)"
    response = re.sub(r'\(Pertinence\s*:\s*\d+%\)', '', response)

    # Pattern 4 : "Document X - filename" (avec ou sans gras markdown)
    response = re.sub(r'\*{0,2}Document\s+\d+\s*[-–]\s*[^\n]+\*{0,2}', '', response, flags=re.IGNORECASE)

    # Pattern 5 : "[seconde]", "[premiere]", "[terminale]" (tags de niveau RAG)
    response = re.sub(r'\[\s*(seconde|première|premiere|terminale)\s*\]', '', response, flags=re.IGNORECASE)

    # Pattern 6 : Nettoyer espaces multiples et lignes vides consécutives
    response = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)
    response = re.sub(r'  +', ' ', response)
    
    return response.strip()


def _tabular_to_markdown(inner: str) -> str:
    """Convertit le contenu d'un environnement tabular LaTeX en tableau markdown."""
    # Supprimer la spec de colonnes {c|c|c|...} en début
    inner = re.sub(r'^\s*\{[^}]*\}\s*', '', inner)
    # Supprimer \hline
    inner = re.sub(r'\\hline', '', inner)
    # Séparer les lignes par \\
    rows = [r.strip() for r in re.split(r'\\\\', inner) if r.strip()]
    if not rows:
        return inner
    md_rows = []
    for i, row in enumerate(rows):
        cells = [c.strip() for c in row.split('&')]
        # \boxed{val} → val dans les cellules
        cells = [re.sub(r'\\boxed\{([^}]*)\}', r'**\1**', c) for c in cells]
        md_rows.append('| ' + ' | '.join(cells) + ' |')
        if i == 0:
            md_rows.append('|' + '---|' * len(cells))
    return '\n' + '\n'.join(md_rows) + '\n'


def fix_latex_for_streamlit(text: str) -> str:
    """
    Convertit les environnements LaTeX non supportés par Streamlit/KaTeX
    en blocs $$ ... $$ ou markdown compatibles.
    Streamlit ne supporte QUE $...$ (inline) et $$...$$ (bloc).
    """
    if not text:
        return text

    # ── \begin{tabular}...\end{tabular} → tableau markdown ───────────────────
    # ((?:[^\\]|\\(?!end))*) = tempered greedy token — linéaire, sans ReDoS
    text = re.sub(
        r'\\begin\{(?:tabular|array)\}((?:[^\\]|\\(?!end))*)\\end\{(?:tabular|array)\}',
        lambda m: _tabular_to_markdown(m.group(1)),
        text
    )

    # ── \begin{align*}...\end{align*} → lignes $$ séparées ──────────────────
    def align_to_dollars(match):
        inner = match.group(1)
        lines = re.split(r'\\\\', inner)
        result = []
        for line in lines:
            line = re.sub(r'&', '', line).strip()
            if line:
                result.append(f'$${line}$$')
        return '\n'.join(result)

    text = re.sub(
        r'\\begin\{align\*?\}((?:[^\\]|\\(?!end))*)\\end\{align\*?\}',
        align_to_dollars,
        text
    )

    # ── \begin{equation}...\end{equation} → $$ ... $$ ───────────────────────
    text = re.sub(
        r'\\begin\{equation\*?\}((?:[^\\]|\\(?!end))*)\\end\{equation\*?\}',
        lambda m: f'$${m.group(1).strip()}$$',
        text
    )

    # ── Autres environnements inconnus → supprimer les balises, garder le contenu
    text = re.sub(
        r'\\begin\{[^}]{1,64}\}((?:[^\\]|\\(?!end))*)\\end\{[^}]{1,64}\}',
        lambda m: m.group(1).strip(),
        text
    )

    # ── \boxed{val} hors math → **val** ──────────────────────────────────────
    text = re.sub(r'\\boxed\{([^}]{0,500})\}', r'**\1**', text)

    # ── Commandes d'espacement non supportées ────────────────────────────────
    text = re.sub(r'\\[hv]space\*?\{[^}]{0,100}\}', ' ', text)
    text = re.sub(r'\\(quad|qquad|,|;|!)\b', ' ', text)

    # ── Délimiter les commandes LaTeX non encadrées par $...$ ────────────────
    # (?:[^$]|\$(?!\$))* = un seul $ OK dans $$…$$, mais pas $$ (délimiteur fin)
    MATH_SPLIT = re.compile(r'(\$\$(?:[^$]|\$(?!\$))*\$\$|\$[^\$\n]{1,5000}\$)')

    # Commandes LaTeX hors $...$ → symboles Unicode lisibles
    _LATEX_UNICODE = [
        (r'\\times',          '×'),
        (r'\\cdot',           '·'),
        (r'\\div',            '÷'),
        (r'\\Rightarrow',     '⇒'),
        (r'\\Leftarrow',      '⇐'),
        (r'\\Leftrightarrow', '⟺'),
        (r'\\rightarrow',     '→'),
        (r'\\leftarrow',      '←'),
        (r'\\implies',        '⇒'),
        (r'\\iff',            '⟺'),
        (r'\\neq',            '≠'),
        (r'\\leq',            '≤'),
        (r'\\geq',            '≥'),
        (r'\\approx',         '≈'),
        (r'\\infty',          '∞'),
        (r'\\pm',             '±'),
        (r'\\in\b',           '∈'),
        (r'\\subset',         '⊂'),
        (r'\\mathbb\{R\}',    'ℝ'),
        (r'\\mathbb\{N\}',    'ℕ'),
        (r'\\mathbb\{Z\}',    'ℤ'),
        (r'\\mathbb\{Q\}',    'ℚ'),
        (r'\\mathbb\{C\}',    'ℂ'),
        (r'\\forall',         '∀'),
        (r'\\exists',         '∃'),
    ]

    def _wrap_raw_latex(segment: str) -> str:
        """Remplace les commandes LaTeX hors $...$ par leurs symboles Unicode."""
        for pattern, symbol in _LATEX_UNICODE:
            segment = re.sub(pattern, symbol, segment)
        # \frac{a}{b} → (a)/(b)
        segment = re.sub(r'\\frac\{([^}]{1,500})\}\{([^}]{1,500})\}', r'(\1)/(\2)', segment)
        # \sqrt{x} → √(x)
        segment = re.sub(r'\\sqrt\{([^}]{1,500})\}', r'√(\1)', segment)
        # \lim_{x→a} → lim
        segment = re.sub(r'\\lim_\{([^}]{1,200})\}', r'lim(\1)', segment)
        segment = re.sub(r'\\lim\b', 'lim', segment)
        return segment

    parts = MATH_SPLIT.split(text)
    text = ''.join(
        part if idx % 2 == 1 else _wrap_raw_latex(part)
        for idx, part in enumerate(parts)
    )

    # ── Nettoyer espaces multiples résiduels ─────────────────────────────────
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    return text


# ============================================================
# GESTION DU CONTEXTE PAR SECTION
# ============================================================

def init_chat_context(section_key: str):
    """Initialise le contexte de conversation pour une section"""
    history_key = f"chat_history_{section_key}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []


def add_to_context(section_key: str, role: str, content: str):
    """Ajoute un message au contexte et auto-sauvegarde la conversation"""
    history_key = f"chat_history_{section_key}"
    st.session_state[history_key].append({
        "role": role,
        "content": content
    })
    _autosave_conversation(section_key)


def _autosave_conversation(section_key: str):
    """Crée ou met à jour la conversation courante dans l'historique sidebar."""
    history_key = f"chat_history_{section_key}"
    messages = st.session_state.get(history_key, [])
    if not messages:
        return

    if 'conversations' not in st.session_state:
        st.session_state.conversations = []
    if 'conversation_counter' not in st.session_state:
        st.session_state.conversation_counter = 0

    # Titre = premier message utilisateur (50 chars max)
    first_user = next((m for m in messages if m['role'] == 'user'), None)
    title = (first_user['content'][:50] + "…") if first_user and len(first_user['content']) > 50 else (first_user['content'] if first_user else "Conversation")

    conv_id_key = f"current_conv_id_{section_key}"
    current_id = st.session_state.get(conv_id_key)

    if current_id is not None:
        # Mettre à jour la conversation existante
        for conv in st.session_state.conversations:
            if conv['id'] == current_id:
                conv['messages'] = list(messages)
                conv['updated_at'] = datetime.now().isoformat()
                conv['title'] = title
                break
    else:
        # Créer une nouvelle entrée
        st.session_state.conversation_counter += 1
        new_id = st.session_state.conversation_counter
        st.session_state.conversations.append({
            'id': new_id,
            'title': title,
            'section_key': section_key,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'messages': list(messages)
        })
        st.session_state[conv_id_key] = new_id


_STOP_WORDS = {
    'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'en', 'est',
    'que', 'qui', 'il', 'elle', 'on', 'nous', 'vous', 'ils', 'elles', 'je',
    'tu', 'me', 'te', 'se', 'ce', 'au', 'aux', 'par', 'sur', 'sous', 'dans',
    'avec', 'sans', 'pour', 'pas', 'ne', 'plus', 'ou', 'si', 'mais', 'donc',
    'car', 'or', 'ni', 'comment', 'quel', 'quelle', 'quels', 'quelles',
    'quoi', 'cela', 'ceci', 'tout', 'tous', 'bien', 'très', 'aussi', 'faire',
    'avoir', 'être', 'dire', 'voir', 'savoir', 'vouloir', 'pouvoir', 'aller',
    'moi', 'soi', 'lui', 'leur', 'leurs', 'mon', 'ma', 'mes', 'ton', 'ta',
    # Mots trop génériques pour être discriminants
    'exercice', 'exercices', 'résoudre', 'résolvons', 'calculer', 'calcule',
    'donne', 'donner', 'trouver', 'trouve', 'montrer', 'montrons', 'expliquer',
    'question', 'questions', 'répondre', 'réponse', 'problème', 'problemes',
    'capture', 'image', 'fichier', 'document', 'analyse', 'analyser',
}


def _topic_keywords(text: str) -> set:
    """Extrait les mots significatifs (>3 lettres, hors stop words) d'un texte."""
    import re
    words = re.findall(r'[a-záàâäéèêëîïôöùûüç]{4,}', text.lower())
    return {w for w in words if w not in _STOP_WORDS}


def _is_same_topic(new_question: str, history: list) -> bool:
    """
    Retourne True si la nouvelle question semble dans la continuité
    du dernier échange (au moins 1 mot-clé en commun).
    """
    if not history:
        return False
    # Regarder le dernier message utilisateur dans l'historique
    last_user = next(
        (m['content'] for m in reversed(history) if m['role'] == 'user'),
        None
    )
    if not last_user:
        return False
    kw_prev = _topic_keywords(last_user)
    kw_new = _topic_keywords(new_question)
    # Exige au moins 2 mots-clés communs pour éviter les faux positifs
    return len(kw_prev & kw_new) >= 2


def get_context(section_key: str, current_question: str = "") -> str:
    """
    Récupère les 6 derniers messages (3 échanges) pour le contexte conversationnel,
    uniquement si la question actuelle est dans la continuité du sujet précédent.
    Chaque message est tronqué à 400 caractères max.
    """
    history_key = f"chat_history_{section_key}"

    if history_key not in st.session_state or not st.session_state[history_key]:
        return ""

    history = st.session_state[history_key]

    # Si un nouveau sujet est détecté, ne pas injecter l'ancien contexte
    if current_question and not _is_same_topic(current_question, history):
        return ""

    context_parts = []
    for msg in history[-6:]:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:400] + "…" if len(msg["content"]) > 400 else msg["content"]
        context_parts.append(f"{role}: {content}")

    return "\n".join(context_parts)


def clear_context(section_key: str):
    """Efface le contexte d'une section"""
    history_key = f"chat_history_{section_key}"
    if history_key in st.session_state:
        st.session_state[history_key] = []


def render_chat_messages(section_key: str):
    """Affiche l'historique des messages d'une section"""
    history_key = f"chat_history_{section_key}"
    
    if history_key in st.session_state and st.session_state[history_key]:
        st.markdown("### 💬 Conversation")
        for msg in st.session_state[history_key]:
            avatar = "👨‍🎓" if msg["role"] == "user" else "👨‍🏫"
            with st.chat_message(msg["role"], avatar=avatar):
                if msg["role"] == "user":
                    # Préserve la mise en forme du texte collé (sauts de ligne)
                    st.markdown(msg["content"].replace('\n', '  \n'))
                else:
                    st.markdown(msg["content"])
        st.divider()


# ============================================================
# INTERFACE DE CHAT CONTINUE AVEC LLM ROUTER
# ============================================================

def _split_into_exercises(text: str) -> list:
    """
    Découpe un texte en sections résolubles séparément.
    Priorité :
      1. Plusieurs exercices  → "Exercice 1", "Exercice 2"...
      2. Un exercice avec des parties → "Partie A", "Partie B"...
      3. Aucun marqueur → [(None, texte_entier)]
    """
    # ── 1. Plusieurs exercices ──────────────────────────────────────────────
    ex_pattern = r'(?:^|\n)((?:Exercice|EXERCICE|Exercise)\s*\d+\s*(?:[:\-–—].*)?)'
    parts = re.split(ex_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if len(parts) > 1:
        exercises = []
        for i in range(1, len(parts) - 1, 2):
            titre = parts[i].strip()
            contenu = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if contenu:
                exercises.append((titre, contenu))
        if exercises:
            return exercises

    # ── 2. Un exercice avec plusieurs parties ───────────────────────────────
    partie_pattern = r'(?:^|\n)((?:Partie|PARTIE|Part)\s+[A-Za-z]\s*(?:[:\-–—].*)?)'
    parts = re.split(partie_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if len(parts) > 1:
        header = parts[0].strip()   # texte avant la première partie (énoncé général)
        sections = []
        for i in range(1, len(parts) - 1, 2):
            titre = parts[i].strip()
            contenu = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if contenu:
                # Inclure l'en-tête (données de l'exercice) dans chaque partie
                full = f"{header}\n\n{titre}\n{contenu}" if header else f"{titre}\n{contenu}"
                sections.append((titre, full))
        if sections:
            return sections

    # ── 3. Aucun marqueur → tout d'un bloc ─────────────────────────────────
    return [(None, text.strip())]


def _build_ordered_prompt(exercise_text: str) -> str:
    """
    Pré-extrait les questions numérotées de l'énoncé et les reformate
    en liste ordonnée explicite pour forcer le modèle à les traiter dans l'ordre.

    Exemples reconnus : 1.  1.1.  1.1  a)  A.  •  -
    Retourne le texte original si aucune question détectée.
    """
    # Patterns de numérotation reconnus
    q_pattern = re.compile(
        r'^\s*'
        r'('
        r'\d+(?:\.\d+)*\.?\s'    # 1.  1.1.  1.1
        r'|[a-zA-Z]\)\s'         # a)  A)
        r'|[•\-\*]\s'            # •  -  *
        r')',
        re.MULTILINE
    )

    lines = exercise_text.split('\n')
    questions = []
    header_lines = []
    current_q_lines = []
    current_num = None
    in_questions = False

    for line in lines:
        m = q_pattern.match(line)
        if m:
            in_questions = True
            if current_q_lines and current_num is not None:
                questions.append((current_num.strip(), ' '.join(current_q_lines).strip()))
            current_num = m.group(1)
            current_q_lines = [line[m.end():].strip()]
        elif in_questions and line.strip():
            current_q_lines.append(line.strip())
        elif not in_questions:
            header_lines.append(line)

    if current_q_lines and current_num is not None:
        questions.append((current_num.strip(), ' '.join(current_q_lines).strip()))

    if not questions:
        return exercise_text  # rien à reformater

    header = '\n'.join(header_lines).strip()
    q_list = '\n'.join(
        f"  {num} {content}" for num, content in questions
    )

    return (
        f"{header}\n\n" if header else ""
    ) + (
        f"Questions à traiter DANS CET ORDRE EXACT :\n{q_list}"
    )


def _store_file_names(section_key: str):
    """Callback on_change : stocke les noms ET les octets bruts pour traitement panel fermé."""
    files = st.session_state.get(f"uploader_{section_key}") or []
    if not files:
        return
    if 'uploaded_files_names' not in st.session_state:
        st.session_state.uploaded_files_names = []
    saved = {}
    for f in files:
        if f.name not in st.session_state.uploaded_files_names:
            st.session_state.uploaded_files_names.append(f.name)
        try:
            f.seek(0)
            saved[f.name] = {'bytes': f.read(), 'type': f.type, 'name': f.name, 'size': f.size}
            f.seek(0)
        except Exception:
            pass
    if saved:
        st.session_state['_saved_file_data'] = saved




def _process_question(section_key: str, question: str, rag_function_type: str, level: str = None, docs_content: list = None):
    """Traite une question via le LLM Router + RAG et l'ajoute au contexte."""
    if docs_content is None:
        docs_content = st.session_state.get('uploaded_documents_content') or []

    # Classifier la question
    with st.spinner("🤔 Analyse de ta question..."):
        category, _ = llm_manager.classify_query(question)

    # Si des documents sont présents et que la question est conversationnelle,
    # l'utilisateur s'exprime à propos du document → forcer MATH_RAG
    if docs_content and category == "CONVERSATION":
        category = "MATH_RAG"

    add_to_context(section_key, "user", question)

    if category == "CONVERSATION":
        response = llm_manager.get_conversation_response(question)
        add_to_context(section_key, "assistant", response)
        st.rerun()
        return

    if category == "OUT_OF_SCOPE":
        response = llm_manager.get_out_of_scope_response()
        add_to_context(section_key, "assistant", response)
        st.rerun()
        return

    # Extraire le contenu brut du document (pour la classification + le prompt)
    doc_context = ""
    if docs_content:
        doc_parts = [doc.get('text', '')[:4000] for doc in docs_content]
        doc_context = "\n\n".join(doc_parts)

    # MATH_RAG — classification de l'intention : EXERCICE ou COURS
    # Si un document est fourni, on classifie d'après son contenu (pas seulement la question)
    classify_text = f"{question} {doc_context[:600]}" if doc_context else question
    math_intent = llm_manager.classify_math_intent(classify_text, rag_function_type)
    conv_context = get_context(section_key, current_question=question)

    conv_part = f"\n\nContexte de la conversation :\n{conv_context}" if conv_context else ""

    # ── EXERCICE : résolution directe via ollama.chat() (system/user séparés) ──
    if math_intent == "EXERCICE":
        level_info = f"NIVEAU : {level}" if level else ""

        # ── Requête RAG avant la construction du prompt ─────────────────────
        rag_query = doc_context[:400] if doc_context else question[:400]
        try:
            rag_knowledge, _ = rag_service.query_with_sources(rag_query, top_k=3)
            rag_knowledge = rag_knowledge.strip() if rag_knowledge else ""
        except Exception:
            rag_knowledge = ""

        rag_section = ""
        if rag_knowledge:
            rag_section = f"""
⚠️ RÈGLE N°4 — JUSTIFICATIONS OBLIGATOIRES :
Pour toute justification, démonstration ou explication, appuie-toi sur les
propriétés et théorèmes ci-dessous. Cite-les explicitement dans ta réponse :
"D'après la définition de ...", "Par le théorème de ...", "En appliquant la propriété ...".

📚 Propriétés, théorèmes et définitions du cours :
{rag_knowledge[:1200]}"""

        # Quand un document est fourni, ne pas injecter l'historique (il confond le modèle)
        conv_part_ex = "" if doc_context else conv_part

        exercise_system = f"""Tu es IntelliMath, professeur de mathématiques de lycée.{(' ' + level_info) if level_info else ''}

🚫 INTERDICTIONS ABSOLUES — à respecter SANS EXCEPTION :
- JAMAIS dire "revenez avec les questions suivantes", "je continuerai", "dans un second temps" ou toute phrase similaire.
- JAMAIS t'arrêter avant d'avoir résolu la DERNIÈRE question de l'énoncé.
- JAMAIS demander à l'élève de poser une autre question pour continuer.
- JAMAIS dire que tu n'as pas accès au document ou que tu as besoin d'autres informations.
- Tu DOIS résoudre TOUTES les questions, TOUTES les parties, dans UNE SEULE réponse complète.

⚠️ RÈGLE N°1 — SUIS L'ÉNONCÉ À LA LETTRE :
- Lis attentivement l'énoncé fourni, identifie toutes les données et toutes les questions.
- Réponds à CHAQUE question posée, dans leur ordre exact, de la première à la dernière.
- Copie le numéro/intitulé de chaque question EXACTEMENT comme il apparaît (1., 1.1., Partie A, etc.).

⚠️ RÈGLE N°2 — CALCULS :
- Symboles directement dans le texte : × ÷ ⇒ → ≤ ≥ ≠ ≈ ∞ ± ∈ ℝ ² ³ √ Δ
- Exposants : x², x³, xⁿ — Fractions : (−b ± √Δ) / 2a — Indices : x₁, x₂
- JAMAIS : \\times, \\Rightarrow, \\mathbb{{R}}, \\begin{{...}}, \\boxed{{...}}, $...$, $$...$$
- Parenthèses OBLIGATOIRES autour de tout nombre négatif : f(−1) = 2 × (−1) − 3
- Détaille chaque étape (× ÷ avant + −)

⚠️ RÈGLE N°3 — TABLEAUX DE VALEURS :
Si l'énoncé demande un tableau de valeurs, calcule CHAQUE valeur et affiche un tableau markdown COMPLET :
| x   | v1 | v2 | v3 | v4 | v5 |
|-----|----|----|----|----|-----|
| f(x)| r1 | r2 | r3 | r4 | r5 |
→ Remplace chaque cellule par la valeur calculée réelle. Ne laisse JAMAIS de case vide ou avec "?".
{rag_section}
⚠️ RÈGLE N°4 — JUSTIFICATIONS :
✓ "D'après la définition d'une fonction affine f(x) = ax + b, ici a = 2 et b = −3."
✓ "Par la propriété de croissance : comme a = 2 > 0, la fonction est croissante sur ℝ."

FORMAT OBLIGATOIRE (partie par partie, question par question) :
📋 **Données** : [reprend les données clés de l'énoncé]
✏️ **Résolution**
**[Titre de la partie ou numéro de question]** [Réponse complète et détaillée]
...jusqu'à la DERNIÈRE question.
📝 **Conclusion** : [synthèse brève]{conv_part_ex}"""

        if doc_context:
            exercises = _split_into_exercises(doc_context)
        else:
            exercises = [(None, question)]

        if len(exercises) <= 1:
            titre, contenu = exercises[0]
            exercise_text = contenu if doc_context else question
            if doc_context:
                exercise_user = f"Voici l'exercice complet à résoudre :\n\n{exercise_text[:5000]}\n\nRésous TOUTES les questions dans l'ordre exact."
            else:
                exercise_user = f"Voici l'exercice à résoudre :\n\n{exercise_text}\n\nRésous TOUTES les questions dans l'ordre exact."

            with st.spinner("✏️ Résolution en cours..."):
                try:
                    response = fix_latex_for_streamlit(
                        rag_service._call_llm(exercise_system, exercise_user, long=True)
                    )
                    add_to_context(section_key, "assistant", response)
                except Exception as e:
                    add_to_context(section_key, "assistant", f"❌ Erreur : {str(e)}")
        else:
            # Plusieurs exercices → résolution séquentielle, un par un
            all_responses = []
            n = len(exercises)
            for idx, (titre, contenu) in enumerate(exercises, 1):
                import time as _time
                label = titre if titre else f"Exercice {idx}"
                with st.spinner(f"✏️ Résolution {idx}/{n} — {label}..."):
                    try:
                        if idx > 1:
                            _time.sleep(1)  # éviter le rate limiting de l'API
                        exercise_user = f"Voici l'exercice à résoudre :\n\n{label}\n\n{contenu[:4000]}\n\nRésous TOUTES les questions dans l'ordre exact."
                        response = fix_latex_for_streamlit(
                            rag_service._call_llm(exercise_system, exercise_user, long=True)
                        )
                        all_responses.append(f"## {label}\n\n{response}")
                    except Exception as e:
                        all_responses.append(f"## {label}\n\n❌ Erreur : {str(e)}")

            add_to_context(section_key, "assistant", "\n\n---\n\n".join(all_responses))

    # ── COURS / DÉFINITION / PROPRIÉTÉ / THÉORÈME ──────────────────────────────
    else:
        # Document uploadé → LLM direct avec le contenu complet du fichier
        if doc_context:
            level_info = f"NIVEAU : {level}" if level else ""
            cours_system = f"""Tu es IntelliMath, professeur de mathématiques de lycée.{(' ' + level_info) if level_info else ''}

Le document fourni est le contexte de référence. Lis-le entièrement et réponds à la question à partir de ce document.
- Symboles directement dans le texte : × ÷ ⇒ → ≤ ≥ ≠ ≈ ∞ ± ∈ ℝ ² ³ √ Δ
- Exposants : x², x³ — Fractions : (−b ± √Δ) / 2a — Indices : x₁, x₂
- JAMAIS : \\times, \\Rightarrow, \\mathbb{{R}}, \\begin{{...}}, \\boxed{{...}}, $...$, $$...$$
- Tableau de valeurs → tableau markdown (JAMAIS \\begin{{tabular}}){conv_part}"""

            cours_user = f"Question : {question}\n\nDocument :\n{doc_context}"
            with st.spinner("📄 Analyse du document..."):
                try:
                    response = fix_latex_for_streamlit(rag_service._call_llm(cours_system, cours_user))
                    add_to_context(section_key, "assistant", response)
                except Exception as e:
                    add_to_context(section_key, "assistant", f"❌ Erreur : {str(e)}")

        # Pas de document → RAG classique sur la base de connaissances
        else:
            rag_query = question  # vector search = question seule (conv_context injecté dans le prompt LLM)
            with st.spinner("📚 Recherche dans la base de cours..."):
                try:
                    if rag_function_type == "definition" and level:
                        response, _ = rag_service.get_definition_with_sources(question, level)
                    else:
                        response, _ = rag_service.query_with_sources(rag_query, top_k=5)
                    response = fix_latex_for_streamlit(clean_rag_response(response))
                    add_to_context(section_key, "assistant", response)
                except Exception as e:
                    add_to_context(section_key, "assistant", f"❌ Erreur : {str(e)}")

    # ── Post-traitement de la dernière réponse ──────────────────
    from utils.variation_table import parse_variation_block, strip_variation_block

    history_key = f'chat_history_{section_key}'
    last_llm_response = next(
        (m['content'] for m in reversed(st.session_state.get(history_key, []))
         if m.get('role') == 'assistant'), ''
    )

    # Détecter un tableau de variations dans la réponse
    vt_data = parse_variation_block(last_llm_response)
    if vt_data:
        st.session_state[f'variation_table_{section_key}'] = vt_data
        # Supprimer le bloc brut du texte affiché dans le chat
        history = st.session_state.get(history_key, [])
        for msg in reversed(history):
            if msg.get('role') == 'assistant':
                msg['content'] = strip_variation_block(msg['content'])
                break
    else:
        st.session_state.pop(f'variation_table_{section_key}', None)
    detection_text = '\n'.join(filter(None, [
        question,
        doc_context[:400] if doc_context else '',
        last_llm_response[:600],
    ]))
    fig_detection = detect_figure_needed(detection_text)
    if fig_detection.get('needs_figure'):
        st.session_state[f'figure_detection_{section_key}'] = fig_detection
    else:
        st.session_state.pop(f'figure_detection_{section_key}', None)

    # TTS automatique si mode vocal activé
    if st.session_state.get('setting_voice_mode') and last_llm_response:
        st.session_state[f'pending_tts_{section_key}'] = last_llm_response

    st.rerun()


def render_continuous_chat(
    section_key: str,
    placeholder: str,
    rag_function_type: str,
    level: str = None
):
    """
    Chat continu avec LLM Router intelligent + nettoyage automatique

    Args:
        section_key: Identifiant unique de la section
        placeholder: Texte du placeholder
        rag_function_type: Type de fonction RAG ("query", "definition", "exercise")
        level: Niveau scolaire optionnel
    """

    # Initialiser
    init_chat_context(section_key)

    # Traiter une question vocale en attente (transcrite au rendu précédent)
    pending_key = f'pending_voice_{section_key}'
    if st.session_state.get(pending_key):
        voice_q = st.session_state.pop(pending_key)
        docs_snapshot = list(st.session_state.get('uploaded_documents_content') or [])
        st.session_state.uploaded_files_names = []
        st.session_state.uploaded_documents_content = []
        _process_question(section_key, voice_q, rag_function_type, level, docs_snapshot)
        return

    # Afficher l'historique + bouton "Nouvelle conversation"
    history_key = f"chat_history_{section_key}"
    if st.session_state.get(history_key):
        col_title, col_btn = st.columns([8, 2])
        with col_btn:
            if st.button("✦ Nouvelle conversation", key=f"new_conv_{section_key}", use_container_width=True):
                st.session_state[history_key] = []
                st.session_state.pop(f'figure_detection_{section_key}', None)
                st.session_state.pop(f'current_conv_id_{section_key}', None)
                st.session_state.uploaded_files_names = []
                st.session_state.uploaded_documents_content = []
                st.session_state.pop('_saved_file_data', None)
                st.rerun()

    render_chat_messages(section_key)

    # Tracé automatique si un graphique a été détecté
    fig_key = f'figure_detection_{section_key}'
    if st.session_state.get(fig_key):
        detection = st.session_state[fig_key]
        fig_plot = auto_draw_figure(detection)
        if fig_plot:
            func_display = detection.get('parameters', {}).get('function', '')
            if func_display:
                st.markdown(f"**📈 Représentation graphique** — $f(x) = {func_display}$")
            else:
                st.markdown(f"**📐 Figure — {detection.get('figure_type', '').capitalize()}**")
            st.plotly_chart(fig_plot, use_container_width=True)

    # Tableau de variations (si détecté dans la dernière réponse)
    vt_key = f'variation_table_{section_key}'
    if st.session_state.get(vt_key):
        from utils.variation_table import render_variation_table
        vt = st.session_state[vt_key]
        try:
            st.markdown("**📊 Tableau de variations**")
            html = render_variation_table(vt['x_labels'], vt['signs'], vt['f_values'])
            st.markdown(html, unsafe_allow_html=True)
        except Exception:
            pass

    # Lecture vocale automatique si mode vocal activé
    tts_key = f'pending_tts_{section_key}'
    if st.session_state.get(tts_key) and st.session_state.get('setting_voice_mode'):
        audio_html = voice_service.text_to_speech(st.session_state[tts_key], auto_play=True)
        if audio_html:
            st.markdown("🔊 *Lecture de la réponse...*")
            st.markdown(audio_html, unsafe_allow_html=True)
        del st.session_state[tts_key]

    # Indicateur de contexte
    '''history_key = f"chat_history_{section_key}"
    if st.session_state.get(history_key):
        nb_messages = len(st.session_state[history_key])
        nb_exchanges = nb_messages // 2
        st.caption(f"💬 {nb_messages} message(s) ({nb_exchanges} échange(s)) • Mémoire : 40 messages max")
    '''
    # Panel upload — affiché AU-DESSUS du chatbar
    if st.session_state.get(f'show_upload_{section_key}'):
        st.markdown("""
        <style>
        .upload-panel {
            border: 1px solid #e5e5e5;
            border-radius: 16px;
            padding: 16px 20px;
            background: #fafafa;
            margin-bottom: 10px;
            max-width: 700px;
        }
        </style>
        <div class="upload-panel">
        """, unsafe_allow_html=True)

        col_title, col_close = st.columns([9, 1])
        with col_title:
            st.markdown("**📎 Uploader des documents**")
        with col_close:
            if st.button("✖", key=f"close_upload_{section_key}", help="Fermer"):
                st.session_state[f'show_upload_{section_key}'] = False
                st.rerun()

        uploaded_files = st.file_uploader(
            "Glissez vos fichiers ici",
            type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key=f"uploader_{section_key}",
            label_visibility="collapsed",
            on_change=_store_file_names,
            args=(section_key,)
        )
        if uploaded_files:
            # Auto-traitement dès que des fichiers sont présents (pas besoin de bouton manuel)
            current_names = sorted([f.name for f in uploaded_files])
            already_loaded = sorted([
                d.get('filename', '') for d in (st.session_state.get('uploaded_documents_content') or [])
            ])
            if current_names != already_loaded:
                from services.document_processor import document_processor
                with st.spinner("📖 Extraction du texte..."):
                    try:
                        processed_docs = document_processor.process_files(uploaded_files)
                    except Exception as e:
                        st.error(f"❌ Erreur lors du traitement : {e}")
                        processed_docs = []
                if processed_docs:
                    st.session_state.uploaded_documents_content = []
                    for doc in processed_docs:
                        st.session_state.uploaded_documents_content.append({
                            'filename': doc['metadata'].get('filename', 'Document'),
                            'text': doc['text']
                        })
                    st.rerun()  # Rerun uniquement si extraction réussie (évite boucle infinie)

        st.markdown('</div>', unsafe_allow_html=True)

    # Tags fichiers persistants
    _names = st.session_state.get('uploaded_files_names') or []
    if _names:
        st.info("📎 " + "  ·  ".join(_names))
        if st.button("✕ Retirer les fichiers", key=f"clear_files_{section_key}"):
            st.session_state.uploaded_files_names = []
            st.session_state.uploaded_documents_content = []
            st.session_state.pop('_saved_file_data', None)
            st.rerun()

    # Barre de saisie TOUJOURS visible
    with st.form(key=f"form_{section_key}", clear_on_submit=True):

        st.markdown("""
        <style>
        .gpt-bar {
            display: flex;
            align-items: center;
            gap: 10px;
            border-radius: 40px;
            border: 1px solid #e5e5e5;
            padding: 8px 15px;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }
        .gpt-bar textarea {
            border: none !important;
            outline: none !important;
            resize: none !important;
            box-shadow: none !important;
            font-size: 15px !important;
        }
        .icon-circle button {
            border-radius: 50% !important;
            width: 36px !important;
            height: 36px !important;
            padding: 0 !important;
            font-size: 18px !important;
            border: none !important;
        }
        .send-circle button {
            background: black !important;
            color: white !important;
        }
        .icon-circle button:hover {
            background: #f2f2f2 !important;
        }
        /* Sélecteurs natifs Streamlit (appliqués réellement dans le DOM) */
        div[data-testid="stForm"] {
            border-radius: 50px !important;
            border: 2px solid #E8E8E8 !important;
            padding: 0 !important;
            background: #f4f4f4 !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
            overflow: hidden !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
        }
        div[data-testid="stForm"]:focus-within {
            border-color: #FF4500 !important;
            box-shadow: 0 4px 12px rgba(255, 69, 0, 0.15) !important;
        }
        div[data-testid="stForm"] .stHorizontalBlock {
            align-items: center !important;
            gap: 0 !important;
            padding: 4px 6px !important;
        }
        div[data-testid="stForm"] .stColumn {
            padding: 0 !important;
        }
        div[data-testid="stForm"] .stColumn:first-child {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-width: 48px !important;
            max-width: 48px !important;
            flex-shrink: 0 !important;
            padding-left: 10px !important;
        }
        div[data-testid="stForm"] .stColumn:last-child,
        div[data-testid="stForm"] .stColumn:nth-last-child(2) {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            min-width: 48px !important;
            max-width: 48px !important;
            flex-shrink: 0 !important;
        }
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        div[data-testid="stForm"] .stTextArea,
        div[data-testid="stForm"] .stTextArea > div,
        div[data-testid="stForm"] .stTextArea > div > div,
        div[data-testid="stForm"] .stTextArea [data-baseweb="textarea"],
        div[data-testid="stForm"] .stTextArea [data-baseweb="base-input"] {
            border: none !important;
            box-shadow: none !important;
            border-radius: 0 !important;
            background: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        div[data-testid="stForm"] .stTextArea textarea {
            border: none !important;
            outline: none !important;
            resize: none !important;
            box-shadow: none !important;
            font-size: 15px !important;
            background: transparent !important;
            padding: 8px 0 !important;
            color: #2D2D2D !important;
        }
        button[kind="secondaryFormSubmit"] {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 50% !important;
            width: 36px !important;
            height: 36px !important;
            padding: 0 !important;
            font-size: 18px !important;
            border: none !important;
            background: #e8e8e8 !important;
            color: #333 !important;
        }
        button[kind="secondaryFormSubmit"]:hover {
            background: #d0d0d0 !important;
        }
        button[kind="primaryFormSubmit"] {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
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

        col1, col2, col3, col4 = st.columns([1, 10, 1, 1])

        with col1:
            upload_clicked = st.form_submit_button("📎", help="Uploader un fichier")

        with col2:
            user_input = st.text_area(
                "",
                placeholder=placeholder,
                height=40,
                key=f"input_{section_key}",
                label_visibility="collapsed"
            )

        with col3:
            voice_clicked = st.form_submit_button("🎤", help="Saisie vocale")

        with col4:
            submitted = st.form_submit_button("➤", type="primary", help="Envoyer")

        if upload_clicked:
            st.session_state[f'show_upload_{section_key}'] = not st.session_state.get(f'show_upload_{section_key}', False)

        if voice_clicked:
            st.session_state[f'show_voice_{section_key}'] = True

        if submitted:
            typed = user_input.strip() if user_input else ""
            docs_content = st.session_state.get('uploaded_documents_content') or []
            file_names = st.session_state.get('uploaded_files_names') or []

            # ── Retry 1 : panel encore ouvert → utiliser les fichiers du widget ──
            if not docs_content:
                raw_files = st.session_state.get(f"uploader_{section_key}") or []
                if raw_files:
                    from services.document_processor import document_processor
                    with st.spinner("📖 Extraction du texte..."):
                        try:
                            _pdocs = document_processor.process_files(raw_files)
                            if _pdocs:
                                docs_content = [
                                    {'filename': d['metadata'].get('filename', 'Document'), 'text': d['text']}
                                    for d in _pdocs
                                ]
                                st.session_state.uploaded_documents_content = docs_content
                        except Exception:
                            pass

            # ── Retry 2 : panel fermé → utiliser les octets sauvegardés au moment de la sélection ──
            if not docs_content:
                saved_data = st.session_state.get('_saved_file_data') or {}
                if saved_data:
                    import io as _io
                    class _FakeFile:
                        def __init__(self, d):
                            self.name = d['name']; self.type = d['type']; self.size = d['size']
                            self._buf = _io.BytesIO(d['bytes'])
                        def read(self, *a): return self._buf.read(*a)
                        def seek(self, *a): return self._buf.seek(*a)
                        def getbuffer(self): return self._buf.getbuffer()
                    fake_files = [_FakeFile(d) for d in saved_data.values()]
                    from services.document_processor import document_processor
                    with st.spinner("📖 Extraction du texte..."):
                        try:
                            _pdocs = document_processor.process_files(fake_files)
                            if _pdocs:
                                docs_content = [
                                    {'filename': d['metadata'].get('filename', 'Document'), 'text': d['text']}
                                    for d in _pdocs
                                ]
                                st.session_state.uploaded_documents_content = docs_content
                        except Exception:
                            pass

            has_docs = bool(docs_content or file_names)

            # Fichiers déclarés mais extraction échouée (LLaVA absent, format invalide…)
            if file_names and not docs_content:
                ext_list = [n.rsplit('.', 1)[-1].lower() for n in file_names]
                if any(e in ('png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'tiff') for e in ext_list):
                    hint = "Vérifiez que LLaVA est installé (`ollama pull llava`) ou convertissez l'image en PDF."
                else:
                    hint = "Vérifiez que le fichier n'est pas vide ni corrompu."
                st.error(f"⚠️ Le contenu du fichier n'a pas pu être extrait. {hint}")
            elif typed or has_docs:
                docs_snapshot = list(docs_content)
                st.session_state.uploaded_files_names = []
                st.session_state.uploaded_documents_content = []
                st.session_state.pop('_saved_file_data', None)
                question = typed if typed else "Analyse et résous ce document"
                _process_question(section_key, question, rag_function_type, level, docs_snapshot)

    # Enter = soumettre, Shift+Enter = nouvelle ligne
    st_components.html(f"""<script>
    (function() {{
        var doc = window.parent.document;
        function setup() {{
            doc.querySelectorAll('textarea').forEach(function(ta) {{
                if (ta.__enterBound) return;
                ta.__enterBound = true;
                ta.addEventListener('keydown', function(e) {{
                    if (e.key === 'Enter' && !e.shiftKey) {{
                        e.preventDefault();
                        var btn = doc.querySelector('button[kind="primaryFormSubmit"]');
                        if (btn) btn.click();
                    }}
                }});
            }});
        }}
        setup();
        new MutationObserver(setup).observe(doc.body, {{childList: true, subtree: true}});
    }})();
    </script>""", height=1)

    # Widget audio hors du form (contrainte Streamlit)
    if st.session_state.get(f'show_voice_{section_key}'):
        st.markdown("---")
        col_title, col_close = st.columns([9, 1])
        with col_title:
            st.caption("🎤 Parle, ta question sera transcrite dans le chatbar.")
        with col_close:
            if st.button("✖", key=f"close_voice_{section_key}", help="Fermer"):
                st.session_state[f'show_voice_{section_key}'] = False
                st.rerun()

        transcribed = voice_service.render_audio_input()
        if transcribed:
            st.session_state[f'show_voice_{section_key}'] = False
            st.session_state[f'pending_voice_{section_key}'] = transcribed
            st.rerun()


# ============================================================
# FORMULES DE RÉFÉRENCE (même base de données qu'avant)
# ============================================================

FORMULES_REFERENCE = {
    "Seconde": {
        "📐 Géométrie": [
            {
                "titre": "Théorème de Pythagore",
                "formule": "a² + b² = c²",
                "description": "Dans un triangle rectangle",
                "legende": "a, b = côtés de l'angle droit • c = hypoténuse"
            },
            {
                "titre": "Théorème de Thalès",
                "formule": "AB/AB' = AC/AC' = BC/B'C'",
                "description": "Droites parallèles dans un triangle",
                "legende": "A, B, C = points du grand triangle • B', C' = points du petit triangle parallèle"
            },
            {
                "titre": "Aire d'un triangle",
                "formule": "A = (base × hauteur) / 2",
                "description": "Formule générale",
                "legende": "A = aire • base = un côté quelconque • hauteur = perpendiculaire à la base"
            },
            {
                "titre": "Périmètre du cercle",
                "formule": "P = 2πr = πd",
                "description": "Circonférence",
                "legende": "P = périmètre • r = rayon • d = diamètre • π ≈ 3,14159"
            },
            {
                "titre": "Aire du cercle",
                "formule": "A = πr²",
                "description": "Surface du disque",
                "legende": "A = aire • r = rayon • π ≈ 3,14159"
            },
            {
                "titre": "Périmètre rectangle",
                "formule": "P = 2(L + l)",
                "description": "Somme des côtés",
                "legende": "P = périmètre • L = longueur • l = largeur"
            },
            {
                "titre": "Aire rectangle",
                "formule": "A = L × l",
                "description": "Surface",
                "legende": "A = aire • L = longueur • l = largeur"
            },
            {
                "titre": "Volume cube",
                "formule": "V = a³",
                "description": "Cube parfait",
                "legende": "V = volume • a = longueur de l'arête"
            },
            {
                "titre": "Volume parallélépipède",
                "formule": "V = L × l × h",
                "description": "Pavé droit",
                "legende": "V = volume • L = longueur • l = largeur • h = hauteur"
            },
            {
                "titre": "Volume cylindre",
                "formule": "V = πr²h",
                "description": "Cylindre de révolution",
                "legende": "V = volume • r = rayon de la base • h = hauteur • π ≈ 3,14159"
            },
        ],
        
        "🔢 Calcul littéral": [
            {
                "titre": "Identité remarquable 1",
                "formule": "(a + b)² = a² + 2ab + b²",
                "description": "Carré d'une somme",
                "legende": "a, b = nombres ou expressions quelconques"
            },
            {
                "titre": "Identité remarquable 2",
                "formule": "(a - b)² = a² - 2ab + b²",
                "description": "Carré d'une différence",
                "legende": "a, b = nombres ou expressions quelconques"
            },
            {
                "titre": "Identité remarquable 3",
                "formule": "(a + b)(a - b) = a² - b²",
                "description": "Différence de deux carrés",
                "legende": "a, b = nombres ou expressions quelconques"
            },
            {
                "titre": "Distributivité",
                "formule": "k(a + b) = ka + kb",
                "description": "Facteur commun",
                "legende": "k = facteur commun • a, b = termes à distribuer"
            },
            {
                "titre": "Puissances",
                "formule": "aⁿ × aᵐ = aⁿ⁺ᵐ\naⁿ / aᵐ = aⁿ⁻ᵐ\n(aⁿ)ᵐ = aⁿˣᵐ",
                "description": "Propriétés des puissances",
                "legende": "a = base (nombre) • n, m = exposants (puissances)"
            },
        ],
        
        "📊 Fonctions": [
            {
                "titre": "Fonction affine",
                "formule": "f(x) = ax + b",
                "description": "Droite non horizontale",
                "legende": "a = coefficient directeur (pente) • b = ordonnée à l'origine • x = variable"
            },
            {
                "titre": "Fonction linéaire",
                "formule": "f(x) = ax",
                "description": "Droite passant par l'origine",
                "legende": "a = coefficient directeur (pente) • x = variable"
            },
            {
                "titre": "Fonction carré",
                "formule": "f(x) = x²",
                "description": "Parabole centrée",
                "legende": "x = variable • f(x) = image de x"
            },
            {
                "titre": "Fonction inverse",
                "formule": "f(x) = 1/x",
                "description": "Hyperbole",
                "legende": "x = variable (x ≠ 0) • f(x) = image de x"
            },
            {
                "titre": "Taux de variation",
                "formule": "t = (f(b) - f(a)) / (b - a)",
                "description": "Pente entre deux points",
                "legende": "t = taux de variation • a, b = valeurs de x • f(a), f(b) = images correspondantes"
            },
        ],
        
        "📈 Statistiques": [
            {
                "titre": "Moyenne",
                "formule": "x̄ = (x₁ + x₂ + ... + xₙ) / n",
                "description": "Valeur centrale",
                "legende": "x̄ = moyenne • x₁, x₂, ..., xₙ = valeurs • n = nombre de valeurs"
            },
            {
                "titre": "Médiane",
                "formule": "Me",
                "description": "Valeur centrale (50%)",
                "legende": "Me = médiane (valeur qui sépare les données en deux groupes égaux)"
            },
            {
                "titre": "Étendue",
                "formule": "e = max - min",
                "description": "Écart entre extrêmes",
                "legende": "e = étendue • max = valeur maximale • min = valeur minimale"
            },
            {
                "titre": "Fréquence",
                "formule": "f = effectif / effectif total",
                "description": "Proportion",
                "legende": "f = fréquence (entre 0 et 1) • effectif = nombre d'occurrences"
            },
        ],
    },
    
    "Première": {
        "📐 Trigonométrie": [
            {
                "titre": "Formule fondamentale",
                "formule": "cos²(x) + sin²(x) = 1",
                "description": "Cercle trigonométrique",
                "legende": "x = angle (en radians) • cos(x) = cosinus • sin(x) = sinus"
            },
            {
                "titre": "Tangente",
                "formule": "tan(x) = sin(x) / cos(x)",
                "description": "Rapport sinus/cosinus",
                "legende": "x = angle (en radians) • tan(x) = tangente • cos(x) ≠ 0"
            },
            {
                "titre": "Valeurs remarquables π/6",
                "formule": "cos(π/6) = √3/2\nsin(π/6) = 1/2\ntan(π/6) = √3/3",
                "description": "Angle de 30°",
                "legende": "π/6 rad = 30° • π ≈ 3,14159"
            },
            {
                "titre": "Valeurs remarquables π/4",
                "formule": "cos(π/4) = √2/2\nsin(π/4) = √2/2\ntan(π/4) = 1",
                "description": "Angle de 45°",
                "legende": "π/4 rad = 45° • π ≈ 3,14159"
            },
            {
                "titre": "Valeurs remarquables π/3",
                "formule": "cos(π/3) = 1/2\nsin(π/3) = √3/2\ntan(π/3) = √3",
                "description": "Angle de 60°",
                "legende": "π/3 rad = 60° • π ≈ 3,14159"
            },
            {
                "titre": "Formules d'addition",
                "formule": "cos(a + b) = cos(a)cos(b) - sin(a)sin(b)\nsin(a + b) = sin(a)cos(b) + cos(a)sin(b)",
                "description": "Somme d'angles",
                "legende": "a, b = angles (en radians)"
            },
            {
                "titre": "Formules de duplication",
                "formule": "cos(2x) = cos²(x) - sin²(x)\nsin(2x) = 2sin(x)cos(x)",
                "description": "Angle double",
                "legende": "x = angle (en radians) • 2x = angle double"
            },
        ],
        
        "📊 Suites": [
            {
                "titre": "Suite arithmétique",
                "formule": "uₙ = u₀ + nr",
                "description": "Terme général",
                "legende": "uₙ = terme de rang n • u₀ = premier terme • n = rang • r = raison (différence constante)"
            },
            {
                "titre": "Somme arithmétique",
                "formule": "Sₙ = n × (u₀ + uₙ) / 2",
                "description": "Somme des n premiers termes",
                "legende": "Sₙ = somme • n = nombre de termes • u₀ = premier terme • uₙ = dernier terme"
            },
            {
                "titre": "Suite géométrique",
                "formule": "uₙ = u₀ × qⁿ",
                "description": "Terme général",
                "legende": "uₙ = terme de rang n • u₀ = premier terme • n = rang • q = raison (quotient constant)"
            },
            {
                "titre": "Somme géométrique",
                "formule": "Sₙ = u₀ × (1 - qⁿ⁺¹) / (1 - q)",
                "description": "Somme des n+1 premiers termes",
                "legende": "Sₙ = somme • u₀ = premier terme • n = nombre de termes - 1 • q = raison (q ≠ 1)"
            },
            {
                "titre": "Sens de variation",
                "formule": "uₙ₊₁ - uₙ",
                "description": "Différence entre termes consécutifs",
                "legende": "Si > 0 : suite croissante • Si < 0 : suite décroissante • Si = 0 : suite constante"
            },
        ],
        
        "📈 Dérivées": [
            {
                "titre": "Dérivée constante",
                "formule": "(k)' = 0",
                "description": "Constante",
                "legende": "k = constante (nombre fixe) • 0 = dérivée nulle"
            },
            {
                "titre": "Dérivée puissance",
                "formule": "(xⁿ)' = nxⁿ⁻¹",
                "description": "Puissance de x",
                "legende": "x = variable • n = exposant (nombre réel) • n-1 = exposant diminué de 1"
            },
            {
                "titre": "Dérivée somme",
                "formule": "(u + v)' = u' + v'",
                "description": "Linéarité",
                "legende": "u, v = fonctions dérivables • u', v' = dérivées respectives"
            },
            {
                "titre": "Dérivée produit",
                "formule": "(uv)' = u'v + uv'",
                "description": "Règle du produit",
                "legende": "u, v = fonctions dérivables • u', v' = dérivées respectives"
            },
            {
                "titre": "Dérivée quotient",
                "formule": "(u/v)' = (u'v - uv') / v²",
                "description": "Quotient de fonctions",
                "legende": "u, v = fonctions dérivables • v ≠ 0 • u', v' = dérivées respectives"
            },
            {
                "titre": "Dérivée √x",
                "formule": "(√x)' = 1/(2√x)",
                "description": "Racine carrée",
                "legende": "x = variable (x > 0) • √x = racine carrée de x"
            },
            {
                "titre": "Dérivée 1/x",
                "formule": "(1/x)' = -1/x²",
                "description": "Fonction inverse",
                "legende": "x = variable (x ≠ 0)"
            },
        ],
        
        "🔢 Second degré": [
            {
                "titre": "Forme canonique",
                "formule": "ax² + bx + c = a(x - α)² + β",
                "description": "Forme développée → forme canonique",
                "legende": "a, b, c = coefficients de l'équation • α = -b/(2a) = sommet abscisse • β = f(α) = sommet ordonnée"
            },
            {
                "titre": "Discriminant",
                "formule": "Δ = b² - 4ac",
                "description": "Détermine le nombre de racines",
                "legende": "Δ = discriminant • a, b, c = coefficients de ax² + bx + c • Δ > 0 : 2 racines • Δ = 0 : 1 racine • Δ < 0 : 0 racine"
            },
            {
                "titre": "Racines (Δ > 0)",
                "formule": "x₁ = (-b - √Δ)/(2a)\nx₂ = (-b + √Δ)/(2a)",
                "description": "Deux solutions distinctes",
                "legende": "x₁, x₂ = racines • a, b = coefficients • Δ = discriminant • √Δ = racine carrée du discriminant"
            },
            {
                "titre": "Racine double (Δ = 0)",
                "formule": "x₀ = -b/(2a)",
                "description": "Une solution double",
                "legende": "x₀ = racine double • a, b = coefficients • Cette racine est aussi l'abscisse du sommet"
            },
            {
                "titre": "Somme et produit",
                "formule": "S = x₁ + x₂ = -b/a\nP = x₁ × x₂ = c/a",
                "description": "Relations coefficients-racines",
                "legende": "S = somme des racines • P = produit des racines • x₁, x₂ = racines • a, b, c = coefficients"
            },
        ],
        
        "📊 Probabilités": [
            {
                "titre": "Probabilité événement",
                "formule": "P(A) = nombre de cas favorables / nombre de cas possibles",
                "description": "Loi de Laplace (équiprobabilité)",
                "legende": "P(A) = probabilité de l'événement A (entre 0 et 1) • A = événement"
            },
            {
                "titre": "Événement contraire",
                "formule": "P(Ā) = 1 - P(A)",
                "description": "Complémentaire",
                "legende": "Ā = événement contraire de A (non-A) • P(Ā) = probabilité du contraire"
            },
            {
                "titre": "Réunion",
                "formule": "P(A ∪ B) = P(A) + P(B) - P(A ∩ B)",
                "description": "A ou B (au moins un)",
                "legende": "A ∪ B = réunion (A ou B) • A ∩ B = intersection (A et B) • Si incompatibles : P(A ∩ B) = 0"
            },
            {
                "titre": "Intersection",
                "formule": "P(A ∩ B) = P(A) × P(B)",
                "description": "Événements indépendants",
                "legende": "A ∩ B = intersection (A et B) • Valable seulement si A et B sont indépendants"
            },
            {
                "titre": "Probabilité conditionnelle",
                "formule": "P_A(B) = P(A ∩ B) / P(A)",
                "description": "Probabilité de B sachant A",
                "legende": "P_A(B) = probabilité de B sachant que A est réalisé • P(A) ≠ 0"
            },
        ],
        
        "📐 Produit scalaire": [
            {
                "titre": "Définition",
                "formule": "u⃗ · v⃗ = ||u⃗|| × ||v⃗|| × cos(θ)",
                "description": "Définition géométrique",
                "legende": "u⃗, v⃗ = vecteurs • ||u⃗||, ||v⃗|| = normes (longueurs) • θ = angle entre les vecteurs • cos(θ) = cosinus de l'angle"
            },
            {
                "titre": "Coordonnées",
                "formule": "u⃗ · v⃗ = xx' + yy'",
                "description": "Calcul avec coordonnées",
                "legende": "u⃗(x, y) = vecteur u de coordonnées (x, y) • v⃗(x', y') = vecteur v de coordonnées (x', y')"
            },
            {
                "titre": "Propriétés",
                "formule": "u⃗ · v⃗ = v⃗ · u⃗\n(ku⃗) · v⃗ = k(u⃗ · v⃗)",
                "description": "Commutativité et linéarité",
                "legende": "u⃗, v⃗ = vecteurs • k = nombre réel (scalaire)"
            },
            {
                "titre": "Orthogonalité",
                "formule": "u⃗ · v⃗ = 0 ⟺ u⃗ ⊥ v⃗",
                "description": "Vecteurs perpendiculaires",
                "legende": "u⃗ ⊥ v⃗ = u et v sont perpendiculaires • ⟺ = équivalence (si et seulement si)"
            },
        ],
    },
    
    "Terminale": {
        "∞ Limites": [
            {
                "titre": "Limite somme",
                "formule": "lim(u + v) = lim(u) + lim(v)",
                "description": "Somme de limites",
                "legende": "u, v = fonctions ou suites • lim = limite quand x→a ou n→+∞ • Valable si les limites existent"
            },
            {
                "titre": "Limite produit",
                "formule": "lim(u × v) = lim(u) × lim(v)",
                "description": "Produit de limites",
                "legende": "u, v = fonctions ou suites • lim = limite quand x→a ou n→+∞ • Valable si les limites existent"
            },
            {
                "titre": "Limite quotient",
                "formule": "lim(u/v) = lim(u) / lim(v)",
                "description": "Quotient de limites",
                "legende": "u, v = fonctions ou suites • lim = limite quand x→a ou n→+∞ • Valable si lim(v) ≠ 0"
            },
            {
                "titre": "Formes indéterminées",
                "formule": "∞ - ∞, 0 × ∞, ∞/∞, 0/0",
                "description": "Nécessitent une étude particulière",
                "legende": "∞ = infini • 0 = zéro • Ces formes ne permettent pas de conclure directement"
            },
            {
                "titre": "Croissances comparées",
                "formule": "lim(eˣ/xⁿ) = +∞\nlim(ln(x)/x) = 0",
                "description": "Quand x → +∞",
                "legende": "e = exponentielle • ln = logarithme népérien • x = variable • n = puissance quelconque • e croît plus vite que toute puissance"
            },
            {
                "titre": "Limites usuelles",
                "formule": "lim(sin(x)/x) = 1\nlim((1+1/x)ˣ) = e",
                "description": "Limites remarquables",
                "legende": "x→0 pour sin(x)/x • x→+∞ pour (1+1/x)ˣ • e ≈ 2,71828 • sin = sinus"
            },
        ],
        
        "∫ Intégrales": [
            {
                "titre": "Primitive puissance",
                "formule": "∫xⁿ dx = xⁿ⁺¹/(n+1) + K",
                "description": "Puissance de x",
                "legende": "x = variable • n = exposant (n ≠ -1) • K = constante d'intégration • ∫ = intégrale"
            },
            {
                "titre": "Primitive 1/x",
                "formule": "∫(1/x) dx = ln|x| + K",
                "description": "Fonction inverse",
                "legende": "x = variable (x ≠ 0) • ln = logarithme népérien • |x| = valeur absolue • K = constante"
            },
            {
                "titre": "Primitive exponentielle",
                "formule": "∫eˣ dx = eˣ + K",
                "description": "Fonction invariante",
                "legende": "e = exponentielle (e ≈ 2,71828) • x = variable • K = constante • eˣ est sa propre primitive"
            },
            {
                "titre": "Primitive cos",
                "formule": "∫cos(x) dx = sin(x) + K",
                "description": "Cosinus",
                "legende": "x = angle (en radians) • cos = cosinus • sin = sinus • K = constante"
            },
            {
                "titre": "Primitive sin",
                "formule": "∫sin(x) dx = -cos(x) + K",
                "description": "Sinus",
                "legende": "x = angle (en radians) • sin = sinus • cos = cosinus • K = constante • Attention au signe -"
            },
            {
                "titre": "Intégrale définie",
                "formule": "∫ₐᵇ f(x)dx = F(b) - F(a)",
                "description": "Théorème fondamental",
                "legende": "a, b = bornes d'intégration • f = fonction • F = primitive de f • F(b) - F(a) = aire algébrique"
            },
            {
                "titre": "Linéarité",
                "formule": "∫(u + v) = ∫u + ∫v\n∫ku = k∫u",
                "description": "Propriétés",
                "legende": "u, v = fonctions • k = constante réelle • ∫ = symbole d'intégrale"
            },
            {
                "titre": "Relation Chasles",
                "formule": "∫ₐᵇ + ∫ᵇᶜ = ∫ₐᶜ",
                "description": "Découpage d'intervalle",
                "legende": "a, b, c = bornes d'intégration (a < b < c) • Permet de découper une intégrale"
            },
            {
                "titre": "Valeur moyenne",
                "formule": "μ = 1/(b-a) × ∫ₐᵇ f(x)dx",
                "description": "Moyenne d'une fonction",
                "legende": "μ = valeur moyenne • a, b = bornes • f = fonction • b-a = longueur de l'intervalle"
            },
        ],
        
        "📈 Fonction exponentielle": [
            {
                "titre": "Définition",
                "formule": "exp(x) = eˣ",
                "description": "Notation exponentielle",
                "legende": "e ≈ 2,71828 (nombre d'Euler) • x = variable • exp = fonction exponentielle"
            },
            {
                "titre": "Propriété fondamentale",
                "formule": "(eˣ)' = eˣ",
                "description": "Dérivée égale à elle-même",
                "legende": "e = exponentielle • x = variable • ' = symbole de dérivée • Unique fonction ayant cette propriété"
            },
            {
                "titre": "Propriétés algébriques",
                "formule": "eᵃ × eᵇ = eᵃ⁺ᵇ\neᵃ / eᵇ = eᵃ⁻ᵇ\n(eᵃ)ᵇ = eᵃˣᵇ",
                "description": "Règles de calcul",
                "legende": "e = exponentielle • a, b = exposants (nombres réels) • Permet de simplifier les calculs"
            },
            {
                "titre": "Valeurs particulières",
                "formule": "e⁰ = 1\ne¹ = e\neˡⁿ⁽ˣ⁾ = x",
                "description": "Valeurs remarquables",
                "legende": "e ≈ 2,71828 • ln = logarithme népérien • x > 0 • e et ln sont réciproques"
            },
            {
                "titre": "Limites",
                "formule": "lim(eˣ) = +∞ (x→+∞)\nlim(eˣ) = 0 (x→-∞)",
                "description": "Comportement asymptotique",
                "legende": "x = variable • +∞ = plus l'infini • 0 = zéro • Croissance exponentielle vers +∞"
            },
        ],
        
        "📊 Fonction logarithme": [
            {
                "titre": "Définition",
                "formule": "ln(x) = log_e(x)",
                "description": "Logarithme népérien",
                "legende": "ln = logarithme népérien • log_e = logarithme en base e • x > 0 • Réciproque de exp"
            },
            {
                "titre": "Dérivée",
                "formule": "(ln(x))' = 1/x",
                "description": "Dérivée du logarithme",
                "legende": "ln = logarithme népérien • x = variable (x > 0) • ' = symbole de dérivée"
            },
            {
                "titre": "Propriétés algébriques",
                "formule": "ln(ab) = ln(a) + ln(b)\nln(a/b) = ln(a) - ln(b)\nln(aⁿ) = n×ln(a)",
                "description": "Règles de calcul",
                "legende": "ln = logarithme népérien • a, b > 0 • n = exposant réel • Transforme produits en sommes"
            },
            {
                "titre": "Valeurs particulières",
                "formule": "ln(1) = 0\nln(e) = 1\nln(eˣ) = x",
                "description": "Valeurs remarquables",
                "legende": "ln = logarithme népérien • e ≈ 2,71828 • x = variable • e et ln sont réciproques"
            },
            {
                "titre": "Limites",
                "formule": "lim(ln(x)) = +∞ (x→+∞)\nlim(ln(x)) = -∞ (x→0⁺)",
                "description": "Comportement asymptotique",
                "legende": "x = variable • +∞ = plus l'infini • -∞ = moins l'infini • 0⁺ = zéro par valeurs positives"
            },
        ],
        
        "🎲 Lois de probabilité": [
            {
                "titre": "Loi binomiale",
                "formule": "P(X = k) = C_n^k × p^k × (1-p)^(n-k)",
                "description": "Répétitions d'expériences",
                "legende": "X = variable aléatoire • k = nombre de succès • n = nombre d'essais • p = probabilité de succès • C_n^k = coefficient binomial"
            },
            {
                "titre": "Espérance binomiale",
                "formule": "E(X) = np",
                "description": "Valeur moyenne attendue",
                "legende": "E(X) = espérance (moyenne) • n = nombre d'essais • p = probabilité de succès"
            },
            {
                "titre": "Variance binomiale",
                "formule": "V(X) = np(1-p)",
                "description": "Mesure de dispersion",
                "legende": "V(X) = variance • n = nombre d'essais • p = probabilité de succès • 1-p = probabilité d'échec"
            },
            {
                "titre": "Écart-type",
                "formule": "σ(X) = √V(X)",
                "description": "Dispersion autour de la moyenne",
                "legende": "σ = écart-type (sigma) • V(X) = variance • √ = racine carrée • Même unité que X"
            },
            {
                "titre": "Loi normale",
                "formule": "X ~ N(μ, σ²)",
                "description": "Courbe en cloche",
                "legende": "X = variable aléatoire • μ = moyenne (mu) • σ² = variance • σ = écart-type • ~ = suit la loi"
            },
            {
                "titre": "Variable centrée réduite",
                "formule": "Z = (X - μ) / σ",
                "description": "Standardisation",
                "legende": "Z = variable centrée réduite • X = variable initiale • μ = moyenne • σ = écart-type • Z ~ N(0,1)"
            },
            {
                "titre": "Théorème central limite",
                "formule": "X̄ ~ N(μ, σ²/n)",
                "description": "Moyenne de n variables",
                "legende": "X̄ = moyenne de n variables • μ = moyenne théorique • σ² = variance • n = nombre de variables"
            },
        ],
        
        "📐 Géométrie dans l'espace": [
            {
                "titre": "Produit scalaire",
                "formule": "u⃗ · v⃗ = xx' + yy' + zz'",
                "description": "Coordonnées en 3D",
                "legende": "u⃗(x, y, z) = vecteur u • v⃗(x', y', z') = vecteur v • · = produit scalaire"
            },
            {
                "titre": "Norme vecteur",
                "formule": "||u⃗|| = √(x² + y² + z²)",
                "description": "Longueur dans l'espace",
                "legende": "||u⃗|| = norme (longueur) • u⃗(x, y, z) = vecteur • √ = racine carrée"
            },
            {
                "titre": "Distance points",
                "formule": "AB = √((xB-xA)² + (yB-yA)² + (zB-zA)²)",
                "description": "Distance en 3D",
                "legende": "A(xA, yA, zA) = point A • B(xB, yB, zB) = point B • AB = distance entre A et B"
            },
            {
                "titre": "Équation plan",
                "formule": "ax + by + cz + d = 0",
                "description": "Plan dans l'espace",
                "legende": "a, b, c = coordonnées du vecteur normal n⃗(a,b,c) • d = constante • x, y, z = coordonnées d'un point du plan"
            },
            {
                "titre": "Équation droite paramétrique",
                "formule": "x = x₀ + at\ny = y₀ + bt\nz = z₀ + ct",
                "description": "Droite dans l'espace",
                "legende": "M₀(x₀, y₀, z₀) = point de la droite • u⃗(a, b, c) = vecteur directeur • t = paramètre réel"
            },
            {
                "titre": "Volume pyramide",
                "formule": "V = (1/3) × Aire_base × hauteur",
                "description": "Volume d'une pyramide",
                "legende": "V = volume • Aire_base = aire de la base • hauteur = distance perpendiculaire sommet-base"
            },
            {
                "titre": "Volume sphère",
                "formule": "V = (4/3)πr³",
                "description": "Volume d'une boule",
                "legende": "V = volume • r = rayon • π ≈ 3,14159 • 4/3 = coefficient"
            },
            {
                "titre": "Aire sphère",
                "formule": "A = 4πr²",
                "description": "Surface d'une sphère",
                "legende": "A = aire • r = rayon • π ≈ 3,14159 • 4πr² = quatre fois l'aire d'un disque"
            },
        ],
        
        "🔢 Suites": [
            {
                "titre": "Limite suite",
                "formule": "∀ε > 0, ∃N, ∀n > N : |uₙ - l| < ε",
                "description": "Définition formelle",
                "legende": "∀ = pour tout • ∃ = il existe • ε = epsilon (petit nombre > 0) • N = rang • n = indice • l = limite"
            },
            {
                "titre": "Suite convergente",
                "formule": "lim(uₙ) = l",
                "description": "Converge vers l",
                "legende": "uₙ = terme de rang n • l = limite finie • n→+∞ = quand n tend vers l'infini"
            },
            {
                "titre": "Suite divergente",
                "formule": "lim(uₙ) = ±∞ ou n'existe pas",
                "description": "Ne converge pas",
                "legende": "uₙ = terme de rang n • +∞ = plus l'infini • -∞ = moins l'infini • Pas de limite finie"
            },
            {
                "titre": "Théorème encadrement",
                "formule": "Si uₙ ≤ vₙ ≤ wₙ et lim(uₙ) = lim(wₙ) = l\nalors lim(vₙ) = l",
                "description": "Théorème des gendarmes",
                "legende": "uₙ, vₙ, wₙ = suites • l = limite commune • vₙ est 'coincée' entre uₙ et wₙ"
            },
            {
                "titre": "Suite géométrique",
                "formule": "Si |q| < 1 : lim(qⁿ) = 0\nSi q > 1 : lim(qⁿ) = +∞",
                "description": "Comportement selon la raison",
                "legende": "q = raison • n = rang • |q| = valeur absolue • Si |q| = 1 : pas de limite sauf si q=1"
            },
        ],
        
        "📊 Convexité": [
            {
                "titre": "Fonction convexe",
                "formule": "f''(x) ≥ 0",
                "description": "Courbe en forme de U",
                "legende": "f = fonction deux fois dérivable • f'' = dérivée seconde • x = variable • ≥ 0 = positive ou nulle"
            },
            {
                "titre": "Fonction concave",
                "formule": "f''(x) ≤ 0",
                "description": "Courbe en forme de ∩",
                "legende": "f = fonction deux fois dérivable • f'' = dérivée seconde • x = variable • ≤ 0 = négative ou nulle"
            },
            {
                "titre": "Point d'inflexion",
                "formule": "f''(x) = 0 et f'' change de signe",
                "description": "Changement de courbure",
                "legende": "f'' = dérivée seconde • x = point d'inflexion • Change de convexe à concave ou inversement"
            },
            {
                "titre": "Inégalité convexité",
                "formule": "f(λa + (1-λ)b) ≤ λf(a) + (1-λ)f(b)",
                "description": "Définition alternative",
                "legende": "f = fonction convexe • a, b = points • λ = coefficient (λ ∈ [0,1]) • Corde au-dessus de la courbe"
            },
        ],
    }
}


def render_formula_card(formule_data: dict, level: str):
    """Version alternative utilisant moins de HTML complexe"""
    
    titre = formule_data['titre']
    formule = formule_data['formule']
    description = formule_data['description']
    legende = formule_data.get('legende', '')
    
    # Version ultra-simplifiée
    st.markdown(
        f'<div style="background:white;border:2px solid #E8E8E8;border-radius:16px;padding:20px;margin-bottom:20px;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:10px;">'
        f'<span style="color:#FF6B35;font-weight:700;">{_html.escape(titre)}</span>'
        f'<span style="background:#FF6B35;color:white;padding:4px 8px;border-radius:10px;font-size:10px;">{_html.escape(level)}</span>'
        f'</div>'
        f'<div style="background:#F8F9FA;padding:12px;border-left:4px solid #FF6B35;margin:10px 0;">'
        f'<pre style="margin:0;font-family:monospace;font-size:13px;">{_html.escape(formule)}</pre>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Légende en composant natif Streamlit
    if legende:
        st.info(f"📝 {legende}")

    # Description
    st.markdown(
        f'<div style="color:#6C757D;font-size:12px;font-style:italic;">{_html.escape(description)}</div>'
        f'</div>',
        unsafe_allow_html=True
    )



# ============================================================
# TABS
# ============================================================

def render_tabs():
    """Affiche les onglets principaux"""
    tab1, tab2 = st.tabs(["📖 Cours & Références", "✏️ Exercices"])
    
    with tab1:
        render_cours_tab()
    
    with tab2:
        render_exercices_tab()



"""
BASE DE DONNÉES COMPLÈTE DES THÉORÈMES ET PROPRIÉTÉS
Par niveau (Seconde, Première, Terminale) et par chapitre
"""


THEOREMES_PROPRIETES = {
    "Seconde": {
        "📐 Géométrie": [
            {
                "titre": "Théorème de Pythagore",
                "type": "Théorème",
                "enonce": "Dans un triangle rectangle, le carré de l'hypoténuse est égal à la somme des carrés des deux autres côtés",
                "hypothese": "ABC triangle rectangle en A",
                "conclusion": "BC² = AB² + AC²",
                "legende": "A = angle droit • AB, AC = côtés de l'angle droit • BC = hypoténuse"
            },
            {
                "titre": "Réciproque de Pythagore",
                "type": "Théorème",
                "enonce": "Si dans un triangle le carré du plus grand côté est égal à la somme des carrés des deux autres côtés, alors ce triangle est rectangle",
                "hypothese": "ABC triangle avec BC² = AB² + AC²",
                "conclusion": "ABC est rectangle en A",
                "legende": "BC = plus grand côté • A = sommet de l'angle droit"
            },
            {
                "titre": "Théorème de Thalès",
                "type": "Théorème",
                "enonce": "Si deux droites sont parallèles, alors elles déterminent sur deux sécantes des segments proportionnels",
                "hypothese": "(BC) // (B'C') dans le triangle AA'C",
                "conclusion": "AB/AB' = AC/AC' = BC/B'C'",
                "legende": "A, B, C = points du grand triangle • B', C' = points du petit triangle • // = parallèle"
            },
            {
                "titre": "Réciproque de Thalès",
                "type": "Théorème",
                "enonce": "Si les rapports de longueurs sont égaux, alors les droites sont parallèles",
                "hypothese": "AB/AB' = AC/AC' et les points alignés dans le même ordre",
                "conclusion": "(BC) // (B'C')",
                "legende": "A, B, C = points • Points alignés dans le même ordre (essentiel)"
            },
            {
                "titre": "Propriété des triangles isocèles",
                "type": "Propriété",
                "enonce": "Dans un triangle isocèle, les angles à la base sont égaux",
                "hypothese": "ABC isocèle en A (AB = AC)",
                "conclusion": "Angle ABC = Angle ACB",
                "legende": "A = sommet principal • AB = AC = côtés égaux • B, C = base"
            },
            {
                "titre": "Somme des angles d'un triangle",
                "type": "Propriété",
                "enonce": "La somme des angles d'un triangle vaut 180°",
                "hypothese": "ABC triangle quelconque",
                "conclusion": "Angle A + Angle B + Angle C = 180°",
                "legende": "A, B, C = sommets du triangle • 180° = π radians"
            },
            {
                "titre": "Propriété de la médiatrice",
                "type": "Propriété",
                "enonce": "Tout point de la médiatrice d'un segment est équidistant des extrémités de ce segment",
                "hypothese": "M sur la médiatrice de [AB]",
                "conclusion": "MA = MB",
                "legende": "M = point quelconque de la médiatrice • A, B = extrémités • Médiatrice = perpendiculaire au milieu"
            },
            {
                "titre": "Angles opposés par le sommet",
                "type": "Propriété",
                "enonce": "Deux angles opposés par le sommet sont égaux",
                "hypothese": "Deux droites sécantes",
                "conclusion": "Les angles opposés sont égaux",
                "legende": "Sommet = point d'intersection des droites"
            },
        ],
        
        "🔢 Calcul et nombres": [
            {
                "titre": "Propriété de la racine carrée",
                "type": "Propriété",
                "enonce": "Le carré de la racine carrée d'un nombre positif donne ce nombre",
                "hypothese": "a ≥ 0",
                "conclusion": "(√a)² = a",
                "legende": "a = nombre positif ou nul • √a = racine carrée de a"
            },
            {
                "titre": "Racine d'un produit",
                "type": "Propriété",
                "enonce": "La racine d'un produit est le produit des racines",
                "hypothese": "a ≥ 0 et b ≥ 0",
                "conclusion": "√(a × b) = √a × √b",
                "legende": "a, b = nombres positifs ou nuls"
            },
            {
                "titre": "Racine d'un quotient",
                "type": "Propriété",
                "enonce": "La racine d'un quotient est le quotient des racines",
                "hypothese": "a ≥ 0 et b > 0",
                "conclusion": "√(a/b) = √a / √b",
                "legende": "a = nombre positif ou nul • b = nombre strictement positif"
            },
            {
                "titre": "Puissances de même base",
                "type": "Propriété",
                "enonce": "Pour multiplier des puissances de même base, on additionne les exposants",
                "hypothese": "a ≠ 0, m et n entiers",
                "conclusion": "aᵐ × aⁿ = aᵐ⁺ⁿ",
                "legende": "a = base • m, n = exposants"
            },
            {
                "titre": "Puissance d'un produit",
                "type": "Propriété",
                "enonce": "La puissance d'un produit est le produit des puissances",
                "hypothese": "a ≠ 0, b ≠ 0, n entier",
                "conclusion": "(a × b)ⁿ = aⁿ × bⁿ",
                "legende": "a, b = bases non nulles • n = exposant"
            },
        ],
        
        "📊 Fonctions": [
            {
                "titre": "Sens de variation et signe de la dérivée",
                "type": "Propriété",
                "enonce": "Une fonction est croissante si et seulement si le taux de variation est positif",
                "hypothese": "f fonction sur un intervalle I",
                "conclusion": "f croissante ⟺ pour tout a < b : f(b) > f(a)",
                "legende": "f = fonction • I = intervalle • a, b = points de I • ⟺ = équivalence"
            },
            {
                "titre": "Image d'un intervalle par fonction croissante",
                "type": "Propriété",
                "enonce": "L'image d'un intervalle par une fonction croissante est un intervalle",
                "hypothese": "f croissante sur [a;b]",
                "conclusion": "f([a;b]) = [f(a);f(b)]",
                "legende": "f = fonction croissante • [a;b] = intervalle • f([a;b]) = ensemble des images"
            },
            {
                "titre": "Tableau de signes",
                "type": "Propriété",
                "enonce": "Le signe d'un produit dépend du nombre de facteurs négatifs",
                "hypothese": "Produit de facteurs",
                "conclusion": "Nombre pair de facteurs négatifs → produit positif",
                "legende": "Facteurs = nombres multipliés • Pair = 0, 2, 4... • Impair = 1, 3, 5..."
            },
        ],
        
        "📈 Statistiques": [
            {
                "titre": "Propriété de la médiane",
                "type": "Propriété",
                "enonce": "La médiane partage une série statistique en deux groupes de même effectif",
                "hypothese": "Série statistique ordonnée",
                "conclusion": "50% des valeurs ≤ médiane et 50% ≥ médiane",
                "legende": "Médiane = valeur centrale • Effectif = nombre de données"
            },
            {
                "titre": "Moyenne pondérée",
                "type": "Propriété",
                "enonce": "La moyenne d'une série pondérée se calcule en divisant la somme des produits valeur×effectif par l'effectif total",
                "hypothese": "Série avec effectifs",
                "conclusion": "x̄ = Σ(xᵢ × nᵢ) / N",
                "legende": "x̄ = moyenne • xᵢ = valeurs • nᵢ = effectifs • N = effectif total • Σ = somme"
            },
        ],
        
        # ✅ NOUVEAU CHAPITRE
        "📐 Vecteurs": [
            {
                "titre": "Relation de Chasles",
                "type": "Propriété",
                "enonce": "Pour tous points A, B et C, la somme des vecteurs AB et BC est égale au vecteur AC",
                "hypothese": "A, B, C points quelconques du plan",
                "conclusion": "AB⃗ + BC⃗ = AC⃗",
                "legende": "AB⃗ = vecteur de A vers B • Valable même si les points sont alignés ou confondus"
            },
            {
                "titre": "Règle du parallélogramme",
                "type": "Propriété",
                "enonce": "ABCD est un parallélogramme si et seulement si les vecteurs AB et DC sont égaux",
                "hypothese": "ABCD quadrilatère",
                "conclusion": "ABCD parallélogramme ⟺ AB⃗ = DC⃗",
                "legende": "⟺ = équivalence • Aussi équivalent à : AD⃗ = BC⃗ ou AC⃗ = AB⃗ + AD⃗"
            },
            {
                "titre": "Colinéarité de vecteurs",
                "type": "Propriété",
                "enonce": "Deux vecteurs sont colinéaires si et seulement si l'un est un multiple de l'autre",
                "hypothese": "u⃗ et v⃗ vecteurs non nuls",
                "conclusion": "u⃗ et v⃗ colinéaires ⟺ ∃k ∈ ℝ : u⃗ = kv⃗",
                "legende": "k = coefficient de colinéarité • ∃ = il existe • ℝ = ensemble des réels • Vecteurs de même direction"
            },
            {
                "titre": "Théorème du milieu",
                "type": "Théorème",
                "enonce": "I est le milieu de [AB] si et seulement si le vecteur AI est égal au vecteur IB",
                "hypothese": "A, B points distincts, I point du segment [AB]",
                "conclusion": "I milieu de [AB] ⟺ AI⃗ = IB⃗",
                "legende": "Aussi équivalent à : OI⃗ = (OA⃗ + OB⃗)/2 pour tout point O"
            },
            {
                "titre": "Coordonnées du milieu",
                "type": "Propriété",
                "enonce": "Les coordonnées du milieu d'un segment sont les moyennes des coordonnées des extrémités",
                "hypothese": "A(xA, yA) et B(xB, yB) points du plan",
                "conclusion": "I milieu de [AB] ⟹ I((xA+xB)/2, (yA+yB)/2)",
                "legende": "xI = (xA + xB)/2 • yI = (yA + yB)/2 • Formule du point milieu"
            },
            {
                "titre": "Norme d'un vecteur",
                "type": "Propriété",
                "enonce": "La norme d'un vecteur de coordonnées (x, y) est la racine carrée de la somme des carrés de ses coordonnées",
                "hypothese": "u⃗(x, y) vecteur du plan",
                "conclusion": "||u⃗|| = √(x² + y²)",
                "legende": "||u⃗|| = norme (longueur) du vecteur • √ = racine carrée • Formule de Pythagore"
            },
            {
                "titre": "Vecteurs et translation",
                "type": "Propriété",
                "enonce": "Deux vecteurs sont égaux si et seulement si ils ont même direction, même sens et même longueur",
                "hypothese": "AB⃗ et CD⃗ vecteurs",
                "conclusion": "AB⃗ = CD⃗ ⟺ ABDC est un parallélogramme (éventuellement aplati)",
                "legende": "Translation = déplacement sans rotation • Vecteurs égaux = parallélogramme"
            },
            {
                "titre": "Somme de vecteurs - coordonnées",
                "type": "Propriété",
                "enonce": "Les coordonnées de la somme de deux vecteurs sont les sommes des coordonnées",
                "hypothese": "u⃗(x, y) et v⃗(x', y') vecteurs",
                "conclusion": "u⃗ + v⃗ = (x + x', y + y')",
                "legende": "Addition coordonnée par coordonnée • Règle du parallélogramme en coordonnées"
            },
            {
                "titre": "Produit d'un vecteur par un réel",
                "type": "Propriété",
                "enonce": "Les coordonnées du produit d'un vecteur par un réel sont les coordonnées multipliées par ce réel",
                "hypothese": "u⃗(x, y) vecteur et k réel",
                "conclusion": "ku⃗ = (kx, ky)",
                "legende": "k > 0 : même sens • k < 0 : sens opposé • |k| = facteur d'agrandissement • Multiplication externe"
            },
            {
                "titre": "Distance et vecteurs",
                "type": "Propriété",
                "enonce": "La distance entre deux points est égale à la norme du vecteur qui les relie",
                "hypothese": "A et B points du plan",
                "conclusion": "AB = ||AB⃗||",
                "legende": "AB = distance entre A et B • ||AB⃗|| = norme du vecteur AB⃗ • Lien géométrie-algèbre"
            },
        ],
        
        # ✅ NOUVEAU CHAPITRE
        "🔢 Équations et inéquations": [
            {
                "titre": "Propriété d'addition dans les équations",
                "type": "Propriété",
                "enonce": "On peut ajouter ou soustraire un même nombre aux deux membres d'une équation",
                "hypothese": "a = b équation, c nombre réel",
                "conclusion": "a = b ⟺ a + c = b + c ⟺ a - c = b - c",
                "legende": "⟺ = équivalence (solutions identiques) • Opération réversible"
            },
            {
                "titre": "Propriété de multiplication dans les équations",
                "type": "Propriété",
                "enonce": "On peut multiplier ou diviser les deux membres d'une équation par un même nombre non nul",
                "hypothese": "a = b équation, c nombre réel non nul",
                "conclusion": "a = b ⟺ a × c = b × c ⟺ a/c = b/c",
                "legende": "c ≠ 0 indispensable pour la division • Multiplication par 0 perd l'équivalence"
            },
            {
                "titre": "Propriété d'addition dans les inéquations",
                "type": "Propriété",
                "enonce": "On peut ajouter un même nombre aux deux membres d'une inéquation sans changer le sens",
                "hypothese": "a < b inéquation, c nombre réel",
                "conclusion": "a < b ⟺ a + c < b + c",
                "legende": "Valable pour <, >, ≤, ≥ • Le sens de l'inégalité ne change pas"
            },
            {
                "titre": "Propriété de multiplication dans les inéquations",
                "type": "Propriété",
                "enonce": "En multipliant par un nombre positif le sens ne change pas, par un nombre négatif le sens s'inverse",
                "hypothese": "a < b inéquation, c nombre réel non nul",
                "conclusion": "Si c > 0 : a < b ⟺ ac < bc • Si c < 0 : a < b ⟺ ac > bc",
                "legende": "c > 0 : sens conservé • c < 0 : sens inversé • ATTENTION au changement de sens !"
            },
            {
                "titre": "Tableau de signes d'un produit",
                "type": "Propriété",
                "enonce": "Le signe d'un produit dépend du nombre de facteurs négatifs",
                "hypothese": "P = a × b × c produit de facteurs",
                "conclusion": "Nombre pair de facteurs négatifs → P ≥ 0 • Nombre impair → P ≤ 0",
                "legende": "Pair = 0, 2, 4... • Impair = 1, 3, 5... • Règle des signes"
            },
        ],
    },
    
    "Première": {
        "📈 Dérivation": [
            {
                "titre": "Théorème de dérivabilité",
                "type": "Théorème",
                "enonce": "Si une fonction est dérivable en a, alors elle est continue en a",
                "hypothese": "f dérivable en a",
                "conclusion": "f continue en a",
                "legende": "f = fonction • a = point • Continue = pas de rupture • Réciproque FAUSSE"
            },
            {
                "titre": "Lien dérivée et variations",
                "type": "Théorème",
                "enonce": "Si f'(x) > 0 sur un intervalle, alors f est strictement croissante sur cet intervalle",
                "hypothese": "f dérivable sur I, f'(x) > 0",
                "conclusion": "f strictement croissante sur I",
                "legende": "f' = dérivée de f • I = intervalle • > 0 = strictement positive"
            },
            {
                "titre": "Extremum et dérivée",
                "type": "Théorème",
                "enonce": "Si f admet un extremum en a et est dérivable en a, alors f'(a) = 0",
                "hypothese": "f dérivable en a, f admet extremum en a",
                "conclusion": "f'(a) = 0",
                "legende": "a = point d'extremum • Extremum = maximum ou minimum local • Réciproque FAUSSE"
            },
            {
                "titre": "Dérivée de la composée",
                "type": "Propriété",
                "enonce": "La dérivée de u∘v est u'(v) × v'",
                "hypothese": "u et v dérivables",
                "conclusion": "(u∘v)' = u'(v) × v'",
                "legende": "u∘v = fonction composée (u de v) • u'(v) = dérivée de u évaluée en v"
            },
        ],
        
        "📊 Suites numériques": [
            {
                "titre": "Propriété suite arithmétique",
                "type": "Propriété",
                "enonce": "Une suite est arithmétique si et seulement si la différence entre deux termes consécutifs est constante",
                "hypothese": "Suite (uₙ)",
                "conclusion": "uₙ arithmétique ⟺ uₙ₊₁ - uₙ = r (constante)",
                "legende": "uₙ = terme de rang n • r = raison • Constante = même valeur pour tout n"
            },
            {
                "titre": "Propriété suite géométrique",
                "type": "Propriété",
                "enonce": "Une suite est géométrique si et seulement si le quotient de deux termes consécutifs est constant",
                "hypothese": "Suite (uₙ) avec uₙ ≠ 0",
                "conclusion": "uₙ géométrique ⟺ uₙ₊₁/uₙ = q (constante)",
                "legende": "uₙ = terme de rang n (≠ 0) • q = raison • Constante = même valeur pour tout n"
            },
            {
                "titre": "Limite suite arithmétique",
                "type": "Théorème",
                "enonce": "Une suite arithmétique de raison non nulle tend vers l'infini",
                "hypothese": "uₙ arithmétique de raison r ≠ 0",
                "conclusion": "Si r > 0 : lim(uₙ) = +∞ • Si r < 0 : lim(uₙ) = -∞",
                "legende": "r = raison • > 0 = positif • < 0 = négatif"
            },
            {
                "titre": "Limite suite géométrique",
                "type": "Théorème",
                "enonce": "Le comportement d'une suite géométrique dépend de sa raison",
                "hypothese": "uₙ géométrique de raison q",
                "conclusion": "Si |q| < 1 : lim = 0 • Si q > 1 : lim = ±∞ • Si q ≤ -1 : pas de limite",
                "legende": "q = raison • |q| = valeur absolue • Comportement différent selon q"
            },
        ],
        
        "📐 Trigonométrie": [
            {
                "titre": "Formule fondamentale",
                "type": "Propriété",
                "enonce": "Pour tout angle x, le carré du cosinus plus le carré du sinus vaut 1",
                "hypothese": "x angle quelconque",
                "conclusion": "cos²(x) + sin²(x) = 1",
                "legende": "x = angle en radians • cos = cosinus • sin = sinus • Toujours vrai"
            },
            {
                "titre": "Propriété tangente",
                "type": "Propriété",
                "enonce": "La tangente est le quotient du sinus par le cosinus",
                "hypothese": "x tel que cos(x) ≠ 0",
                "conclusion": "tan(x) = sin(x)/cos(x)",
                "legende": "tan = tangente • Définie seulement si cos(x) ≠ 0"
            },
            {
                "titre": "Angles associés - opposé",
                "type": "Propriété",
                "enonce": "Le cosinus est pair et le sinus est impair",
                "hypothese": "x angle quelconque",
                "conclusion": "cos(-x) = cos(x) • sin(-x) = -sin(x)",
                "legende": "Pair = symétrique par rapport à l'axe • Impair = antisymétrique"
            },
            {
                "titre": "Angles associés - supplémentaire",
                "type": "Propriété",
                "enonce": "Pour des angles supplémentaires",
                "hypothese": "x angle quelconque",
                "conclusion": "cos(π - x) = -cos(x) • sin(π - x) = sin(x)",
                "legende": "π - x = angle supplémentaire • π = 180°"
            },
        ],
        
        "🔢 Second degré": [
            {
                "titre": "Théorème du discriminant",
                "type": "Théorème",
                "enonce": "Le signe du discriminant détermine le nombre de racines",
                "hypothese": "ax² + bx + c = 0 avec a ≠ 0",
                "conclusion": "Δ > 0 : 2 racines • Δ = 0 : 1 racine • Δ < 0 : 0 racine",
                "legende": "Δ = b² - 4ac • a, b, c = coefficients • Racines réelles"
            },
            {
                "titre": "Propriété somme-produit",
                "type": "Propriété",
                "enonce": "La somme et le produit des racines dépendent des coefficients",
                "hypothese": "ax² + bx + c ayant deux racines x₁ et x₂",
                "conclusion": "x₁ + x₂ = -b/a • x₁ × x₂ = c/a",
                "legende": "x₁, x₂ = racines • a, b, c = coefficients • Relations de Viète"
            },
            {
                "titre": "Forme factorisée",
                "type": "Propriété",
                "enonce": "Un trinôme avec deux racines peut se factoriser",
                "hypothese": "ax² + bx + c avec racines x₁ et x₂",
                "conclusion": "ax² + bx + c = a(x - x₁)(x - x₂)",
                "legende": "a = coefficient dominant • x₁, x₂ = racines"
            },
            {
                "titre": "Signe du trinôme",
                "type": "Propriété",
                "enonce": "Le signe d'un trinôme dépend du signe de a et de la position par rapport aux racines",
                "hypothese": "ax² + bx + c avec a ≠ 0",
                "conclusion": "Entre les racines : signe opposé à a • En dehors : même signe que a",
                "legende": "a = coefficient dominant • Si Δ < 0 : toujours le signe de a"
            },
        ],
        
        "📊 Probabilités": [
            {
                "titre": "Propriété probabilité contraire",
                "type": "Propriété",
                "enonce": "La somme des probabilités d'un événement et de son contraire vaut 1",
                "hypothese": "A événement",
                "conclusion": "P(A) + P(Ā) = 1",
                "legende": "A = événement • Ā = événement contraire • P = probabilité"
            },
            {
                "titre": "Formule des probabilités totales",
                "type": "Théorème",
                "enonce": "Si des événements forment une partition, la probabilité totale est la somme",
                "hypothese": "A₁, A₂, ..., Aₙ partition de l'univers",
                "conclusion": "P(B) = P(B∩A₁) + P(B∩A₂) + ... + P(B∩Aₙ)",
                "legende": "Partition = événements disjoints dont la réunion est l'univers • ∩ = intersection"
            },
            {
                "titre": "Formule de Bayes",
                "type": "Théorème",
                "enonce": "Probabilité conditionnelle inversée",
                "hypothese": "A et B événements avec P(B) ≠ 0",
                "conclusion": "P_B(A) = P_A(B) × P(A) / P(B)",
                "legende": "P_B(A) = probabilité de A sachant B • P_A(B) = probabilité de B sachant A"
            },
        ],
        
        "📐 Produit scalaire": [
            {
                "titre": "Propriété de symétrie",
                "type": "Propriété",
                "enonce": "Le produit scalaire est commutatif",
                "hypothese": "u⃗ et v⃗ vecteurs",
                "conclusion": "u⃗ · v⃗ = v⃗ · u⃗",
                "legende": "u⃗, v⃗ = vecteurs • · = produit scalaire • Commutatif = ordre sans importance"
            },
            {
                "titre": "Propriété de linéarité",
                "type": "Propriété",
                "enonce": "Le produit scalaire est linéaire",
                "hypothese": "u⃗, v⃗, w⃗ vecteurs et k réel",
                "conclusion": "u⃗ · (v⃗ + w⃗) = u⃗ · v⃗ + u⃗ · w⃗ • (ku⃗) · v⃗ = k(u⃗ · v⃗)",
                "legende": "k = scalaire (nombre réel) • Distributivité sur l'addition"
            },
            {
                "titre": "Caractérisation orthogonalité",
                "type": "Propriété",
                "enonce": "Deux vecteurs sont orthogonaux si et seulement si leur produit scalaire est nul",
                "hypothese": "u⃗ et v⃗ vecteurs non nuls",
                "conclusion": "u⃗ ⊥ v⃗ ⟺ u⃗ · v⃗ = 0",
                "legende": "⊥ = orthogonaux (perpendiculaires) • ⟺ = équivalence"
            },
            {
                "titre": "Formule d'Al-Kashi",
                "type": "Théorème",
                "enonce": "Généralisation du théorème de Pythagore",
                "hypothese": "ABC triangle quelconque",
                "conclusion": "BC² = AB² + AC² - 2×AB×AC×cos(Â)",
                "legende": "Â = angle en A • Si Â = 90° : retrouve Pythagore"
            },
        ],
        
        # ✅ NOUVEAU CHAPITRE
        "📊 Variables aléatoires": [
            {
                "titre": "Définition loi de probabilité",
                "type": "Propriété",
                "enonce": "Une loi de probabilité associe à chaque issue sa probabilité, avec une somme totale égale à 1",
                "hypothese": "X variable aléatoire avec valeurs x₁, x₂, ..., xₙ",
                "conclusion": "P(X = x₁) + P(X = x₂) + ... + P(X = xₙ) = 1 • Chaque P(X = xᵢ) ∈ [0,1]",
                "legende": "P(X = xᵢ) = probabilité que X prenne la valeur xᵢ • Somme des probabilités = 1"
            },
            {
                "titre": "Espérance mathématique",
                "type": "Propriété",
                "enonce": "L'espérance d'une variable aléatoire est la moyenne pondérée de ses valeurs par leurs probabilités",
                "hypothese": "X variable aléatoire avec valeurs x₁, x₂, ..., xₙ",
                "conclusion": "E(X) = x₁P(X=x₁) + x₂P(X=x₂) + ... + xₙP(X=xₙ)",
                "legende": "E(X) = espérance (valeur moyenne attendue) • Somme des valeurs × probabilités"
            },
            {
                "titre": "Variance d'une variable aléatoire",
                "type": "Propriété",
                "enonce": "La variance mesure la dispersion des valeurs autour de l'espérance",
                "hypothese": "X variable aléatoire d'espérance E(X)",
                "conclusion": "V(X) = E[(X - E(X))²] = E(X²) - [E(X)]²",
                "legende": "V(X) = variance • E(X²) = espérance du carré • [E(X)]² = carré de l'espérance • Formule de König-Huygens"
            },
            {
                "titre": "Propriété de linéarité de l'espérance",
                "type": "Propriété",
                "enonce": "L'espérance d'une transformation affine est la transformation affine de l'espérance",
                "hypothese": "X variable aléatoire, a et b réels",
                "conclusion": "E(aX + b) = aE(X) + b",
                "legende": "a, b = constantes réelles • Linéarité : passe à travers la somme et le produit par constante"
            },
            {
                "titre": "Propriété de la variance",
                "type": "Propriété",
                "enonce": "La variance d'une transformation affine dépend du carré du coefficient multiplicateur",
                "hypothese": "X variable aléatoire, a et b réels",
                "conclusion": "V(aX + b) = a²V(X) • σ(aX + b) = |a|σ(X)",
                "legende": "a² = carré du coefficient • |a| = valeur absolue • σ = écart-type • La constante b n'affecte pas la variance"
            },
        ],
    },
    
    "Terminale": {
        "∞ Limites et continuité": [
            {
                "titre": "Théorème des gendarmes",
                "type": "Théorème",
                "enonce": "Si une fonction est encadrée par deux fonctions ayant même limite, elle a cette limite",
                "hypothese": "f, g, h telles que g ≤ f ≤ h et lim(g) = lim(h) = l",
                "conclusion": "lim(f) = l",
                "legende": "f, g, h = fonctions • l = limite commune • Encadrement sur un intervalle"
            },
            {
                "titre": "Théorème des valeurs intermédiaires (TVI)",
                "type": "Théorème",
                "enonce": "Une fonction continue sur un intervalle prend toute valeur intermédiaire",
                "hypothese": "f continue sur [a;b], k entre f(a) et f(b)",
                "conclusion": "∃c ∈ [a;b] tel que f(c) = k",
                "legende": "f = fonction continue • ∃ = il existe • k = valeur intermédiaire • c = antécédent"
            },
            {
                "titre": "Corollaire du TVI (bijection)",
                "type": "Théorème",
                "enonce": "Une fonction continue et strictement monotone réalise une bijection",
                "hypothese": "f continue et strictement monotone sur [a;b]",
                "conclusion": "Pour tout k entre f(a) et f(b), l'équation f(x) = k a une unique solution",
                "legende": "Monotone = toujours croissante ou toujours décroissante • Unique = une seule"
            },
            {
                "titre": "Limite d'une somme",
                "type": "Propriété",
                "enonce": "La limite d'une somme est la somme des limites",
                "hypothese": "lim(f) = l et lim(g) = l'",
                "conclusion": "lim(f + g) = l + l'",
                "legende": "f, g = fonctions • l, l' = limites finies • Attention aux formes indéterminées"
            },
            {
                "titre": "Croissances comparées",
                "type": "Théorème",
                "enonce": "L'exponentielle croît plus vite que toute puissance",
                "hypothese": "x → +∞",
                "conclusion": "lim(eˣ/xⁿ) = +∞ pour tout n • lim(xⁿ/eˣ) = 0",
                "legende": "e = exponentielle • n = puissance quelconque • e domine toutes les puissances"
            },
        ],
        
        "📈 Dérivabilité": [
            {
                "titre": "Théorème de Rolle",
                "type": "Théorème",
                "enonce": "Si une fonction dérivable a deux valeurs égales, sa dérivée s'annule entre ces points",
                "hypothese": "f dérivable sur [a;b], continue sur [a;b], f(a) = f(b)",
                "conclusion": "∃c ∈ ]a;b[ tel que f'(c) = 0",
                "legende": "f = fonction • c = point où la tangente est horizontale • ∃ = il existe"
            },
            {
                "titre": "Théorème des accroissements finis (TAF)",
                "type": "Théorème",
                "enonce": "Il existe un point où la tangente est parallèle à la corde",
                "hypothese": "f dérivable sur [a;b], continue sur [a;b]",
                "conclusion": "∃c ∈ ]a;b[ tel que f'(c) = (f(b) - f(a))/(b - a)",
                "legende": "f'(c) = pente de la tangente • (f(b)-f(a))/(b-a) = pente de la corde"
            },
            {
                "titre": "Inégalité des accroissements finis",
                "type": "Théorème",
                "enonce": "Si la dérivée est bornée, la fonction est lipschitzienne",
                "hypothese": "f dérivable, |f'(x)| ≤ M pour tout x",
                "conclusion": "|f(b) - f(a)| ≤ M|b - a|",
                "legende": "M = borne (constante) • | | = valeur absolue • Contrôle la variation"
            },
        ],
        
        "∫ 📈 🔄 F Primitives" : [
            {
                "titre": "Définition d'une primitive",
                "type": "Propriété",
                "enonce": "F est une primitive de f sur un intervalle I si F est dérivable sur I et F' = f",
                "hypothese": "f fonction continue sur I",
                "conclusion": "F primitive de f ⟺ F'(x) = f(x) pour tout x ∈ I",
                "legende": "F = primitive • F' = dérivée de F • f = fonction à primitiver"
            },
            {
                "titre": "Unicité à une constante près",
                "type": "Théorème",
                "enonce": "Si F et G sont deux primitives d'une même fonction, elles diffèrent d'une constante",
                "hypothese": "F et G primitives de f sur I",
                "conclusion": "∃C ∈ ℝ : G(x) = F(x) + C pour tout x ∈ I",
                "legende": "C = constante d'intégration • Toutes les primitives diffèrent d'une constante"
            },
            {
                "titre": "Primitive de xⁿ",
                "type": "Propriété",
                "enonce": "Une primitive de xⁿ est xⁿ⁺¹/(n+1) pour n ≠ -1",
                "hypothese": "n entier ou réel, n ≠ -1",
                "conclusion": "Primitive de xⁿ : F(x) = xⁿ⁺¹/(n+1) + C",
                "legende": "C = constante d'intégration • Cas particulier n=-1 : primitive de 1/x = ln|x|"
            },
            {
                "titre": "Primitive de 1/x",
                "type": "Propriété",
                "enonce": "Une primitive de 1/x est ln|x|",
                "hypothese": "x ≠ 0",
                "conclusion": "Primitive de 1/x : F(x) = ln|x| + C",
                "legende": "ln = logarithme népérien • |x| = valeur absolue • C = constante"
            },
            {
                "titre": "Primitive de l'exponentielle",
                "type": "Propriété",
                "enonce": "Une primitive de eˣ est eˣ",
                "hypothese": "x réel",
                "conclusion": "Primitive de eˣ : F(x) = eˣ + C",
                "legende": "e = exponentielle • C = constante • L'exponentielle est sa propre primitive"
            },
            {
                "titre": "Primitives des fonctions trigonométriques",
                "type": "Propriété",
                "enonce": "Primitives de cos et sin",
                "hypothese": "x réel",
                "conclusion": "Primitive de cos(x) : sin(x) + C • Primitive de sin(x) : -cos(x) + C",
                "legende": "C = constante • Attention au signe - pour sin"
            },
            {
                "titre": "Primitive de u'×uⁿ",
                "type": "Propriété",
                "enonce": "Une primitive de u'(x)×[u(x)]ⁿ est [u(x)]ⁿ⁺¹/(n+1)",
                "hypothese": "u dérivable, n ≠ -1",
                "conclusion": "Primitive de u'×uⁿ : F(x) = uⁿ⁺¹/(n+1) + C",
                "legende": "u' = dérivée de u • Formule de composition • Très utile pour les calculs"
            },
            {
                "titre": "Primitive de u'/u",
                "type": "Propriété",
                "enonce": "Une primitive de u'(x)/u(x) est ln|u(x)|",
                "hypothese": "u dérivable, u(x) ≠ 0",
                "conclusion": "Primitive de u'/u : F(x) = ln|u(x)| + C",
                "legende": "u' = dérivée de u • Cas particulier n=-1 de la formule précédente"
            },
            {
                "titre": "Primitive de u'×eᵘ",
                "type": "Propriété",
                "enonce": "Une primitive de u'(x)×eᵘ⁽ˣ⁾ est eᵘ⁽ˣ⁾",
                "hypothese": "u dérivable",
                "conclusion": "Primitive de u'×eᵘ : F(x) = eᵘ + C",
                "legende": "u' = dérivée de u • Composition avec l'exponentielle"
            },
            {
                "titre": "Linéarité des primitives",
                "type": "Propriété",
                "enonce": "Une primitive d'une combinaison linéaire est la combinaison linéaire des primitives",
                "hypothese": "f et g admettent des primitives, a et b réels",
                "conclusion": "Primitive de af + bg : aF + bG où F primitive de f, G primitive de g",
                "legende": "a, b = constantes • Linéarité comme pour les dérivées"
            },
        ],
        
        "∫ Intégrales": [
            {
                "titre": "Propriété de linéarité",
                "type": "Propriété",
                "enonce": "L'intégrale est linéaire",
                "hypothese": "f et g continues, k réel",
                "conclusion": "∫(f + g) = ∫f + ∫g • ∫(kf) = k∫f",
                "legende": "∫ = intégrale • k = constante • Linéarité = somme et multiplication par constante"
            },
            {
                "titre": "Relation de Chasles",
                "type": "Propriété",
                "enonce": "On peut découper une intégrale en plusieurs morceaux",
                "hypothese": "f continue sur un intervalle contenant a, b, c",
                "conclusion": "∫ₐᵇ f + ∫ᵇᶜ f = ∫ₐᶜ f",
                "legende": "a, b, c = bornes • Ordre quelconque (peut donner des signes -)"
            },
            {
                "titre": "Positivité de l'intégrale",
                "type": "Propriété",
                "enonce": "L'intégrale d'une fonction positive est positive",
                "hypothese": "f ≥ 0 sur [a;b] avec a ≤ b",
                "conclusion": "∫ₐᵇ f ≥ 0",
                "legende": "f ≥ 0 = fonction positive • Aire algébrique positive"
            },
            {
                "titre": "Croissance de l'intégrale",
                "type": "Propriété",
                "enonce": "Si f ≤ g, alors leurs intégrales respectent l'inégalité",
                "hypothese": "f ≤ g sur [a;b] avec a ≤ b",
                "conclusion": "∫ₐᵇ f ≤ ∫ₐᵇ g",
                "legende": "Ordre préservé par l'intégration"
            },
            {
                "titre": "Inégalité triangulaire",
                "type": "Propriété",
                "enonce": "La valeur absolue de l'intégrale est inférieure à l'intégrale de la valeur absolue",
                "hypothese": "f continue sur [a;b]",
                "conclusion": "|∫ₐᵇ f| ≤ ∫ₐᵇ |f|",
                "legende": "| | = valeur absolue • Aire signée ≤ aire totale"
            },
        ],
        
        "📈 Fonction exponentielle": [
            {
                "titre": "Unicité de l'exponentielle",
                "type": "Théorème",
                "enonce": "Il existe une unique fonction égale à sa dérivée et valant 1 en 0",
                "hypothese": "f' = f et f(0) = 1",
                "conclusion": "f = exp",
                "legende": "exp = fonction exponentielle • Caractérisation unique"
            },
            {
                "titre": "Propriété algébrique fondamentale",
                "type": "Propriété",
                "enonce": "L'exponentielle transforme somme en produit",
                "hypothese": "a, b réels",
                "conclusion": "eᵃ⁺ᵇ = eᵃ × eᵇ",
                "legende": "e = exponentielle • Propriété fondamentale (morphisme)"
            },
            {
                "titre": "Stricte positivité",
                "type": "Propriété",
                "enonce": "L'exponentielle est toujours strictement positive",
                "hypothese": "x réel quelconque",
                "conclusion": "eˣ > 0",
                "legende": "e⁰ = 1 • Jamais nulle ni négative"
            },
        ],
        
        "📊 Fonction logarithme": [
            {
                "titre": "Propriété réciproque",
                "type": "Propriété",
                "enonce": "ln et exp sont réciproques l'une de l'autre",
                "hypothese": "x > 0 et y réel",
                "conclusion": "ln(eʸ) = y • eˡⁿ⁽ˣ⁾ = x",
                "legende": "ln = logarithme népérien • exp = exponentielle • Fonctions réciproques"
            },
            {
                "titre": "Propriété algébrique fondamentale",
                "type": "Propriété",
                "enonce": "Le logarithme transforme produit en somme",
                "hypothese": "a > 0, b > 0",
                "conclusion": "ln(a × b) = ln(a) + ln(b)",
                "legende": "ln = logarithme • Propriété fondamentale (morphisme)"
            },
            {
                "titre": "Logarithme d'une puissance",
                "type": "Propriété",
                "enonce": "Le logarithme d'une puissance fait descendre l'exposant",
                "hypothese": "a > 0, n réel",
                "conclusion": "ln(aⁿ) = n × ln(a)",
                "legende": "n = exposant quelconque (même non entier)"
            },
        ],
        
        "🎲 Probabilités": [
            {
                "titre": "Théorème central limite",
                "type": "Théorème",
                "enonce": "La somme de variables aléatoires indépendantes suit asymptotiquement une loi normale",
                "hypothese": "X₁, ..., Xₙ variables i.i.d. de moyenne μ et variance σ²",
                "conclusion": "(X₁ + ... + Xₙ - nμ)/(σ√n) → N(0,1)",
                "legende": "i.i.d. = indépendantes et identiquement distribuées • n → ∞ • N(0,1) = loi normale centrée réduite"
            },
            {
                "titre": "Loi faible des grands nombres",
                "type": "Théorème",
                "enonce": "La moyenne empirique converge vers l'espérance",
                "hypothese": "X₁, ..., Xₙ variables i.i.d. d'espérance μ",
                "conclusion": "X̄ₙ = (X₁ + ... + Xₙ)/n → μ quand n → ∞",
                "legende": "X̄ₙ = moyenne empirique • μ = espérance théorique"
            },
            {
                "titre": "Propriété loi normale",
                "type": "Propriété",
                "enonce": "Une loi normale est caractérisée par sa moyenne et son écart-type",
                "hypothese": "X ~ N(μ, σ²)",
                "conclusion": "P(μ - σ ≤ X ≤ μ + σ) ≈ 0,68 • P(μ - 2σ ≤ X ≤ μ + 2σ) ≈ 0,95",
                "legende": "μ = moyenne • σ = écart-type • Règle des 68-95-99,7"
            },
            {
                "titre": "Théorème de Moivre-Laplace",
                "type": "Théorème",
                "enonce": "Une loi binomiale converge vers une loi normale",
                "hypothese": "Xₙ ~ B(n, p) avec n grand",
                "conclusion": "Xₙ ≈ N(np, np(1-p))",
                "legende": "B = loi binomiale • n grand (n > 30) • np > 5 et n(1-p) > 5"
            },
        ],
        
        "📐 Géométrie dans l'espace": [
            {
                "titre": "Propriété colinéarité",
                "type": "Propriété",
                "enonce": "Deux vecteurs sont colinéaires si leurs coordonnées sont proportionnelles",
                "hypothese": "u⃗(x, y, z) et v⃗(x', y', z') non nuls",
                "conclusion": "u⃗ et v⃗ colinéaires ⟺ ∃k : v⃗ = ku⃗",
                "legende": "k = coefficient de proportionnalité • ⟺ = équivalence • Colinéaires = même direction"
            },
            {
                "titre": "Propriété vecteur normal",
                "type": "Propriété",
                "enonce": "Un vecteur normal à un plan est orthogonal à tous les vecteurs du plan",
                "hypothese": "n⃗ vecteur normal au plan P",
                "conclusion": "Pour tout u⃗ dans P : n⃗ · u⃗ = 0",
                "legende": "n⃗ = vecteur normal • P = plan • · = produit scalaire • Orthogonal à tout vecteur du plan"
            },
            {
                "titre": "Plans parallèles",
                "type": "Propriété",
                "enonce": "Deux plans sont parallèles s'ils ont des vecteurs normaux colinéaires",
                "hypothese": "P₁ de vecteur normal n⃗₁, P₂ de vecteur normal n⃗₂",
                "conclusion": "P₁ // P₂ ⟺ n⃗₁ et n⃗₂ colinéaires",
                "legende": "// = parallèles • Vecteurs normaux dans la même direction"
            },
            {
                "titre": "Droites orthogonales",
                "type": "Propriété",
                "enonce": "Deux droites sont orthogonales si leurs vecteurs directeurs le sont",
                "hypothese": "D₁ de vecteur directeur u⃗₁, D₂ de vecteur directeur u⃗₂",
                "conclusion": "D₁ ⊥ D₂ ⟺ u⃗₁ · u⃗₂ = 0",
                "legende": "⊥ = orthogonales (perpendiculaires) • Vecteurs directeurs orthogonaux"
            },
        ],
        
        "🔢 Suites": [
            {
                "titre": "Théorème de convergence monotone",
                "type": "Théorème",
                "enonce": "Une suite croissante majorée converge",
                "hypothese": "Suite (uₙ) croissante et majorée",
                "conclusion": "(uₙ) converge vers une limite finie",
                "legende": "Croissante = uₙ₊₁ ≥ uₙ • Majorée = uₙ ≤ M pour tout n • M = majorant"
            },
            {
                "titre": "Théorème suite adjacente",
                "type": "Théorème",
                "enonce": "Deux suites adjacentes convergent vers la même limite",
                "hypothese": "(uₙ) croissante, (vₙ) décroissante, lim(vₙ - uₙ) = 0",
                "conclusion": "(uₙ) et (vₙ) convergent vers la même limite l",
                "legende": "Adjacentes = l'une croissante, l'autre décroissante, écart tend vers 0"
            },
            {
                "titre": "Unicité de la limite",
                "type": "Propriété",
                "enonce": "Une suite convergente a une unique limite",
                "hypothese": "(uₙ) converge",
                "conclusion": "La limite est unique",
                "legende": "Pas de convergence vers deux valeurs différentes"
            },
        ],
        
        "📊 Convexité": [
            {
                "titre": "Caractérisation par la dérivée seconde",
                "type": "Théorème",
                "enonce": "Le signe de la dérivée seconde caractérise la convexité",
                "hypothese": "f deux fois dérivable",
                "conclusion": "f convexe ⟺ f'' ≥ 0 • f concave ⟺ f'' ≤ 0",
                "legende": "f'' = dérivée seconde • ⟺ = équivalence • Convexe = courbe en U, concave = courbe en ∩"
            },
            {
                "titre": "Inégalité de Jensen",
                "type": "Théorème",
                "enonce": "Pour une fonction convexe, la valeur au barycentre est inférieure au barycentre des valeurs",
                "hypothese": "f convexe, λ ∈ [0,1]",
                "conclusion": "f(λa + (1-λ)b) ≤ λf(a) + (1-λ)f(b)",
                "legende": "λ = coefficient de pondération • Corde au-dessus de la courbe"
            },
            {
                "titre": "Position tangente-courbe",
                "type": "Propriété",
                "enonce": "Pour une fonction convexe, la courbe est au-dessus de ses tangentes",
                "hypothese": "f convexe dérivable",
                "conclusion": "f(x) ≥ f(a) + f'(a)(x - a) pour tout x et a",
                "legende": "f(a) + f'(a)(x-a) = équation de la tangente en a • Courbe au-dessus"
            },
        ],
        
        # ✅ NOUVEAU CHAPITRE
        "🔢 Combinatoire et dénombrement": [
            {
                "titre": "Principe multiplicatif",
                "type": "Théorème",
                "enonce": "Si une tâche peut être décomposée en k étapes avec n₁, n₂, ..., nₖ possibilités, le nombre total est le produit",
                "hypothese": "k étapes successives indépendantes avec n₁, n₂, ..., nₖ choix",
                "conclusion": "Nombre total de possibilités = n₁ × n₂ × ... × nₖ",
                "legende": "Étapes indépendantes = le choix à une étape n'affecte pas les autres • Principe fondamental du dénombrement"
            },
            {
                "titre": "Nombre de permutations",
                "type": "Théorème",
                "enonce": "Le nombre de façons d'ordonner n objets distincts est n factorielle",
                "hypothese": "n objets distincts à ordonner",
                "conclusion": "Nombre de permutations = n! = n × (n-1) × (n-2) × ... × 2 × 1",
                "legende": "n! = factorielle de n • Par convention : 0! = 1 • Arrangements complets"
            },
            {
                "titre": "Nombre d'arrangements",
                "type": "Théorème",
                "enonce": "Le nombre de façons de choisir et ordonner k objets parmi n est le nombre d'arrangements",
                "hypothese": "n objets distincts, choix de k objets (k ≤ n) avec ordre",
                "conclusion": "Aₙᵏ = n!/(n-k)! = n × (n-1) × ... × (n-k+1)",
                "legende": "Aₙᵏ = arrangements de k parmi n • Ordre important • k facteurs décroissants à partir de n"
            },
            {
                "titre": "Nombre de combinaisons",
                "type": "Théorème",
                "enonce": "Le nombre de façons de choisir k objets parmi n sans tenir compte de l'ordre est le coefficient binomial",
                "hypothese": "n objets distincts, choix de k objets (k ≤ n) sans ordre",
                "conclusion": "C(n,k) = n!/(k!(n-k)!) = Aₙᵏ/k!",
                "legende": "C(n,k) = combinaisons (notation aussi : Cₙᵏ ou (n k)) • Ordre sans importance • Nombre de sous-ensembles"
            },
            {
                "titre": "Formule du binôme de Newton",
                "type": "Théorème",
                "enonce": "Le développement de (a+b)ⁿ fait intervenir les coefficients binomiaux",
                "hypothese": "a, b nombres réels, n entier naturel",
                "conclusion": "(a+b)ⁿ = Σₖ₌₀ⁿ C(n,k)aⁿ⁻ᵏbᵏ",
                "legende": "Σ = somme de k=0 à k=n • C(n,k) = coefficient binomial • Triangle de Pascal"
            },
        ],
            "ℂ 🔢 🧮 i ∞ Nombres complexes": [
            {
                "titre": "Forme algébrique d'un complexe",
                "type": "Propriété",
                "enonce": "Tout nombre complexe s'écrit de manière unique sous la forme a + ib avec a et b réels",
                "hypothese": "z nombre complexe",
                "conclusion": "z = a + ib avec a, b ∈ ℝ et i² = -1",
                "legende": "a = partie réelle Re(z) • b = partie imaginaire Im(z) • i = unité imaginaire"
            },
            {
                "titre": "Égalité de nombres complexes",
                "type": "Propriété",
                "enonce": "Deux complexes sont égaux si et seulement si leurs parties réelles et imaginaires sont égales",
                "hypothese": "z = a + ib et z' = a' + ib'",
                "conclusion": "z = z' ⟺ a = a' et b = b'",
                "legende": "⟺ = équivalence • Permet de résoudre des équations complexes"
            },
            {
                "titre": "Conjugué d'un nombre complexe",
                "type": "Propriété",
                "enonce": "Le conjugué d'un complexe z = a + ib est z̄ = a - ib",
                "hypothese": "z = a + ib complexe",
                "conclusion": "z̄ = a - ib • z + z̄ = 2a • z × z̄ = a² + b²",
                "legende": "z̄ = conjugué • Symétrie par rapport à l'axe réel • z × z̄ toujours réel positif"
            },
            {
                "titre": "Module d'un nombre complexe",
                "type": "Propriété",
                "enonce": "Le module de z = a + ib est |z| = √(a² + b²)",
                "hypothese": "z = a + ib complexe",
                "conclusion": "|z| = √(a² + b²) = √(z × z̄)",
                "legende": "|z| = module (distance à l'origine) • Toujours positif • |z|² = z × z̄"
            },
            {
                "titre": "Propriétés du module",
                "type": "Propriété",
                "enonce": "Le module respecte le produit et transforme la somme en inégalité triangulaire",
                "hypothese": "z, z' complexes",
                "conclusion": "|z × z'| = |z| × |z'| • |z + z'| ≤ |z| + |z'|",
                "legende": "Multiplicativité du module • Inégalité triangulaire comme pour les vecteurs"
            },
            {
                "titre": "Argument d'un nombre complexe",
                "type": "Propriété",
                "enonce": "L'argument de z non nul est une mesure de l'angle (Ox, Oz)",
                "hypothese": "z ≠ 0 complexe",
                "conclusion": "arg(z) = θ tel que z = |z|(cos θ + i sin θ)",
                "legende": "θ = argument (défini modulo 2π) • Angle polaire • O = origine, x = axe réel"
            },
            {
                "titre": "Forme trigonométrique (ou forme polaire)",
                "type": "Propriété",
                "enonce": "Tout complexe non nul s'écrit z = r(cos θ + i sin θ) avec r = |z| et θ = arg(z)",
                "hypothese": "z ≠ 0 complexe",
                "conclusion": "z = r(cos θ + i sin θ) où r = |z|, θ = arg(z)",
                "legende": "r = module • θ = argument • Notation aussi : z = r·eⁱᶿ"
            },
            {
                "titre": "Formule de Moivre",
                "type": "Théorème",
                "enonce": "Pour tout entier n, (cos θ + i sin θ)ⁿ = cos(nθ) + i sin(nθ)",
                "hypothese": "θ réel, n entier",
                "conclusion": "(cos θ + i sin θ)ⁿ = cos(nθ) + i sin(nθ)",
                "legende": "Permet de linéariser cosⁿ et sinⁿ • Puissances de complexes de module 1"
            },
            {
                "titre": "Formule d'Euler",
                "type": "Théorème",
                "enonce": "L'exponentielle complexe vérifie eⁱᶿ = cos θ + i sin θ",
                "hypothese": "θ réel",
                "conclusion": "eⁱᶿ = cos θ + i sin θ",
                "legende": "Lien fondamental entre exponentielle et trigonométrie • Notation exponentielle"
            },
            {
                "titre": "Interprétation géométrique",
                "type": "Propriété",
                "enonce": "Dans le plan complexe, z correspond au point M de coordonnées (a, b) ou au vecteur OM⃗",
                "hypothese": "z = a + ib complexe",
                "conclusion": "M(a, b) = affixe z • |z| = OM • arg(z) = angle (Ox, OM⃗)",
                "legende": "Plan complexe = identification ℂ ↔ ℝ² • Géométrie ↔ algèbre"
            },
        ],
        '📏 📐Homothéties' : [
            {
                "titre": "Définition de l'homothétie",
                "type": "Propriété",
                "enonce": "L'homothétie de centre O et de rapport k transforme M en M' tel que OM'⃗ = k × OM⃗",
                "hypothese": "O centre, k réel non nul (rapport)",
                "conclusion": "h(O,k)(M) = M' ⟺ OM'⃗ = k·OM⃗",
                "legende": "k > 0 : même sens • k < 0 : sens opposé • |k| = facteur d'agrandissement"
            },
            {
                "titre": "Image d'une droite par homothétie",
                "type": "Théorème",
                "enonce": "L'image d'une droite par une homothétie est une droite parallèle (ou la même si elle passe par le centre)",
                "hypothese": "h(O,k) homothétie, d droite",
                "conclusion": "Si O ∈ d : h(d) = d • Si O ∉ d : h(d) // d",
                "legende": "// = parallèle • Conservation du parallélisme"
            },
            {
                "titre": "Conservation des angles",
                "type": "Propriété",
                "enonce": "Une homothétie conserve les angles orientés",
                "hypothese": "h(O,k) homothétie",
                "conclusion": "(MA⃗, MB⃗) = (M'A'⃗, M'B'⃗) où M' = h(M), A' = h(A), B' = h(B)",
                "legende": "Angles conservés • Les figures sont semblables"
            },
            {
                "titre": "Rapport de longueurs",
                "type": "Propriété",
                "enonce": "Les longueurs sont multipliées par |k|",
                "hypothese": "h(O,k) homothétie, A' = h(A), B' = h(B)",
                "conclusion": "A'B' = |k| × AB",
                "legende": "|k| = valeur absolue du rapport • Facteur d'agrandissement (si |k| > 1) ou de réduction (si |k| < 1)"
            },
            {
                "titre": "Composition d'homothéties de même centre",
                "type": "Théorème",
                "enonce": "La composée de deux homothéties de même centre est une homothétie de même centre",
                "hypothese": "h(O,k) et h(O,k') homothéties de même centre O",
                "conclusion": "h(O,k') ∘ h(O,k) = h(O, k×k')",
                "legende": "∘ = composition • Les rapports se multiplient"
            },
            {
                "titre": "Homothétie et vecteurs",
                "type": "Propriété",
                "enonce": "L'image d'un vecteur par une homothétie de rapport k est le vecteur multiplié par k",
                "hypothese": "h(O,k) homothétie, u⃗ vecteur",
                "conclusion": "h(AB⃗) = A'B'⃗ = k·AB⃗",
                "legende": "Multiplication vectorielle • k peut être négatif (changement de sens)"
            },
        ]

    }
}



def render_theorem_card(theorem_data: dict, level: str):
    """Affiche une carte de théorème ou propriété"""
    
    titre = theorem_data['titre']
    type_item = theorem_data['type']
    enonce = theorem_data['enonce']
    hypothese = theorem_data.get('hypothese', '')
    conclusion = theorem_data.get('conclusion', '')
    legende = theorem_data.get('legende', '')
    
    if type_item == "Théorème":
        badge_class = "theorem-badge"
        card_class = "theorem-card"
    else:
        badge_class = "property-badge"
        card_class = "property-card"
    
    html = f'''<div class="{card_class}">
    <div class="theorem-header">
        <span class="theorem-title">{_html.escape(titre)}</span>
        <div class="badges">
            <span class="{badge_class}">{_html.escape(type_item)}</span>
            <span class="level-badge-small">{_html.escape(level)}</span>
        </div>
    </div>
    <div class="theorem-enonce">{_html.escape(enonce)}</div>'''

    if hypothese and conclusion:
        html += f'''
    <div class="theorem-statement">
        <div class="hypothesis">
            <span class="statement-label">📌 Hypothèse :</span>
            <span class="statement-text">{_html.escape(hypothese)}</span>
        </div>
        <div class="conclusion">
            <span class="statement-label">✅ Conclusion :</span>
            <span class="statement-text">{_html.escape(conclusion)}</span>
        </div>
    </div>'''

    if legende:
        html += f'''
    <div class="theorem-legend">
        <span class="legend-icon">💡</span>
        {_html.escape(legende)}
    </div>'''
    
    html += '\n</div>'
    
    # CRITIQUE : Cette ligne est INDISPENSABLE
    st.markdown(html, unsafe_allow_html=True)




def render_theoremes():
    """Section Théorèmes et Propriétés - Affichage complet"""
    st.subheader("📜 Théorèmes et Propriétés")
    
    level = st.selectbox(
        "🎓 Niveau",
        ["Seconde", "Première", "Terminale"],
        key="level_theorem"
    )
    
    chapitres = list(THEOREMES_PROPRIETES[level].keys())
    selected = st.radio(
        "Chapitre",
        ["Tous"] + chapitres,
        horizontal=True,
        key="chap_theorem"
    )
    
    st.divider()
    
    # Affichage des théorèmes
    if selected == "Tous":
        for chapitre_name, items in THEOREMES_PROPRIETES[level].items():
            st.markdown(
                f'<div class="theorem-category-header">{_html.escape(chapitre_name)}</div>',
                unsafe_allow_html=True
            )
            
            cols = st.columns(2)  # 2 colonnes pour les théorèmes
            for i, item in enumerate(items):
                with cols[i % 2]:
                    render_theorem_card(item, level)
    else:
        items = THEOREMES_PROPRIETES[level][selected]
        st.markdown(
            f'<div class="theorem-category-header">{_html.escape(selected)}</div>',
            unsafe_allow_html=True
        )
        
        cols = st.columns(2)
        for i, item in enumerate(items):
            with cols[i % 2]:
                render_theorem_card(item, level)
    
    st.divider()
    
    # Bouton téléchargement
    if st.button("📥 Télécharger PDF", type="primary", use_container_width=False, key="dl_thm"):
        content = f"# Théorèmes et Propriétés - {level}\n\n"
        for chap, items in THEOREMES_PROPRIETES[level].items():
            content += f"## {chap}\n\n"
            for item in items:
                content += f"### {item['type']} : {item['titre']}\n\n"
                content += f"**Énoncé :** {item['enonce']}\n\n"
                if item.get('hypothese'):
                    content += f"**Hypothèse :** {item['hypothese']}\n\n"
                if item.get('conclusion'):
                    content += f"**Conclusion :** {item['conclusion']}\n\n"
                if item.get('legende'):
                    content += f"**Légende :** {item['legende']}\n\n"
                content += "---\n\n"
        
        pdf = markdown_to_pdf(content, f"Théorèmes {level}")
        st.download_button(
            label="💾 Sauvegarder le PDF",
            data=pdf,
            file_name=f"theoremes_{level.lower()}.pdf",
            mime="application/pdf",
            key="dl_theorem_pdf"
        )



def render_cours_tab():
    """Onglet Cours avec sous-sections"""
    st.header("📖 Cours & Références")
    
    section = st.radio(
        "Que cherches-tu ?",
        ["📚 Explication", "📖 Définitions", "📜 Théorèmes & Propriétés", "📌 Référence"],
        horizontal=True,
        key="section"
    )
    
    st.divider()
    
    if section == "📌 Référence":
        render_reference_rapide()
    elif section == "📚 Explication":
        render_explication()
    elif section == "📖 Définitions":
        render_definitions()
    else:
        render_theoremes()


def render_explication():
    """Section Explication avec LLM Router"""
    st.subheader("📚 Explication")
    
    col1, col2 = st.columns(2)
    with col1:
        create_topic_selector(key="topic")
    with col2:
        level = create_level_selector(key="level")
    
    st.divider()
    
    # CHAT CONTINU AVEC LLM ROUTER
    render_continuous_chat(
        section_key="explication",
        placeholder="Ex : Explique le théorème de Pythagore",
        rag_function_type="query",
        level=level
    )


def render_definitions():
    """Section Définitions avec LLM Router"""
    st.subheader("📖 Définitions")
    
    level = create_level_selector(key="level_def")
    
    st.divider()
    
    # CHAT CONTINU AVEC LLM ROUTER
    render_continuous_chat(
        section_key="definitions",
        placeholder="Ex : Qu'est-ce qu'une dérivée ?",
        rag_function_type="definition",
        level=level
    )




def render_exercices_tab():
    """Onglet Exercices avec LLM Router"""
    st.header("✏️ Exercices")

    render_continuous_chat(
        section_key="exercices",
        placeholder="Ex : Résous x² - 4x + 3 = 0",
        rag_function_type="exercise",
    )


def render_reference_rapide():
    """Référence rapide (pas de chat - juste consultation)"""
    st.subheader("📌 Référence rapide")
    st.info("💡 Toutes les formules essentielles à portée de main")
    
    level = st.selectbox(
        "🎓 Niveau",
        ["Seconde", "Première", "Terminale"],
        key="level_ref"
    )
    
    categories = list(FORMULES_REFERENCE[level].keys())
    selected = st.radio(
        "Catégorie",
        ["Toutes"] + categories,
        horizontal=True,
        key="cat"
    )
    
    st.divider()
    
    # Affichage des formules
    if selected == "Toutes":
        for cat_name, formules in FORMULES_REFERENCE[level].items():
            st.markdown(f'<div class="category-header">{_html.escape(cat_name)}</div>', unsafe_allow_html=True)
            
            cols = st.columns(3)
            for i, formule in enumerate(formules):
                with cols[i % 3]:
                    render_formula_card(formule, level)
    else:
        formules = FORMULES_REFERENCE[level][selected]
        st.markdown(f'<div class="category-header">{_html.escape(selected)}</div>', unsafe_allow_html=True)
        
        cols = st.columns(3)
        for i, formule in enumerate(formules):
            with cols[i % 3]:
                render_formula_card(formule, level)
    
    st.divider()
    
    # Bouton téléchargement
    if st.button("📥 Télécharger PDF", type="primary", use_container_width=False):
        content = f"# Référence Mathématiques - {level}\n\n"
        for cat, fs in FORMULES_REFERENCE[level].items():
            content += f"## {cat}\n\n"
            for f in fs:
                content += f"**{f['titre']}**\n\n"
                content += f"```\n{f['formule']}\n```\n\n"
                content += f"_{f['description']}_\n\n---\n\n"
        
        pdf = markdown_to_pdf(content, f"Référence {level}")
        st.download_button(
            label="💾 Sauvegarder le PDF",
            data=pdf,
            file_name=f"reference_{level.lower()}.pdf",
            mime="application/pdf",
            key="dl_ref"
        )
