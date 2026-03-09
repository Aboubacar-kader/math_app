# 🧮 IntelliMath

Un assistant pédagogique intelligent pour les élèves de lycée (Seconde à Terminale) en mathématiques, déployé sur Streamlit Cloud.

## ✨ Fonctionnalités

- **💬 Chat conversationnel** — Pose tes questions en langage naturel
- **📚 Base de connaissances RAG** — Réponses ancrées dans tes cours indexés, avec fallback automatique sur le LLM
- **📄 Upload de documents** — Analyse tes cours et exercices (PDF, DOCX, TXT, images PNG/JPG)
- **🖼️ Figures géométriques automatiques** — Détection et tracé automatique (fonctions, paraboles, hyperboles, ellipses, angles, polygones réguliers, cercles, triangles, vecteurs…)
- **📐 Fonctions composées** — Tracé de toute fonction (ln, exp, sin, cos, sinh, cosh, log₁₀, racine cubique…)
- **📸 OCR intelligent** — Extraction de texte et formules depuis images et PDFs scannés via GPT-4o vision
- **📊 Tableau de variations** — Génération automatique depuis la réponse du LLM
- **🎤 Entrée vocale** — Dicte ta question
- **🔊 Synthèse vocale** — Écoute les réponses (gTTS)
- **📥 Export PDF** — Télécharge les réponses en PDF
- **💬 Historique des conversations** — Navigue entre tes conversations

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit >= 1.38 |
| LLM | OpenAI API (gpt-4o) |
| Vision / OCR | GPT-4o vision (images, PDFs scannés) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (local, gratuit) |
| Base vectorielle | Qdrant (local) |
| Orchestration | LangChain |
| Documents | pdfplumber, python-docx, fpdf2 |
| Tracé | Plotly (interactif), Matplotlib |
| Audio | SpeechRecognition >= 3.11, gTTS, pydub |

## 📋 Prérequis

- Python >= 3.12
- Une clé API [OpenAI](https://platform.openai.com/api-keys) (modèle gpt-4o)

## 🚀 Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/Aboubacar-kader/math_app.git
cd math_app
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
# Windows :
.venv\Scripts\activate
# macOS / Linux :
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Crée un fichier `.env` à la racine :

```env
# LLM — OpenAI
MIN_AI_API_KEY=sk-proj-...
MIN_AI_MODEL=gpt-4o
MIN_AI_BASE_URL=https://api.openai.com/v1
MIN_AI_TEMPERATURE=0.2

# Embeddings (local, gratuit)
EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Qdrant (base vectorielle locale)
QDRANT_PATH=./data/vectorstore
QDRANT_COLLECTION_NAME=math_documents

# Traitement des documents
MAX_FILE_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Langue
SPEECH_LANG=fr-FR
TTS_LANG=fr
```

### 5. Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre à `http://localhost:8501`

## ☁️ Déploiement Streamlit Cloud

1. Pousse le dépôt sur GitHub
2. Connecte-le sur [share.streamlit.io](https://share.streamlit.io)
3. Ajoute les secrets dans **App settings → Secrets** (même contenu que `.env`, sans tabulation avant la clé)
4. Déploie

> **Note :** La base vectorielle Qdrant est locale et se recrée automatiquement au démarrage à partir des fichiers `data/knowledge_base/`.

## 📁 Structure du projet

```
math_chat/
├── app.py                        # Point d'entrée Streamlit
├── requirements.txt
├── runtime.txt                   # Version Python pour Streamlit Cloud
├── .env                          # Variables d'environnement (non versionné)
│
├── config/
│   └── settings.py               # Paramètres globaux (pydantic-settings)
│
├── core/
│   ├── llm_manager.py            # Client OpenAI API + embeddings locaux
│   ├── vectorstore_manager.py    # Qdrant (stockage vectoriel)
│   ├── knowledge_base_init.py    # Indexation automatique au démarrage
│   └── prompts.py                # Prompts système
│
├── services/
│   ├── rag_service.py            # Pipeline RAG (recherche + fallback LLM)
│   ├── document_processor.py     # PDF / DOCX / TXT / images (OCR GPT-4o vision)
│   ├── voice_service.py          # STT (SpeechRecognition) + TTS (gTTS)
│   └── math_solver.py            # Utilitaires mathématiques
│
├── ui/
│   ├── tabs.py                   # Onglets principaux + gestion des questions
│   ├── sidebar.py                # Barre latérale (historique, paramètres)
│   ├── components.py             # Composants réutilisables
│   └── chat_interface.py         # Interface de chat
│
├── utils/
│   ├── geometry_plotter.py       # Détection et tracé de figures (Plotly)
│   ├── variation_table.py        # Tableau de variations interactif
│   ├── pdf_utils.py              # Export PDF (fpdf2)
│   ├── logger.py                 # Logging centralisé
│   ├── text_utils.py             # Utilitaires texte
│   ├── file_utils.py             # Utilitaires fichiers
│   └── session_utils.py          # Session Streamlit
│
├── data/
│   ├── vectorstore/              # Qdrant (créé automatiquement)
│   └── knowledge_base/           # Cours à indexer (PDF, DOCX, TXT, MD, images)
│       ├── seconde/
│       ├── premiere/
│       ├── terminale/
│       └── transversal/
│
└── assets/
    ├── styles.css                # Design system (CSS variables, Inter font)
    └── logo_Kaydan tech.png
```

## 📚 Base de connaissances

Place tes cours dans `data/knowledge_base/` selon le niveau :

```
knowledge_base/
├── seconde/      ← Cours de Seconde
├── premiere/     ← Cours de Première
├── terminale/    ← Cours de Terminale
└── transversal/  ← Formules et théorèmes généraux
```

**Formats acceptés :** PDF, DOCX, TXT, MD, PNG, JPG

Les fichiers sont indexés automatiquement au démarrage. Seuls les fichiers nouveaux ou modifiés sont ré-indexés.

## 🧠 Pipeline RAG

1. La question est convertie en vecteur (MiniLM-L6-v2, local)
2. Qdrant recherche les chunks les plus proches (seuil : 0.38)
3. Si des extraits pertinents sont trouvés → réponse ancrée dans les documents
4. Sinon → fallback automatique sur GPT-4o qui répond avec ses connaissances en mathématiques

## 📐 Tracé de figures et fonctions

L'application détecte automatiquement les demandes graphiques dans la question et trace :

| Type | Exemples de déclencheurs |
|------|--------------------------|
| Fonctions | `f(x) = x² - 3x + 2`, `représente ln(1+exp(x))` |
| Fonctions multiples | `trace f(x) = 2x−1 et g(x) = x² sur [-3,3]` |
| Cercle / Ellipse | `trace un cercle de rayon 3`, `ellipse avec a=4, b=2` |
| Triangle / Polygone | `triangle A(0,0) B(4,0) C(2,3)`, `trace un hexagone` |
| Angle | `dessine un angle de 60°` |
| Vecteur / Repère | `trace un repère orthonormé` |

## 🎯 Exemples de questions

```
"Explique-moi le théorème de Pythagore avec des exemples"
"Soit f(x) = 2x² - 3x + 1, résous f(x) = 0"
"C'est quoi une dérivée ?"
"Trace f(x) = x² - 3x + 2 sur [-2, 5]"
"Représente la fonction ln(1+exp(x))"
"Trace f(x) = 2x - 1 et g(x) = x² sur [-3, 3]"
"Dessine un angle de 60°"
"Représente une ellipse avec a=4 et b=2"
```

## 🐛 Dépannage

**`ModuleNotFoundError: aifc`** → Utilise Python >= 3.12 et `SpeechRecognition >= 3.11.0`

**`ModuleNotFoundError: fpdf`** → Vérifie que `fpdf2>=2.7.0` est dans `requirements.txt` (pas `fpdf`)

**Connexion LLM échouée** → Vérifie que `MIN_AI_API_KEY` est défini sans tabulation dans les secrets Streamlit Cloud

**Qdrant vide après redéploiement** → Normal : la base vectorielle se recrée à chaque déploiement depuis `data/knowledge_base/`

**Graphique non tracé** → La question doit contenir un mot déclencheur (trace, représente, construis, dessine, montre…) ou une formule explicite `f(x) = ...`

**OCR image échoué** → Vérifie que le modèle configuré (gpt-4o) supporte la vision et que la clé API est valide

## 👨‍💻 Auteur

Développé par **Kaydan Tech**
