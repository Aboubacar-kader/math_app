# 🧮 IntelliMath

Un assistant pédagogique intelligent pour les élèves de lycée (Seconde à Terminale) en mathématiques, déployé sur Streamlit Cloud.

## ✨ Fonctionnalités

- **💬 Chat conversationnel** — Pose tes questions en langage naturel
- **📚 Base de connaissances RAG** — Réponses ancrées dans tes cours indexés
- **📄 Upload de documents** — Indexe tes cours (PDF, DOCX, TXT, images)
- **🖼️ Figures géométriques automatiques** — Détection et tracé automatique via Matplotlib / Plotly
- **🎤 Entrée vocale** — Dicte ta question (Streamlit >= 1.35 requis)
- **🔊 Synthèse vocale** — Écoute les réponses (gTTS)
- **📥 Export PDF** — Télécharge les réponses en PDF
- **💬 Historique des conversations** — Navigue entre tes conversations

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Interface | Streamlit >= 1.38 |
| LLM | 1min.ai (gpt-4o) |
| Orchestration | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Base vectorielle | Qdrant (local) |
| Documents | pdfplumber, python-docx, fpdf2 |
| Audio | SpeechRecognition >= 3.11, gTTS, pydub |
| Visualisation | Matplotlib, Plotly |

## 📋 Prérequis

- Python >= 3.12 (Python 3.13 compatible)
- Une clé API 1min.ai

## 🚀 Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/ton-username/math_chat.git
cd math_chat
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Crée un fichier `.env` à la racine :

```env
MIN_AI_API_KEY=ta_clé_api
MIN_AI_MODEL=gpt-4o
MIN_AI_BASE_URL=https://api.1min.ai
MIN_AI_TEMPERATURE=0.3

EMBEDDINGS_MODEL=sentence-transformers/all-MiniLM-L6-v2

QDRANT_PATH=./data/vectorstore
QDRANT_COLLECTION_NAME=math_documents

MAX_FILE_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

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
3. Ajoute les secrets dans **App settings → Secrets** (même contenu que `.env`)
4. Déploie

## 📁 Structure du projet

```
math_chat/
├── app.py                        # Point d'entrée
├── requirements.txt
├── .env                          # Variables d'environnement (non versionné)
│
├── config/
│   └── settings.py               # Paramètres globaux (pydantic-settings)
│
├── core/
│   ├── llm_manager.py            # Client LLM (1min.ai)
│   ├── vectorstore_manager.py    # Qdrant
│   ├── knowledge_base_init.py    # Indexation au démarrage
│   └── prompts.py                # Prompts système
│
├── services/
│   ├── rag_service.py            # Pipeline RAG
│   ├── document_processor.py     # PDF / DOCX / TXT / images
│   ├── voice_service.py          # STT (SpeechRecognition) + TTS (gTTS)
│   └── math_solver.py            # Logique mathématique
│
├── ui/
│   ├── tabs.py                   # Onglets principaux
│   ├── sidebar.py                # Barre latérale
│   ├── chat_interface.py         # Interface de chat
│   └── components.py             # Composants réutilisables
│
├── utils/
│   ├── geometry_plotter.py       # Détection et tracé de figures
│   ├── pdf_utils.py              # Export PDF (fpdf2)
│   ├── file_utils.py             # Utilitaires fichiers
│   ├── text_utils.py             # Utilitaires texte
│   └── session_utils.py          # Session Streamlit
│
├── data/
│   ├── vectorstore/              # Qdrant (créé automatiquement)
│   └── knowledge_base/           # Cours à indexer
│       ├── seconde/
│       ├── premiere/
│       ├── terminale/
│       └── transversal/
│
└── assets/
    ├── styles.css
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

## 🎯 Exemples de questions

```
"Explique-moi le théorème de Pythagore avec des exemples"
"Soit f(x) = 2x² - 3x + 1, résous f(x) = 0"
"C'est quoi une dérivée ?"
"Trace un triangle rectangle de côtés 3, 4 et 5"
```

## 🐛 Dépannage

**`ModuleNotFoundError: aifc`** → Utilise Python >= 3.12 et `SpeechRecognition >= 3.11.0`

**Connexion LLM échouée** → Vérifie que `MIN_AI_API_KEY` est bien défini dans les secrets Streamlit Cloud

**Qdrant vide après redéploiement** → La base vectorielle est locale, elle se recrée à chaque déploiement à partir des fichiers `data/knowledge_base/`

## 👨‍💻 Auteur

Développé par **Kaydan Tech**
