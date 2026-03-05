# 🧮 Chatbot Mathématiques Lycée

Un assistant intelligent pour aider les élèves de lycée (Seconde à Terminale) en mathématiques.

## ✨ Fonctionnalités

- **💬 Chat conversationnel** : Posez vos questions en langage naturel
- **📚 Explications de cours** : Obtenez des explications détaillées par chapitre
- **✏️ Résolution d'exercices** : Solutions pas-à-pas avec justifications
- **📖 Définitions & Théorèmes** : Base de connaissances complète
- **🎤 Entrée vocale** : Dictez vos questions
- **🔊 Sortie vocale** : Écoutez les réponses
- **📄 Upload de documents** : Indexez vos cours (PDF, DOCX, TXT, Images)
- **🔍 RAG (Retrieval Augmented Generation)** : Réponses basées sur vos documents

## 🛠️ Technologies utilisées

- **Streamlit** : Interface web
- **LangChain** : Orchestration LLM
- **Ollama** : Exécution locale du LLM
- **Qdrant** : Base vectorielle
- **Tesseract** : OCR pour images
- **Google TTS** : Synthèse vocale

## 📋 Prérequis

- Python 3.9+
- Ollama installé et en cours d'exécution
- Tesseract OCR (optionnel, pour l'extraction de texte des images)

## 🚀 Installation

### 1. Cloner le repository
```bash
git clone https://github.com/votre-username/math-chatbot.git
cd math-chatbot
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Installer et configurer Ollama
```bash
# Télécharger Ollama depuis https://ollama.ai

# Lancer Ollama
ollama serve

# Télécharger le modèle (dans un autre terminal)
ollama pull llama3.2
```

### 5. (Optionnel) Installer Tesseract pour l'OCR

**Sur Ubuntu/Debian :**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-fra
```

**Sur macOS :**
```bash
brew install tesseract tesseract-lang
```

**Sur Windows :**
Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki

### 6. Configuration

Copier le fichier d'exemple et ajuster les paramètres :
```bash
cp .env.example .env
```

Éditer `.env` selon vos besoins.

## 🎮 Utilisation

### Lancer l'application
```bash
streamlit run app.py
```

L'application s'ouvrira dans votre navigateur à `http://localhost:8501`

### Workflow typique

1. **Upload de documents** (optionnel) :
   - Allez dans la sidebar → Upload
   - Glissez vos cours (PDF, DOCX, etc.)
   - Cliquez sur "Traiter les documents"

2. **Poser des questions** :
   - Onglet "Chat" : conversation libre
   - Onglet "Cours" : explications par chapitre
   - Onglet "Exercices" : résolution détaillée
   - Onglet "Définitions" : recherche de termes

3. **Utiliser la voix** (optionnel) :
   - Sidebar → Vocal → Activer le microphone
   - Parlez votre question
   - Activez la lecture automatique pour écouter les réponses

## 📁 Structure du projet
```
math-chatbot/
│
├── app.py                      # Point d'entrée
├── requirements.txt
├── .env.example
│
├── config/
│   └── settings.py            # Configuration
│
├── core/
│   ├── llm_manager.py         # Gestion LLM
│   ├── vectorstore_manager.py # Base vectorielle
│   └── prompts.py             # Prompts système
│
├── services/
│   ├── document_processor.py  # Traitement documents
│   ├── rag_service.py         # Service RAG
│   ├── voice_service.py       # Vocal
│   └── math_solver.py         # Logique maths
│
├── ui/
│   ├── sidebar.py             # Barre latérale
│   ├── chat_interface.py      # Interface chat
│   ├── tabs.py                # Onglets
│   └── components.py          # Composants UI
│
├── utils/
│   ├── file_utils.py          # Utilitaires fichiers
│   ├── text_utils.py          # Utilitaires texte
│   └── session_utils.py       # Session Streamlit
│
├── data/                       # Données (créé automatiquement)
│   ├── vectorstore/
│   ├── uploads/
│   └── conversations/
│
└── assets/
    └── styles.css             # Styles personnalisés
```

## 🎯 Exemples d'utilisation

### Demander une explication de cours
```
"Explique-moi le théorème de Pythagore avec des exemples"
```

### Résoudre un exercice
```
Soit f(x) = 2x² - 3x + 1
1) Calculer f(0), f(1), f(2)
2) Résoudre f(x) = 0
3) Étudier le signe de f
```

### Chercher une définition
```
"C'est quoi une dérivée ?"
```

### Demander une démonstration
```
"Démontre le théorème des valeurs intermédiaires"
```

## 🔧 Personnalisation

### Changer le modèle LLM

Dans `.env` :
```
OLLAMA_MODEL=mistral  # ou autre modèle
```

### Ajuster les paramètres RAG

Dans `.env` :
```
CHUNK_SIZE=1500
CHUNK_OVERLAP=300
```

### Modifier les prompts

Éditez `core/prompts.py` pour personnaliser les instructions du LLM.

## 🐛 Dépannage

### Ollama ne se connecte pas
```bash
# Vérifier qu'Ollama tourne
curl http://localhost:11434

# Relancer Ollama
ollama serve
```

### Erreur d'OCR sur les images
```bash
# Vérifier l'installation de Tesseract
tesseract --version

# Installer les langues
# Ubuntu: sudo apt-get install tesseract-ocr-fra
```

### Problème de mémoire

Réduire `CHUNK_SIZE` dans `.env` ou utiliser un modèle plus petit.

## 📚 Documentation

- [Documentation Streamlit](https://docs.streamlit.io)
- [Documentation LangChain](https://python.langchain.com)
- [Documentation Ollama](https://ollama.ai/docs)
- [Documentation Qdrant](https://qdrant.tech/documentation)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/amelioration`)
3. Commit vos changements (`git commit -m 'Ajout fonctionnalité'`)
4. Push vers la branche (`git push origin feature/amelioration`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👨‍💻 Auteur

Votre Nom - [@votre_twitter](https://twitter.com/votre_twitter)

## 🙏 Remerciements

- Anthropic pour Claude
- Communauté Streamlit
- Équipe LangChain
- Ollama team

---

**⭐ Si ce projet vous aide, n'hésitez pas à lui donner une étoile !**