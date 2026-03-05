"""
Gestionnaire du modèle de langage (LLM).
Initialise et gère les interactions avec Ollama.
Inclut la classification intelligente des requêtes.
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from typing import Optional, Tuple
from config.settings import settings


class LLMManager:
    """Gestionnaire centralisé du LLM avec classification intelligente"""
    
    def __init__(self):
        self._llm: Optional[ChatOllama] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
    
    @property
    def llm(self) -> ChatOllama:
        """
        Retourne l'instance du LLM (lazy loading).
        Utilise le cache de Streamlit pour éviter de recharger le modèle.
        """
        if self._llm is None:
            self._llm = self._initialize_llm()
        return self._llm
    
    @property
    def embeddings(self) -> OllamaEmbeddings:
        """
        Retourne l'instance des embeddings (lazy loading).
        """
        if self._embeddings is None:
            self._embeddings = self._initialize_embeddings()
        return self._embeddings
    
    @staticmethod
    def _initialize_llm() -> ChatOllama:
        """Initialise le modèle LLM (pas de cache Streamlit — évite les instances périmées)"""
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE,
            num_ctx=settings.OLLAMA_NUM_CTX,
            num_predict=settings.OLLAMA_NUM_PREDICT,
        )

    @staticmethod
    def _initialize_embeddings() -> OllamaEmbeddings:
        """Initialise le modèle d'embeddings (pas de cache Streamlit)"""
        return OllamaEmbeddings(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    
    def generate_response(self, prompt: str) -> str:
        """
        Génère une réponse à partir d'un prompt.
        
        Args:
            prompt: Le prompt à envoyer au LLM
            
        Returns:
            La réponse générée
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"❌ Erreur lors de la génération : {str(e)}"
    
    def stream_response(self, prompt: str):
        """
        Génère une réponse en streaming (mot par mot).
        Utilisé pour une meilleure UX dans le chat.
        
        Args:
            prompt: Le prompt à envoyer au LLM
            
        Yields:
            Chunks de texte au fur et à mesure
        """
        try:
            for chunk in self.llm.stream(prompt):
                yield chunk.content
        except Exception as e:
            yield f"❌ Erreur : {str(e)}"
    
    # ============================================================
    # CLASSIFICATION INTELLIGENTE DES REQUÊTES
    # ============================================================
    
    def classify_query(self, question: str) -> Tuple[str, str]:
        """
        Classification 100% par mots-clés — aucun appel LLM.
        Évite le rechargement du modèle Ollama qui causait les CUDA errors.
        """
        question_lower = question.lower().strip()

        # ── Mots-clés mathématiques → MATH_RAG ──────────────────────
        math_keywords = [
            'théorème', 'théoreme', 'dérivée', 'derivee', 'dérive',
            'intégrale', 'integrale', 'intègre', 'limite', 'limites',
            'suite', 'suites', 'fonction', 'fonctions',
            'équation', 'equation', 'equations',
            'probabilité', 'probabilite', 'probabilités',
            'statistique', 'statistiques', 'géométrie', 'geometrie',
            'trigonométrie', 'trigonometrie', 'vecteur', 'vecteurs',
            'matrice', 'matrices', 'polynôme', 'polynome',
            'logarithme', 'exponentielle', 'complexe', 'complexes',
            'primitive', 'primitives', 'discriminant', 'racine',
            'pythagore', 'thalès', 'thales', 'gendarmes', 'rolle',
            'résous', 'resous', 'résoudre', 'resoudre',
            'calcule', 'calculer', 'démontre', 'demontre', 'démontrer',
            'prouve', 'prouver', 'factorise', 'développe', 'simplifie',
            'détermine', 'trouve', 'trouver', 'exercice', 'problème',
            'convexe', 'concave', 'asymptote', 'continuité', 'dérivable',
            'homothétie', 'vecteur', 'scalaire', 'matrice', 'angle',
            'sinus', 'cosinus', 'tangente', 'radian', 'degré',
            'aire', 'volume', 'périmètre', 'hypoténuse',
            'médiane', 'moyenne', 'variance', 'écart-type', 'espérance',
        ]
        for keyword in math_keywords:
            if keyword in question_lower:
                return "MATH_RAG", f"Mot-clé: '{keyword}'"

        # ── Mots-clés conversationnels → CONVERSATION ────────────────
        conv_keywords = [
            'salut', 'bonjour', 'hello', 'coucou', 'bonsoir', 'hi', 'hey',
            'merci', 'thanks', 'au revoir', 'bye', 'à bientôt', 'ciao',
            'comment ça va', 'comment ca va', 'ça va', 'ca va',
            'besoin d\'aide', 'aide-moi', 'aide moi',
        ]
        for keyword in conv_keywords:
            if keyword in question_lower:
                return "CONVERSATION", f"Conversationnel: '{keyword}'"

        # Réponses courtes typiquement conversationnelles
        if question_lower in {'ok', 'okay', "d'accord", 'dacord', 'compris',
                               'oui', 'ouais', 'non', 'nan', 'nope', 'super',
                               'cool', 'parfait', 'génial', 'bien'}:
            return "CONVERSATION", "Réponse courte"

        # ── Par défaut : MATH_RAG (pas d'appel LLM) ──────────────────
        return "MATH_RAG", "Défaut"
    
    # ============================================================
    # CLASSIFICATION DE L'INTENTION MATHÉMATIQUE
    # ============================================================

    def classify_math_intent(self, question: str, rag_function_type: str = "query") -> str:
        """
        Classifie l'intention mathématique dans une requête MATH_RAG.

        Returns:
            "EXERCICE" : résoudre, calculer, démontrer, appliquer...
            "COURS"    : définition, théorème, propriété, expliquer...
        """
        if rag_function_type == "exercise":
            return "EXERCICE"

        question_lower = question.lower()

        exercice_keywords = [
            'résous', 'resous', 'résoudre', 'resoudre',
            'calcule', 'calculer',
            'démontre', 'demontre', 'démontrer',
            'prouve', 'prouver',
            'trouve', 'trouver',
            'détermine', 'determiner', 'déterminer',
            'exercice', 'problème', 'probleme',
            'corrige', 'corriger', 'correction',
            'vérifie', 'verifier',
            'applique', 'appliquer',
            'simplifie', 'simplifier',
            'factorise', 'factoriser',
            'développe', 'developpe',
        ]

        for kw in exercice_keywords:
            if kw in question_lower:
                return "EXERCICE"

        return "COURS"

    # ============================================================
    # RÉPONSES CONVERSATIONNELLES
    # ============================================================

    def get_conversation_response(self, question: str) -> str:
        """
        Génère une réponse conversationnelle appropriée.
        
        Args:
            question: Question conversationnelle de l'utilisateur
        
        Returns:
            Réponse conversationnelle
        """
        question_lower = question.lower().strip()
        
        # Salutations
        if any(word in question_lower for word in ['salut', 'bonjour', 'hello', 'coucou', 'hi', 'hey', 'bonsoir']):
            return """Salut ! 👋

Je suis **IntelliMath**, ton assistant en mathématiques de lycée.

**💡 Je peux t'aider à :**
• Comprendre des théorèmes et définitions
• Résoudre des exercices pas à pas
• Expliquer des concepts mathématiques

**Pose-moi une question mathématique !** 📐✨"""
        
        # Comment ça va
        elif any(phrase in question_lower for phrase in ['comment ça va', 'comment ca va', 'ça va', 'ca va', 'comment vas']):
            return """Je vais bien, merci ! 😊

Je suis prêt à t'aider en mathématiques.

**Quelle question as-tu ?** 📚"""
        
        # Demandes d'aide génériques
        elif any(phrase in question_lower for phrase in ['besoin d\'aide', 'besoin daide', 'aide-moi', 'aide moi', 'peux-tu m\'aider', 'tu peux m\'aider']):
            return """D'accord ! Je suis là pour t'aider. 😊

**Quelle est ta question en mathématiques ?**

**Exemples :**
• "Explique le théorème de Pythagore"
• "Qu'est-ce qu'une dérivée ?"
• "Résous x² - 5x + 6 = 0"

**Pose ta question !** 📐"""
        
        # Je ne comprends pas / C'est difficile
        elif any(phrase in question_lower for phrase in ['ne comprends pas', 'ne comprend pas', 'comprends rien', 'difficile', 'compliqué', 'complique']):
            return """Pas de problème ! Je vais t'expliquer clairement. 😊

**Sur quel sujet en mathématiques as-tu besoin d'aide ?**

**Exemple :** "Je ne comprends pas les dérivées"

📐"""
        
        # Remerciements
        elif any(word in question_lower for word in ['merci', 'thanks', 'thank you']):
            return """De rien ! 😊

N'hésite pas si tu as d'autres questions en mathématiques ! 💪📐"""
        
        # Au revoir
        elif any(word in question_lower for word in ['au revoir', 'bye', 'à bientôt', 'a bientot', 'ciao']):
            return """À bientôt ! 👋

Bonne continuation dans tes études ! 📚✨"""
        
        # OK / D'accord / Oui / Non
        elif question_lower in ['ok', 'okay', 'd\'accord', 'dacord', 'compris', 'oui', 'ouais', 'non', 'nan', 'nope']:
            return """Parfait ! 👍

Si tu as une question en mathématiques, je suis là ! 📐"""
        
        # Défaut
        else:
            return """Je peux t'aider en mathématiques ! 📐

**Que veux-tu savoir ?**

• "Explique un théorème"
• "Résous un exercice"
• "Définis un concept"

**Pose ta question !** 😊"""
    
    def get_out_of_scope_response(self) -> str:
        """
        Génère une réponse pour les questions hors du domaine (mathématiques lycée).
        
        Returns:
            Message de refus poli
        """
        return """Je suis spécialisé en **mathématiques de lycée** uniquement. 📐

**Mon domaine :**
• Seconde, Première, Terminale
• Fonctions, dérivées, intégrales
• Suites, limites, probabilités
• Géométrie, trigonométrie
• Équations, théorèmes

**Pose-moi une question de maths lycée !** 😊"""


# Instance globale
llm_manager = LLMManager()
