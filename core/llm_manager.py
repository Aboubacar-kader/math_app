"""
Gestionnaire du modèle de langage (LLM).
Utilise l'API 1min.ai (GPT-4o) pour les réponses.
Utilise HuggingFace sentence-transformers pour les embeddings (local, gratuit).
"""

import requests
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, Tuple
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def call_1minai(system_prompt: str, user_content: str, retries: int = 2) -> str:
    """
    Appelle l'API 1min.ai avec system + user combinés.
    Retourne le texte de la réponse. Retry automatique sur rate limit (429).
    """
    import time
    url = f"{settings.MIN_AI_BASE_URL}/api/chat-with-ai"
    headers = {
        "Content-Type": "application/json",
        "API-KEY": settings.MIN_AI_API_KEY,
    }
    full_prompt = f"{system_prompt}\n\n{user_content}" if system_prompt else user_content
    payload = {
        "type": "CHAT_WITH_AI",
        "model": settings.MIN_AI_MODEL,
        "promptObject": {
            "prompt": full_prompt,
            "isMixed": False,
            "webSearch": False,
        },
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)

            # Rate limit → attendre et réessayer
            if response.status_code == 429:
                wait = 3 * (attempt + 1)
                time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            # Extraire le texte selon le format de la réponse
            result_obj = data.get("aiRecord", {}).get("aiRecordDetail", {}).get("resultObject")
            if isinstance(result_obj, list) and result_obj:
                return result_obj[0]
            elif isinstance(result_obj, str) and result_obj:
                return result_obj
            else:
                # Log interne uniquement — ne pas exposer le contenu de la réponse
                logger.error("Format réponse inattendu : %s", str(data)[:300])
                raise ValueError("Format de réponse inattendu")

        except requests.HTTPError as e:
            # Log interne avec le code HTTP, sans exposer le corps de la réponse
            logger.warning("HTTP %s — tentative %d", e.response.status_code, attempt + 1)
            if attempt < retries:
                time.sleep(2)
                continue
            break
        except Exception as e:
            logger.warning("Erreur tentative %d : %s", attempt + 1, type(e).__name__)
            if attempt < retries:
                time.sleep(2)
                continue
            break

    raise RuntimeError(f"Service LLM indisponible (après {retries + 1} tentatives)")


class LLMManager:
    """Gestionnaire centralisé : embeddings locaux + appels 1min.ai"""

    def __init__(self):
        self._embeddings: Optional[HuggingFaceEmbeddings] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDINGS_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def generate_response(self, prompt: str) -> str:
        try:
            return call_1minai("", prompt)
        except Exception as e:
            return f"❌ Erreur lors de la génération : {str(e)}"

    def stream_response(self, prompt: str):
        """Simule le streaming en retournant la réponse complète en un chunk."""
        try:
            result = call_1minai("", prompt)
            yield result
        except Exception as e:
            yield f"❌ Erreur : {str(e)}"

    # ============================================================
    # CLASSIFICATION INTELLIGENTE DES REQUÊTES (inchangée)
    # ============================================================

    def classify_query(self, question: str) -> Tuple[str, str]:
        """
        Classification 100% par mots-clés — aucun appel LLM.
        """
        question_lower = question.lower().strip()

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

        conv_keywords = [
            'salut', 'bonjour', 'hello', 'coucou', 'bonsoir', 'hi', 'hey',
            'merci', 'thanks', 'au revoir', 'bye', 'à bientôt', 'ciao',
            'comment ça va', 'comment ca va', 'ça va', 'ca va',
            'besoin d\'aide', 'aide-moi', 'aide moi',
        ]
        for keyword in conv_keywords:
            if keyword in question_lower:
                return "CONVERSATION", f"Conversationnel: '{keyword}'"

        if question_lower in {'ok', 'okay', "d'accord", 'dacord', 'compris',
                               'oui', 'ouais', 'non', 'nan', 'nope', 'super',
                               'cool', 'parfait', 'génial', 'bien'}:
            return "CONVERSATION", "Réponse courte"

        return "MATH_RAG", "Défaut"

    # ============================================================
    # CLASSIFICATION DE L'INTENTION MATHÉMATIQUE (inchangée)
    # ============================================================

    def classify_math_intent(self, question: str, rag_function_type: str = "query") -> str:
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
    # RÉPONSES CONVERSATIONNELLES (inchangées)
    # ============================================================

    def get_conversation_response(self, question: str) -> str:
        question_lower = question.lower().strip()

        if any(word in question_lower for word in ['salut', 'bonjour', 'hello', 'coucou', 'hi', 'hey', 'bonsoir']):
            return """Salut ! 👋

Je suis **IntelliMath**, ton assistant en mathématiques de lycée.

**💡 Je peux t'aider à :**
• Comprendre des théorèmes et définitions
• Résoudre des exercices pas à pas
• Expliquer des concepts mathématiques

**Pose-moi une question mathématique !** 📐✨"""

        elif any(phrase in question_lower for phrase in ['comment ça va', 'comment ca va', 'ça va', 'ca va', 'comment vas']):
            return """Je vais bien, merci ! 😊

Je suis prêt à t'aider en mathématiques.

**Quelle question as-tu ?** 📚"""

        elif any(phrase in question_lower for phrase in ['besoin d\'aide', 'besoin daide', 'aide-moi', 'aide moi', 'peux-tu m\'aider', 'tu peux m\'aider']):
            return """D'accord ! Je suis là pour t'aider. 😊

**Quelle est ta question en mathématiques ?**

**Exemples :**
• "Explique le théorème de Pythagore"
• "Qu'est-ce qu'une dérivée ?"
• "Résous x² - 5x + 6 = 0"

**Pose ta question !** 📐"""

        elif any(phrase in question_lower for phrase in ['ne comprends pas', 'ne comprend pas', 'comprends rien', 'difficile', 'compliqué', 'complique']):
            return """Pas de problème ! Je vais t'expliquer clairement. 😊

**Sur quel sujet en mathématiques as-tu besoin d'aide ?**

**Exemple :** "Je ne comprends pas les dérivées"

📐"""

        elif any(word in question_lower for word in ['merci', 'thanks', 'thank you']):
            return """De rien ! 😊

N'hésite pas si tu as d'autres questions en mathématiques ! 💪📐"""


        elif any(word in question_lower for word in ['au revoir', 'bye', 'à bientôt', 'a bientot', 'ciao']):
            return """À bientôt ! 👋

Bonne continuation dans tes études ! 📚✨"""

        elif question_lower in ['ok', 'okay', 'd\'accord', 'dacord', 'compris', 'oui', 'ouais', 'non', 'nan', 'nope']:
            return """Parfait ! 👍

Si tu as une question en mathématiques, je suis là ! 📐"""

        else:
            return """Je peux t'aider en mathématiques ! 📐

**Que veux-tu savoir ?**

• "Explique un théorème"
• "Résous un exercice"
• "Définis un concept"

**Pose ta question !** 😊"""

    def get_out_of_scope_response(self) -> str:
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
