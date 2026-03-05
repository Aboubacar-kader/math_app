"""
Service RAG - IntelliMath
Format structuré avec vérifications anti-hallucination strictes.
"""

from typing import List, Tuple, Dict, Generator
from core.vectorstore_manager import vectorstore_manager
from config.settings import settings
import ollama


class RAGService:
    """Service RAG avec réponses fiables, strictes et format structuré."""

    def __init__(self):
        self.last_sources = []
        self.ollama_model = settings.OLLAMA_MODEL

        self.min_relevance_score = 0.6
        self.max_documents = 3

        # Options Ollama pour GTX 1660 Ti (6GB VRAM)
        # num_ctx=4096 → KV cache ~536MB, stable sur 6GB avec Mistral 7B Q4 (~4.1GB)
        self._ollama_options = {
            'num_ctx': 4096,
            'num_predict': 2048,
        }

        # Options pour la résolution d'exercices longs (réponse plus longue autorisée)
        self._exercise_options = {
            'num_ctx': 6144,
            'num_predict': 3072,
        }

    # ════════════════════════════════════════════════════════
    # HELPER OLLAMA CHAT
    # ════════════════════════════════════════════════════════

    def _call_ollama(self, system_prompt: str, user_content: str, stream: bool = False, long: bool = False):
        """
        Appelle Ollama via l'API chat (system/user roles séparés).
        long=True : utilise _exercise_options pour les réponses longues (exercices complets).
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ]
        options = self._exercise_options if long else self._ollama_options
        if stream:
            return ollama.chat(
                model=self.ollama_model,
                messages=messages,
                stream=True,
                options=options
            )
        else:
            response = ollama.chat(
                model=self.ollama_model,
                messages=messages,
                stream=False,
                options=options
            )
            return response['message']['content']

    # ════════════════════════════════════════════════════════
    # REQUÊTES PRINCIPALES
    # ════════════════════════════════════════════════════════

    def query_with_sources(
        self,
        question: str,
        top_k: int = 5,
        use_streaming: bool = False
    ) -> Tuple[str, List[Dict]]:
        """Requête RAG avec format structuré et réponse fiable."""

        relevant_docs = vectorstore_manager.search(question, top_k=top_k)
        filtered_docs = self._filter_relevant_documents(relevant_docs)

        if not filtered_docs:
            return self._no_info_response(question), []

        best_score = max(doc.get('score', 0) for doc in filtered_docs)
        if best_score < 0.5:
            return self._no_info_response(question), []

        sources = self._prepare_sources(filtered_docs)
        self.last_sources = sources

        context = self._build_context(filtered_docs)
        system_prompt, user_content = self._build_chat_prompt(question, context, filtered_docs)

        if use_streaming:
            def generate():
                response = self._call_ollama(system_prompt, user_content, stream=True)
                for chunk in response:
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        yield content
            return generate(), sources
        else:
            text = self._call_ollama(system_prompt, user_content, stream=False)
            return text, sources

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_streaming: bool = True
    ) -> Generator[str, None, None] | str:
        """Méthode de compatibilité pour streaming."""

        relevant_docs = vectorstore_manager.search(question, top_k=top_k)
        filtered_docs = self._filter_relevant_documents(relevant_docs)

        if not filtered_docs:
            response_text = self._no_info_response(question)
            if use_streaming:
                def empty_gen():
                    yield response_text
                return empty_gen()
            return response_text

        best_score = max(doc.get('score', 0) for doc in filtered_docs)
        if best_score < 0.5:
            response_text = self._no_info_response(question)
            if use_streaming:
                def response_gen():
                    yield response_text
                return response_gen()
            return response_text

        sources = self._prepare_sources(filtered_docs)
        self.last_sources = sources

        context = self._build_context(filtered_docs)
        system_prompt, user_content = self._build_chat_prompt(question, context, filtered_docs)

        if use_streaming:
            response = self._call_ollama(system_prompt, user_content, stream=True)
            for chunk in response:
                content = chunk.get('message', {}).get('content', '')
                if content:
                    yield content
        else:
            return self._call_ollama(system_prompt, user_content, stream=False)

    def get_last_sources(self) -> List[Dict]:
        """Récupère les sources de la dernière requête."""
        return self.last_sources

    # ════════════════════════════════════════════════════════
    # DÉFINITIONS
    # ════════════════════════════════════════════════════════

    def get_definition_with_sources(
        self,
        term: str,
        level: str
    ) -> Tuple[str, List[Dict]]:
        """Obtient une définition avec format structuré et anti-hallucination."""

        query = f"Définition de {term} en mathématiques niveau {level}"
        relevant_docs = vectorstore_manager.search(query, top_k=3)
        filtered_docs = self._filter_relevant_documents(relevant_docs)

        if not filtered_docs:
            return self._no_info_response(f"définition de {term}"), []

        sources = self._prepare_sources(filtered_docs)
        context = self._build_context(filtered_docs)

        system_prompt = f"""Tu es IntelliMath, assistant pédagogique expert en mathématiques de lycée.

DOMAINE : Mathématiques lycée français (Seconde, Première, Terminale) UNIQUEMENT.

⚠️ RÈGLES STRICTES :
1. Lis attentivement les extraits fournis.
2. Vérifie si le terme demandé est effectivement défini dans ces extraits.
3. Si NON → réponds UNIQUEMENT : "Je ne dispose pas de la définition de ce terme dans ma base de connaissances."
4. Si OUI → utilise le format ci-dessous avec une CITATION EXACTE.
5. N'INVENTE PAS de définition absente des extraits.

📝 FORMAT OBLIGATOIRE (si l'information est disponible) :

📘 [Nom exact du concept]

🔹 Définition

[Citation TEXTUELLE et EXACTE depuis les extraits]
[Formules en LaTeX : $...$ inline, $$...$$ pour bloc]

💡 Explication

[Explication simple et accessible pour un lycéen de {level}]

🎯 Exemple concret

[Exemple numérique avec résolution si pertinent]

✨ Points clés

• [Point 1]
• [Point 2]"""

        user_content = f"""📚 EXTRAITS DE LA BASE DE CONNAISSANCES :

{context}

❓ TERME DEMANDÉ : {term}
📊 NIVEAU : {level}

Commence ta réponse avec 📘 :"""

        text = self._call_ollama(system_prompt, user_content)
        return text, sources

    def get_definition(self, term: str, level: str) -> str:
        """Obtient une définition."""
        definition, sources = self.get_definition_with_sources(term, level)
        self.last_sources = sources
        return definition

    # ════════════════════════════════════════════════════════
    # EXERCICES
    # ════════════════════════════════════════════════════════

    def solve_exercise_with_sources(
        self,
        exercise: str,
        level: str
    ) -> Tuple[str, List[Dict]]:
        """Résout un exercice avec le contexte de la base de connaissances."""

        relevant_docs = vectorstore_manager.search(exercise, top_k=5)
        filtered_docs = self._filter_relevant_documents(relevant_docs)

        if not filtered_docs:
            return self._no_info_response(exercise), []

        sources = self._prepare_sources(filtered_docs)
        context = self._build_context(filtered_docs)

        system_prompt = f"""Tu es IntelliMath, professeur de mathématiques de lycée expert.

DOMAINE : Mathématiques lycée français (Seconde, Première, Terminale) UNIQUEMENT.

⚠️ VÉRIFICATION PRÉALABLE :
Si l'exercice n'est PAS un problème de mathématiques de lycée,
réponds UNIQUEMENT : "Ce n'est pas un exercice de mathématiques de lycée."

📝 MÉTHODE DE RÉSOLUTION :

1. 📋 Données : Ce qui est donné et les hypothèses
2. 🎯 Objectif : Ce qu'on cherche à calculer/démontrer
3. 💭 Stratégie : Propriété(s)/théorème(s) à appliquer (avec leur nom)
4. ✏️ Résolution : Étapes détaillées, justifiées, LaTeX pour toutes les formules
   - $formule$ pour inline, $$formule$$ pour bloc
   - Justifie chaque étape par la propriété ou le théorème utilisé
5. ✅ Vérification : Contrôle de cohérence du résultat
6. 📝 Conclusion : Réponse rédigée et encadrée

RÈGLES ABSOLUES :
✓ LaTeX pour TOUTES les expressions mathématiques
✓ Cite le nom du théorème/propriété à chaque étape
✓ N'invente pas de données absentes de l'énoncé"""

        user_content = f"""📚 RAPPELS DE COURS (base de connaissances) :

{context}

❓ EXERCICE : {exercise}
📊 NIVEAU : {level}

Commence la résolution :"""

        text = self._call_ollama(system_prompt, user_content)
        return text, sources

    def solve_exercise(self, exercise: str, level: str) -> str:
        """Résout un exercice."""
        solution, sources = self.solve_exercise_with_sources(exercise, level)
        self.last_sources = sources
        return solution

    # ════════════════════════════════════════════════════════
    # MÉTHODES INTERNES
    # ════════════════════════════════════════════════════════

    def _filter_relevant_documents(self, documents: List[Dict]) -> List[Dict]:
        """Filtre les documents en dessous du seuil de pertinence."""
        if not documents:
            return []
        return [
            doc for doc in documents[:self.max_documents]
            if doc.get('score', 0) >= self.min_relevance_score
        ]

    def _no_info_response(self, question: str) -> str:
        """Réponse stricte quand aucune information pertinente n'est disponible."""
        return f"""### ℹ️ Aucune information disponible

Je n'ai pas d'information sur **« {question} »** dans ma base de connaissances.

**💡 Que faire ?**

1. **Vérifiez l'orthographe** du terme recherché
2. **Reformulez** avec des termes mathématiques précis
3. **Précisez le niveau** (Seconde, Première, Terminale)
4. **Enrichissez** la base en ajoutant des documents dans `data/knowledge_base/`

**Exemples de reformulation :**
• "Théorème de Pythagore" au lieu de "Pythagore"
• "Définition de dérivée" au lieu de "dérivée"
• "Propriétés des suites arithmétiques"
"""

    def _build_chat_prompt(
        self,
        question: str,
        context: str,
        documents: List[Dict]
    ) -> Tuple[str, str]:
        """
        Construit le prompt system + message utilisateur pour ollama.chat().
        Le system prompt contient les règles et le format.
        Le message utilisateur contient le contexte et la question.
        Évite que le modèle reproduise les instructions ou les en-têtes.
        """
        num_sources = len(documents)

        system_prompt = """Tu es IntelliMath, assistant pédagogique EXPERT en mathématiques de lycée.

DOMAINE : Mathématiques lycée français (Seconde, Première, Terminale) UNIQUEMENT.

⚠️ RÈGLES CRITIQUES ANTI-HALLUCINATION :

1. VÉRIFICATION PRÉALABLE :
   - Lis chaque extrait attentivement
   - Compare le sujet des extraits avec la question posée
   - Si les extraits ne couvrent PAS le sujet demandé → réponds UNIQUEMENT :
     "Je n'ai pas d'information sur ce sujet dans ma base de connaissances."

2. PRÉCISION :
   - CITE TEXTUELLEMENT l'énoncé depuis les extraits (ne paraphrase pas)
   - CONSERVE les formules LaTeX exactement telles qu'elles apparaissent
   - Si plusieurs conditions : numérote-les (1. 2. 3.)

3. INTERDICTIONS ABSOLUES :
   ✗ Ne MÉLANGE JAMAIS deux concepts/théorèmes différents
   ✗ N'INVENTE PAS de formules, énoncés ou propriétés absents des extraits
   ✗ Ne fais PAS de réponse approximative si tu n'es pas sûr à 100%
   ✗ N'utilise JAMAIS \\begin{align}, \\begin{equation}, \\hspace ni aucun environnement LaTeX

4. LATEX :
   ✓ Formules inline uniquement avec $...$ — exemple : $u_n$, $f'(x) = 2x$
   ✓ Formules en bloc uniquement avec $$...$$ — exemple : $$f(x) = x^2 - 3x + 2$$

📝 FORMAT OBLIGATOIRE (si l'information est disponible) :

📘 [Nom exact du Théorème / Définition / Propriété]

🔹 Énoncé

[Citation TEXTUELLE et EXACTE depuis les extraits avec $...$ et $$...$$]

💡 Explication claire

[Explication simple et accessible pour un lycéen]

🎯 Exemple d'application

[Exemple concret avec résolution détaillée, LaTeX $...$ et $$...$$]

✨ Points clés

• [Point essentiel 1]
• [Point essentiel 2]
• [Point essentiel 3]"""

        user_content = f"""📚 EXTRAITS DE LA BASE DE CONNAISSANCES ({num_sources} extrait(s)) :

{context}

❓ QUESTION :
{question}

Commence ta réponse avec 📘 :"""

        return system_prompt, user_content

    def _prepare_sources(self, documents: List[Dict]) -> List[Dict]:
        """Prépare les métadonnées de sources."""
        sources = []
        for idx, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            score = doc.get('score', 0)
            sources.append({
                'index': idx,
                'filename': metadata.get('filename', 'Document inconnu'),
                'page': metadata.get('page', 'N/A'),
                'level': metadata.get('level', ''),
                'topic': metadata.get('topic', ''),
                'excerpt': doc.get('text', '')[:200],
                'score': score,
                'relevance': f"{score * 100:.0f}%"
            })
        return sources

    def _build_context(self, documents: List[Dict]) -> str:
        """
        Construit le contexte documentaire injecté dans le message utilisateur.
        Avec ollama.chat(), les en-têtes ne sont plus reproduits dans la réponse
        car le modèle sait distinguer instructions (system) et données (user).
        """
        context_parts = []

        for idx, doc in enumerate(documents, 1):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            level = metadata.get('level', '')

            header = f"--- Extrait {idx}"
            if level:
                header += f" [Niveau {level}]"
            header += " ---"

            truncated_text = text[:800] if len(text) > 800 else text
            context_parts.append(f"{header}\n{truncated_text}\n")

        return "\n".join(context_parts)


# ════════════════════════════════════════════════════════
# INSTANCE SINGLETON
# ════════════════════════════════════════════════════════

rag_service = RAGService()
