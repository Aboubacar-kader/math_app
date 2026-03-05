"""
Service spécialisé pour la résolution de problèmes mathématiques.
Fournit des fonctions utilitaires pour parser et analyser les questions.
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum

class MathTopic(Enum):
    """Énumération des sujets mathématiques"""
    FONCTIONS = "fonctions"
    DERIVEES = "dérivées"
    INTEGRALES = "intégrales"
    SUITES = "suites"
    PROBAS = "probabilités"
    GEOMETRIE = "géométrie"
    TRIGO = "trigonométrie"
    EQUATIONS = "équations"
    LIMITES = "limites"
    VECTEURS = "vecteurs"
    COMPLEXES = "nombres complexes"
    STATS = "statistiques"
    AUTRE = "autre"

class MathLevel(Enum):
    """Niveaux scolaires"""
    SECONDE = "Seconde"
    PREMIERE = "Première"
    TERMINALE = "Terminale"
    SUPERIEUR = "Supérieur"

class MathSolver:
    """Analyseur et résolveur de problèmes mathématiques"""
    
    # Mots-clés par sujet
    TOPIC_KEYWORDS = {
        MathTopic.FONCTIONS: ['fonction', 'f(x)', 'domaine', 'image', 'antécédent'],
        MathTopic.DERIVEES: ['dérivée', "dériver", 'tangente', "f'", 'nombre dérivé'],
        MathTopic.INTEGRALES: ['intégrale', 'primitive', 'aire', '∫'],
        MathTopic.SUITES: ['suite', 'u_n', 'récurrence', 'arithmétique', 'géométrique'],
        MathTopic.PROBAS: ['probabilité', 'événement', 'loi', 'espérance', 'variance'],
        MathTopic.GEOMETRIE: ['cercle', 'triangle', 'point', 'droite', 'plan', 'coordonnées'],
        MathTopic.TRIGO: ['cos', 'sin', 'tan', 'trigonométrie', 'angle'],
        MathTopic.EQUATIONS: ['équation', 'résoudre', 'inconnue', 'système'],
        MathTopic.LIMITES: ['limite', 'lim', 'infini', 'asymptote'],
        MathTopic.VECTEURS: ['vecteur', 'norme', 'produit scalaire', 'colinéaire'],
        MathTopic.COMPLEXES: ['complexe', 'imaginaire', 'module', 'argument', 'i'],
        MathTopic.STATS: ['moyenne', 'médiane', 'écart-type', 'quartile']
    }
    
    def __init__(self):
        pass
    
    def detect_topic(self, text: str) -> MathTopic:
        """
        Détecte le sujet mathématique d'une question.
        
        Args:
            text: Le texte de la question
            
        Returns:
            Le sujet détecté
        """
        text_lower = text.lower()
        scores = {}
        
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[topic] = score
        
        # Retourner le sujet avec le meilleur score
        best_topic = max(scores, key=scores.get)
        
        return best_topic if scores[best_topic] > 0 else MathTopic.AUTRE
    
    def detect_level(self, text: str) -> MathLevel:
        """
        Devine le niveau scolaire basé sur le contenu.
        
        Args:
            text: Le texte de la question
            
        Returns:
            Le niveau estimé
        """
        text_lower = text.lower()
        
        # Mots-clés par niveau
        if any(word in text_lower for word in ['intégrale', 'primitive', 'ln', 'exponentielle']):
            return MathLevel.TERMINALE
        elif any(word in text_lower for word in ['dérivée', 'suite', 'limite']):
            return MathLevel.PREMIERE
        elif any(word in text_lower for word in ['fonction', 'équation du second degré']):
            return MathLevel.SECONDE
        else:
            return MathLevel.SECONDE  # Par défaut
    
    def extract_equations(self, text: str) -> List[str]:
        """
        Extrait les équations mathématiques du texte.
        
        Args:
            text: Le texte contenant des équations
            
        Returns:
            Liste des équations trouvées
        """
        equations = []
        
        # Chercher les expressions LaTeX
        latex_patterns = [
            r'\$\$([^$]+)\$\$',  # Block
            r'\$([^$]+)\$',       # Inline
        ]
        
        for pattern in latex_patterns:
            matches = re.findall(pattern, text)
            equations.extend(matches)
        
        # Chercher les expressions simples (x=..., y=..., etc.)
        simple_pattern = r'[a-zA-Z]\s*=\s*[^,.\n]+'
        simple_matches = re.findall(simple_pattern, text)
        equations.extend(simple_matches)
        
        return list(set(equations))  # Éliminer les doublons
    
    def is_exercise(self, text: str) -> bool:
        """
        Détermine si le texte est un exercice à résoudre.
        
        Args:
            text: Le texte à analyser
            
        Returns:
            True si c'est un exercice
        """
        exercise_indicators = [
            'résoudre', 'calculer', 'démontrer', 'montrer',
            'déterminer', 'trouver', 'soit', 'on considère',
            'exercice', 'problème', 'question'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in exercise_indicators)
    
    def is_definition_request(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Détecte si c'est une demande de définition.
        
        Args:
            text: Le texte à analyser
            
        Returns:
            Tuple (est_definition, terme_à_définir)
        """
        patterns = [
            r"(?:qu'est[- ]ce qu'|c'est quoi|défin(?:is|it))\s+(?:un(?:e)?|le|la|les)?\s*([a-zàâäéèêëïîôùûüÿæœç\s-]+)",
            r"(?:définition|sens|signification)\s+(?:de|du|d')?\s*([a-zàâäéèêëïîôùûüÿæœç\s-]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                term = match.group(1).strip()
                return True, term
        
        return False, None
    
    def format_mathematical_response(
        self, 
        question: str, 
        solution: str,
        topic: Optional[MathTopic] = None
    ) -> str:
        """
        Formate une réponse mathématique avec structure.
        
        Args:
            question: La question posée
            solution: La solution générée
            topic: Le sujet (optionnel)
            
        Returns:
            Réponse formatée
        """
        if topic:
            header = f"### 📐 {topic.value.capitalize()}\n\n"
        else:
            header = ""
        
        formatted = f"""{header}**Question :** {question}

---

{solution}

---

💡 *N'hésite pas à demander des précisions !*
"""
        return formatted

# Instance globale
math_solver = MathSolver()