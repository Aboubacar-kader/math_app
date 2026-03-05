"""
Utilitaires pour le traitement de texte.
Nettoyage, formatage et manipulation de chaînes.
"""

import re
from typing import List, Tuple

def clean_text(text: str) -> str:
    """
    Nettoie un texte (espaces multiples, sauts de ligne, etc.).
    
    Args:
        text: Texte à nettoyer
        
    Returns:
        Texte nettoyé
    """
    # Remplacer les espaces multiples par un seul
    text = re.sub(r'\s+', ' ', text)
    
    # Enlever les espaces en début et fin
    text = text.strip()
    
    # Normaliser les sauts de ligne
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text

def extract_latex_formulas(text: str) -> List[str]:
    """
    Extrait toutes les formules LaTeX d'un texte.
    
    Args:
        text: Texte contenant du LaTeX
        
    Returns:
        Liste des formules trouvées
    """
    formulas = []
    
    # Formules block ($$...$$)
    block_pattern = r'\$\$(.+?)\$\$'
    block_formulas = re.findall(block_pattern, text, re.DOTALL)
    formulas.extend(block_formulas)
    
    # Formules inline ($...$)
    inline_pattern = r'\$([^$]+)\$'
    inline_formulas = re.findall(inline_pattern, text)
    formulas.extend(inline_formulas)
    
    return formulas

def count_words(text: str) -> int:
    """
    Compte le nombre de mots dans un texte.
    
    Args:
        text: Texte à analyser
        
    Returns:
        Nombre de mots
    """
    words = text.split()
    return len(words)

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Tronque un texte à une longueur maximale.
    
    Args:
        text: Texte à tronquer
        max_length: Longueur maximale
        suffix: Suffixe à ajouter si tronqué
        
    Returns:
        Texte tronqué
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def split_into_sentences(text: str) -> List[str]:
    """
    Découpe un texte en phrases.
    
    Args:
        text: Texte à découper
        
    Returns:
        Liste de phrases
    """
    # Pattern simple pour les phrases
    sentences = re.split(r'[.!?]+', text)
    
    # Nettoyer et filtrer les phrases vides
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """
    Met en évidence des mots-clés dans un texte (avec markdown).
    
    Args:
        text: Texte original
        keywords: Liste de mots-clés à mettre en évidence
        
    Returns:
        Texte avec mots-clés en gras
    """
    for keyword in keywords:
        # Insensible à la casse
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        text = pattern.sub(f"**{keyword}**", text)
    
    return text

def remove_special_characters(text: str, keep_spaces: bool = True) -> str:
    """
    Enlève les caractères spéciaux d'un texte.
    
    Args:
        text: Texte à nettoyer
        keep_spaces: Garder les espaces
        
    Returns:
        Texte nettoyé
    """
    if keep_spaces:
        pattern = r'[^a-zA-Z0-9\s]'
    else:
        pattern = r'[^a-zA-Z0-9]'
    
    return re.sub(pattern, '', text)

def normalize_whitespace(text: str) -> str:
    """
    Normalise les espaces blancs dans un texte.
    
    Args:
        text: Texte à normaliser
        
    Returns:
        Texte normalisé
    """
    # Remplacer tous les types d'espaces par un espace simple
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_numbers(text: str) -> List[float]:
    """
    Extrait tous les nombres d'un texte.
    
    Args:
        text: Texte à analyser
        
    Returns:
        Liste de nombres trouvés
    """
    # Pattern pour nombres décimaux et entiers
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    
    numbers = []
    for match in matches:
        try:
            numbers.append(float(match))
        except:
            pass
    
    return numbers

def format_large_number(number: float, decimals: int = 2) -> str:
    """
    Formate un grand nombre pour l'affichage.
    
    Args:
        number: Nombre à formater
        decimals: Nombre de décimales
        
    Returns:
        Nombre formaté avec séparateurs
    """
    return f"{number:,.{decimals}f}".replace(",", " ")

def contains_math_notation(text: str) -> bool:
    """
    Vérifie si un texte contient des notations mathématiques.
    
    Args:
        text: Texte à vérifier
        
    Returns:
        True si contient des notations mathématiques
    """
    math_indicators = [
        r'\$',  # LaTeX
        r'[∫∑∏√∞]',  # Symboles mathématiques Unicode
        r'[α-ωΑ-Ω]',  # Lettres grecques
        r'[≤≥≠≈±]',  # Symboles de comparaison
        r'\^',  # Exposant
        r'_',  # Indice (dans contexte math)
    ]
    
    for pattern in math_indicators:
        if re.search(pattern, text):
            return True
    
    return False

def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extrait les blocs de code d'un texte markdown.
    
    Args:
        text: Texte markdown
        
    Returns:
        Liste de tuples (langage, code)
    """
    pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    return [(lang or 'text', code.strip()) for lang, code in matches]