"""
Utilitaires pour la gestion des fichiers.
Fonctions helper pour sauvegarder, charger et manipuler des fichiers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import shutil
from utils.logger import get_logger

logger = get_logger(__name__)

def save_json(data: Dict[Any, Any], filepath: Path) -> bool:
    """
    Sauvegarde des données en JSON.
    
    Args:
        data: Données à sauvegarder
        filepath: Chemin du fichier
        
    Returns:
        True si succès, False sinon
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error("Erreur sauvegarde JSON (%s) : %s", filepath, e)
        return False

def load_json(filepath: Path) -> Optional[Dict[Any, Any]]:
    """
    Charge des données depuis un JSON.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Données chargées ou None si erreur
    """
    try:
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error("Erreur chargement JSON (%s) : %s", filepath, e)
        return None

def ensure_directory(directory: Path) -> bool:
    """
    S'assure qu'un répertoire existe.
    
    Args:
        directory: Chemin du répertoire
        
    Returns:
        True si le répertoire existe ou a été créé
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error("Erreur création répertoire (%s) : %s", directory, e)
        return False

def get_file_size_mb(filepath: Path) -> float:
    """
    Retourne la taille d'un fichier en MB.
    
    Args:
        filepath: Chemin du fichier
        
    Returns:
        Taille en MB
    """
    try:
        return filepath.stat().st_size / (1024 * 1024)
    except:
        return 0.0

def list_files_in_directory(
    directory: Path, 
    extensions: Optional[List[str]] = None
) -> List[Path]:
    """
    Liste tous les fichiers d'un répertoire.
    
    Args:
        directory: Répertoire à scanner
        extensions: Liste d'extensions à filtrer (ex: ['.pdf', '.docx'])
        
    Returns:
        Liste des fichiers trouvés
    """
    if not directory.exists():
        return []
    
    files = []
    for file in directory.iterdir():
        if file.is_file():
            if extensions is None or file.suffix.lower() in extensions:
                files.append(file)
    
    return sorted(files)

def clean_filename(filename: str) -> str:
    """
    Nettoie un nom de fichier (enlève les caractères spéciaux).
    
    Args:
        filename: Nom de fichier original
        
    Returns:
        Nom nettoyé
    """
    import re
    # Garder uniquement lettres, chiffres, tirets et underscores
    cleaned = re.sub(r'[^\w\s-]', '', filename)
    cleaned = re.sub(r'[-\s]+', '-', cleaned)
    return cleaned.strip('-')

def create_backup(filepath: Path) -> bool:
    """
    Crée une sauvegarde d'un fichier.
    
    Args:
        filepath: Fichier à sauvegarder
        
    Returns:
        True si succès
    """
    try:
        if not filepath.exists():
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = filepath.parent / f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
        shutil.copy2(filepath, backup_path)
        return True
    except Exception as e:
        logger.error("Erreur création backup (%s) : %s", filepath, e)
        return False

def get_directory_size(directory: Path) -> float:
    """
    Calcule la taille totale d'un répertoire en MB.
    
    Args:
        directory: Répertoire à analyser
        
    Returns:
        Taille totale en MB
    """
    total_size = 0
    try:
        for file in directory.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
    except:
        pass
    
    return total_size / (1024 * 1024)

def read_text_file(filepath: Path, encoding: str = 'utf-8') -> Optional[str]:
    """
    Lit un fichier texte avec gestion des encodages.
    
    Args:
        filepath: Chemin du fichier
        encoding: Encodage à utiliser
        
    Returns:
        Contenu du fichier ou None
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        # Essayer avec un autre encodage
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
        except:
            return None
    except Exception as e:
        logger.error("Erreur lecture fichier (%s) : %s", filepath, e)
        return None

def write_text_file(filepath: Path, content: str, encoding: str = 'utf-8') -> bool:
    """
    Écrit dans un fichier texte.
    
    Args:
        filepath: Chemin du fichier
        content: Contenu à écrire
        encoding: Encodage à utiliser
        
    Returns:
        True si succès
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        logger.error("Erreur écriture fichier (%s) : %s", filepath, e)
        return False