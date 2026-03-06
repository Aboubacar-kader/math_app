"""
Initialisation automatique de la base de connaissances.
Ce module charge silencieusement les cours au démarrage.
L'utilisateur ne voit rien, tout se passe en arrière-plan.

VERSION AMÉLIORÉE avec détection automatique du niveau
"""

from pathlib import Path
from typing import List, Dict, Any
import json
import hashlib
from utils.logger import get_logger

logger = get_logger(__name__)

# Répertoires des cours
KNOWLEDGE_DIR = Path("data/knowledge_base")
CACHE_FILE    = Path("data/knowledge_base/.indexed_cache.json")

# Extensions supportées
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    ".png", ".jpg", ".jpeg"
}


def _load_cache() -> Dict[str, str]:
    """
    Charge le cache des fichiers déjà indexés.
    Évite de ré-indexer les fichiers inchangés.

    Returns:
        Dictionnaire {chemin_fichier: hash_contenu}
    """
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_cache(cache: Dict[str, str]):
    """
    Sauvegarde le cache des fichiers indexés.

    Args:
        cache: Dictionnaire à sauvegarder
    """
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.error("Erreur sauvegarde cache : %s", e)


def _get_file_hash(filepath: Path) -> str:
    """
    Calcule le hash MD5 d'un fichier pour détecter les changements.

    Args:
        filepath: Chemin du fichier

    Returns:
        Hash MD5 en hexadécimal
    """
    md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
    except Exception:
        return ""
    return md5.hexdigest()


def _collect_files() -> List[Path]:
    """
    Collecte tous les fichiers de cours dans le répertoire knowledge_base.

    Returns:
        Liste des fichiers à indexer
    """
    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        _create_sample_structure()
        return []

    files = []
    for path in KNOWLEDGE_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            # Ignorer les fichiers cachés
            if not path.name.startswith("."):
                files.append(path)

    return sorted(files)


def _create_sample_structure():
    """
    Crée la structure de dossiers vide avec un README.
    """
    dirs = [
        KNOWLEDGE_DIR / "seconde",
        KNOWLEDGE_DIR / "premiere",
        KNOWLEDGE_DIR / "terminale",
        KNOWLEDGE_DIR / "transversal",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    readme = KNOWLEDGE_DIR / "README.md"
    readme.write_text("""# 📚 Base de connaissances

Placez vos cours dans les dossiers correspondants :

```
knowledge_base/
├── seconde/          ← Cours de Seconde
├── premiere/         ← Cours de Première
├── terminale/        ← Cours de Terminale
└── transversal/      ← Formules, théorèmes généraux
```

## Formats acceptés
- 📕 PDF
- 📘 DOCX (Word)
- 📄 TXT, MD
- 🖼️ PNG, JPG (OCR automatique)

## Fonctionnement
Les fichiers sont **automatiquement indexés** au démarrage.
Seuls les fichiers nouveaux ou modifiés sont ré-indexés.
""", encoding="utf-8")


def _extract_text_from_file(filepath: Path) -> str:
    """
    Extrait le texte d'un fichier selon son extension.

    Args:
        filepath: Chemin du fichier

    Returns:
        Texte extrait
    """
    ext = filepath.suffix.lower()

    try:
        if ext == ".txt" or ext == ".md":
            return filepath.read_text(encoding="utf-8", errors="ignore")

        elif ext == ".pdf":
            import pdfplumber
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""

                    # Extraire les tableaux et les convertir en texte structuré
                    tables = page.extract_tables()
                    table_text = ""
                    for table in tables:
                        for row in table:
                            row_cells = [str(cell or "").strip() for cell in row]
                            table_text += " | ".join(row_cells) + "\n"
                        table_text += "\n"

                    if page_text.strip():
                        text += page_text + "\n"
                    if table_text.strip():
                        text += table_text
            return text

        elif ext == ".docx":
            from docx import Document
            doc = Document(str(filepath))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

        elif ext in {".png", ".jpg", ".jpeg"}:
            try:
                import pytesseract
                from PIL import Image
                image = Image.open(filepath).convert("L")
                return pytesseract.image_to_string(image, lang="fra")
            except ImportError:
                logger.warning("pytesseract non installé, image ignorée : %s", filepath.name)
                return ""

    except Exception as e:
        logger.error("Erreur extraction %s : %s", filepath.name, e, exc_info=True)
        return ""

    return ""


def _detect_level_from_path(filepath: Path) -> str:
    """
    Détecte automatiquement le niveau depuis le chemin complet.
    
    Args:
        filepath: Chemin du fichier
    
    Returns:
        Niveau détecté (Seconde, Première, Terminale) ou nom du dossier parent
    """
    try:
        relative_path = str(filepath.relative_to(KNOWLEDGE_DIR)).lower()
        
        # Détection depuis le chemin complet
        if 'seconde' in relative_path:
            return 'Seconde'
        elif 'premiere' in relative_path or 'première' in relative_path:
            return 'Première'
        elif 'terminale' in relative_path:
            return 'Terminale'
        else:
            # Fallback : nom du dossier parent
            return filepath.parent.name
    except:
        return filepath.parent.name


def init_knowledge_base(force_reindex: bool = False):
    """
    Point d'entrée principal.
    Indexe silencieusement les cours au démarrage.

    Args:
        force_reindex: Si True, ré-indexe tous les fichiers même s'ils n'ont pas changé
    """
    from core.vectorstore_manager import vectorstore_manager

    logger.info("Vérification de la base de connaissances...")

    # Collecter les fichiers disponibles
    files = _collect_files()

    if not files:
        logger.warning("Aucun fichier trouvé dans data/knowledge_base/")
        logger.info("Placez vos cours dans : %s", KNOWLEDGE_DIR.absolute())
        return

    # Charger le cache
    cache = _load_cache() if not force_reindex else {}

    # Filtrer les fichiers à (ré-)indexer
    files_to_index = []
    for filepath in files:
        file_hash = _get_file_hash(filepath)
        cached_hash = cache.get(str(filepath))

        if file_hash != cached_hash:
            files_to_index.append((filepath, file_hash))

    if not files_to_index:
        logger.info("Base à jour — %d fichier(s) déjà indexé(s)", len(files))
        return

    logger.info("%d fichier(s) à indexer...", len(files_to_index))

    # Indexer les fichiers
    indexed_count = 0

    for filepath, file_hash in files_to_index:
        try:
            logger.info("Indexation : %s...", filepath.name)

            # Extraire le texte
            text = _extract_text_from_file(filepath)

            if not text.strip():
                logger.warning("Texte vide : %s", filepath.name)
                continue

            # Métadonnées avec détection automatique du niveau
            level = _detect_level_from_path(filepath)
            
            metadata = {
                "filename": filepath.name,
                "filepath": str(filepath),
                "level":    level,              # ← Détection automatique améliorée
                "source":   "knowledge_base",
                "source_type": "knowledge_base",  # Pour compatibilité vectorstore_manager
                "size":     filepath.stat().st_size
            }

            # Ajouter à Qdrant avec source_type
            vectorstore_manager.add_documents(
                [text], 
                [metadata],
                source_type="knowledge_base"  # Marquer comme knowledge_base
            )

            # Mettre à jour le cache
            cache[str(filepath)] = file_hash
            indexed_count += 1

            logger.info("Indexé : %s (%s)", filepath.name, level)

        except Exception as e:
            logger.error("Erreur indexation %s : %s", filepath.name, e, exc_info=True)

    # Sauvegarder le cache mis à jour
    _save_cache(cache)

    logger.info("Indexation terminée — %d/%d fichier(s)", indexed_count, len(files_to_index))


def get_knowledge_base_stats() -> Dict[str, Any]:
    """
    Retourne les statistiques de la base de connaissances.
    Utile pour la sidebar admin.

    Returns:
        Dictionnaire de statistiques
    """
    files  = _collect_files()
    cache  = _load_cache()

    indexed = sum(
        1 for f in files
        if cache.get(str(f)) == _get_file_hash(f)
    )

    by_level: Dict[str, int] = {}
    for f in files:
        level = _detect_level_from_path(f)
        by_level[level] = by_level.get(level, 0) + 1

    return {
        "total_files":   len(files),
        "indexed_files": indexed,
        "pending_files": len(files) - indexed,
        "by_level":      by_level,
        "knowledge_dir": str(KNOWLEDGE_DIR.absolute()),
    }
