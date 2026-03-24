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
import io
import base64
import time
from utils.logger import get_logger

logger = get_logger(__name__)

# Répertoires des cours
KNOWLEDGE_DIR = Path("data/knowledge_base")
CACHE_FILE    = Path("data/knowledge_base/.indexed_cache.json")

# Version du cache — incrémenter quand l'extraction change (ex: ajout OCR)
# Cela force une ré-indexation complète propre au prochain démarrage.
CACHE_VERSION = "3"

# Extensions supportées
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md",
    ".png", ".jpg", ".jpeg"
}


def _load_cache() -> tuple[Dict[str, str], bool]:
    """
    Charge le cache des fichiers déjà indexés.
    Retourne aussi un booléen indiquant si une migration est nécessaire.

    Returns:
        (cache, needs_migration) — needs_migration=True si le cache est d'une
        ancienne version et que la collection doit être réinitialisée.
    """
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("__version__") != CACHE_VERSION:
                logger.info(
                    "Cache obsolète (v%s → v%s) — ré-indexation complète avec OCR.",
                    data.get("__version__", "1"), CACHE_VERSION,
                )
                return {}, True
            return {k: v for k, v in data.items() if k != "__version__"}, False
    except Exception:
        pass
    return {}, False


def _save_cache(cache: Dict[str, str]):
    """Sauvegarde le cache avec numéro de version."""
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {"__version__": CACHE_VERSION, **cache}
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
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


def _tesseract_available() -> bool:
    """Vérifie si Tesseract OCR est installé sur la machine."""
    import shutil
    import subprocess
    # Chercher dans le PATH d'abord
    if shutil.which("tesseract"):
        return True
    # Emplacement par défaut Windows
    win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(win_path).exists():
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = win_path
        except ImportError:
            pass
        return True
    return False


def _ocr_page_tesseract(pil_image, page_num: int, filename: str) -> str:
    """OCR local via Tesseract — gratuit, sans limite de débit."""
    try:
        import pytesseract
        win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if Path(win_path).exists():
            pytesseract.pytesseract.tesseract_cmd = win_path
        text = pytesseract.image_to_string(pil_image, lang="fra+eng")
        return text.strip()
    except Exception as e:
        logger.warning("Tesseract page %d de '%s' echoue : %s", page_num, filename, e)
        return ""


def _ocr_page_vision(page, page_num: int, filename: str) -> str:
    """OCR d'une page PDF scannée.
    Priorité : Tesseract (local, sans limite) → API Vision (avec retry 429).
    """
    pil_image = page.to_image(resolution=150).original

    # 1. Tesseract local — prioritaire si disponible
    if _tesseract_available():
        return _ocr_page_tesseract(pil_image, page_num, filename)

    # 2. Fallback : API Vision avec retry exponentiel
    from config.settings import settings
    import requests as _req

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    url = f"{settings.MIN_AI_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.MIN_AI_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Transcris intégralement tout le texte et les formules "
                    "mathematiques de cette page de cours. "
                    "Conserve la structure : titres, numeros, tableaux, formules. "
                    "Ne resous rien, transcris uniquement."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "max_tokens": 4096,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.MIN_AI_API_KEY.strip()}",
    }

    delays = [10, 30]  # 2 tentatives suffisent — si ça échoue, quota épuisé
    for attempt, delay in enumerate(delays + [None], start=1):
        try:
            r = _req.post(url, headers=headers, json=payload, timeout=90)
            if r.status_code == 429:
                if delay is not None:
                    logger.warning(
                        "OCR page %d de '%s' — 429 rate limit, retry %d/%d dans %ds",
                        page_num, filename, attempt, len(delays), delay
                    )
                    time.sleep(delay)
                    continue
                # Quota épuisé après tous les retries → signal d'abandon
                return "__RATE_LIMIT__"
            r.raise_for_status()
            time.sleep(3)  # pause légère entre pages
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if "429" in str(e):
                if delay is not None:
                    logger.warning(
                        "OCR page %d de '%s' — 429 rate limit, retry %d/%d dans %ds",
                        page_num, filename, attempt, len(delays), delay
                    )
                    time.sleep(delay)
                    continue
                return "__RATE_LIMIT__"
            logger.warning("OCR page %d de '%s' echoue : %s", page_num, filename, e)
            return ""

    return "__RATE_LIMIT__"


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
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""

                    # Tableaux → texte structuré
                    tables = page.extract_tables()
                    table_text = ""
                    for table in tables:
                        for row in table:
                            row_cells = [str(cell or "").strip() for cell in row]
                            table_text += " | ".join(row_cells) + "\n"
                        table_text += "\n"

                    page_content = page_text
                    if table_text.strip():
                        page_content += "\n" + table_text

                    # Fallback OCR si page vide (PDF scanné)
                    if not page_content.strip():
                        ocr_result = _ocr_page_vision(page, page_num + 1, filepath.name)
                        if ocr_result == "__RATE_LIMIT__":
                            logger.warning(
                                "Quota OCR épuisé sur '%s' (page %d) — fichier ignoré.",
                                filepath.name, page_num + 1
                            )
                            break  # passer au fichier suivant
                        page_content = ocr_result

                    if page_content.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_content}"
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

    # Mode cloud : si Qdrant Cloud est configuré et déjà peuplé → pas d'indexation locale
    from config.settings import settings as _settings
    if _settings.QDRANT_URL:
        try:
            count = vectorstore_manager.client.count(
                collection_name=vectorstore_manager.collection_name
            ).count
            if count > 0:
                logger.info(
                    "Mode cloud — %d vecteurs déjà indexés dans Qdrant Cloud. Indexation locale ignorée.",
                    count
                )
                return
            logger.info("Mode cloud — collection vide, indexation des fichiers locaux...")
        except Exception as e:
            logger.warning("Mode cloud — vérification collection échouée : %s", e)

    # Collecter les fichiers disponibles
    files = _collect_files()

    if not files:
        if _settings.QDRANT_URL:
            logger.info("Mode cloud — aucun fichier local, vectorstore cloud utilisé tel quel.")
        else:
            logger.warning("Aucun fichier trouvé dans data/knowledge_base/")
            logger.info("Placez vos cours dans : %s", KNOWLEDGE_DIR.absolute())
        return

    # Charger le cache (détecte si migration nécessaire)
    if force_reindex:
        cache, needs_migration = {}, True
    else:
        cache, needs_migration = _load_cache()

    # Migration : vider la collection pour repartir proprement avec OCR
    if needs_migration:
        logger.info("Réinitialisation de la collection vectorstore pour migration OCR...")
        try:
            vectorstore_manager.clear_collection()
            logger.info("Collection réinitialisée.")
        except Exception as e:
            logger.error("Erreur réinitialisation collection : %s", e)

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
    files        = _collect_files()
    cache, _     = _load_cache()

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
