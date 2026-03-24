"""
Système de journalisation centralisé — IntelliMath.

Toutes les erreurs, avertissements et événements importants sont écrits dans
logs/intellimath.log avec rotation automatique (5 MB max, 3 sauvegardes).

Usage dans chaque module :
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Message informatif")
    logger.warning("Avertissement")
    logger.error("Erreur", exc_info=True)  # exc_info=True pour inclure la stack trace
"""

import logging
import logging.handlers
from pathlib import Path

# ── Chemins ────────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "intellimath.log"

# ── Paramètres de rotation ──────────────────────────────────────────────────
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB par fichier
_BACKUP_COUNT = 3              # 3 sauvegardes (intellimath.log.1, .2, .3)

# ── Format ──────────────────────────────────────────────────────────────────
_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _setup_root_logger() -> None:
    """Configure le logger racine 'intellimath' une seule fois."""
    root = logging.getLogger("intellimath")
    if root.handlers:
        return  # Déjà configuré

    root.setLevel(logging.INFO)

    # Créer le dossier logs/ si nécessaire
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        # En cas d'erreur (ex. Streamlit Cloud sans accès disque), on continue sans fichier
        return

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── Handler fichier avec rotation ──────────────────────────────────────
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Streamlit Cloud : système de fichiers en lecture seule possible
        pass

    # ── Handler console ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)


_setup_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger nommé, enfant du logger racine 'intellimath'.

    Args:
        name: Nom du module appelant, typiquement __name__

    Returns:
        Logger configuré avec rotation de fichiers
    """
    # Simplifier le nom : "intellimath.core.llm_manager" au lieu de
    # "intellimath.math_chat.core.llm_manager"
    short_name = name.split("math_chat.")[-1] if "math_chat." in name else name
    return logging.getLogger(f"intellimath.{short_name}")
