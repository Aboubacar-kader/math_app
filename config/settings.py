"""
Configuration centralisée de l'application.
Charge les variables d'environnement et définit les paramètres globaux.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """Paramètres de configuration de l'application"""

    # Chemins
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    VECTORSTORE_DIR: Path = DATA_DIR / "vectorstore"
    CONVERSATIONS_DIR: Path = DATA_DIR / "conversations"

    # Configuration 1min.ai
    MIN_AI_API_KEY: str = ""
    MIN_AI_MODEL: str = "gpt-4o"
    MIN_AI_BASE_URL: str = "https://api.1min.ai"
    MIN_AI_TEMPERATURE: float = 0.3

    # Configuration embeddings (local, sans GPU)
    EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Configuration Qdrant
    QDRANT_PATH: str = "./data/vectorstore"
    QDRANT_COLLECTION_NAME: str = "math_documents"

    # Traitement de texte
    MAX_FILE_SIZE_MB: int = 50
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Langues
    SPEECH_LANG: str = "fr-FR"
    TTS_LANG: str = "fr"

    # Interface
    PAGE_TITLE: str = "🧮 IntelliMath"
    PAGE_ICON: str = "🧮"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self):
        """Crée les répertoires nécessaires de manière sécurisée"""
        directories = [
            self.UPLOAD_DIR,
            self.VECTORSTORE_DIR,
            self.CONVERSATIONS_DIR
        ]

        for directory in directories:
            try:
                if directory.exists() and directory.is_file():
                    directory.unlink()
                    print(f"⚠️ Fichier supprimé : {directory}")
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"❌ Erreur lors de la création de {directory}: {e}")
                pass

# Instance globale
settings = Settings()
