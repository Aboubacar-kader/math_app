"""
Service de traitement des documents uploadés.
Extrait le texte de différents formats de fichiers.
"""

import pdfplumber
from docx import Document
import io
from typing import List, Dict, Any, Optional
import streamlit as st
from pathlib import Path
from config.settings import settings
from core.llm_manager import call_1minai

class DocumentProcessor:
    """Processeur de documents multi-formats"""
    
    SUPPORTED_TYPES = {
        'application/pdf': 'pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'text/plain': 'txt',
        'image/png': 'image',
        'image/jpeg': 'image',
        'image/jpg': 'image',
        'image/x-png': 'image',
        'image/webp': 'image',
        'image/bmp': 'image',
        'image/gif': 'image',
        'image/tiff': 'image',
        'image/x-ms-bmp': 'image',
    }
    
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
    
    def process_files(self, uploaded_files: List[Any]) -> List[Dict[str, Any]]:
        """
        Traite une liste de fichiers uploadés.
        
        Args:
            uploaded_files: Liste des fichiers Streamlit
            
        Returns:
            Liste de dictionnaires contenant le texte et les métadonnées
        """
        processed_docs = []
        
        for file in uploaded_files:
            try:
                # Vérifier la taille
                if file.size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(f"⚠️ {file.name} dépasse {settings.MAX_FILE_SIZE_MB}MB")
                    continue
                
                # Extraire le texte selon le type
                text = self._extract_text(file)
                
                if text and text.strip():
                    processed_docs.append({
                        'text': text,
                        'metadata': {
                            'filename': file.name,
                            'type': file.type,
                            'size': file.size
                        }
                    })
                    
                    # Sauvegarder le fichier
                    self._save_file(file)
                else:
                    st.warning(f"⚠️ Aucun texte extrait de {file.name}")
                    
            except Exception as e:
                st.error(f"❌ Erreur avec {file.name}: {str(e)}")
        
        return processed_docs
    
    def _extract_text(self, file) -> str:
        """
        Extrait le texte d'un fichier selon son type.
        Fallback par extension si le MIME type n'est pas reconnu.
        """
        file_type = self.SUPPORTED_TYPES.get(file.type)

        # Fallback par extension si MIME type inconnu (ex : image/x-png sur certains navigateurs)
        if not file_type:
            ext = Path(file.name).suffix.lower()
            if ext in ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'):
                file_type = 'image'
            elif ext == '.pdf':
                file_type = 'pdf'
            elif ext in ('.docx', '.doc'):
                file_type = 'docx'
            elif ext == '.txt':
                file_type = 'txt'

        if file_type == 'pdf':
            return self._extract_from_pdf(file)
        elif file_type == 'docx':
            return self._extract_from_docx(file)
        elif file_type == 'txt':
            return self._extract_from_txt(file)
        elif file_type == 'image':
            return self._extract_from_image(file)
        else:
            st.warning(f"⚠️ Type de fichier non supporté : {file.type} ({file.name})")
            return ""
    
    def _extract_from_pdf(self, file) -> str:
        """Extrait le texte et les tableaux d'un PDF via pdfplumber"""
        try:
            file_bytes = io.BytesIO(file.read())
            text = ""

            with pdfplumber.open(file_bytes) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""

                    # Extraire les tableaux et les convertir en texte structuré
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

                    if page_content.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_content}"

            return text
        except Exception as e:
            st.error(f"Erreur PDF: {str(e)}")
            return ""
    
    def _extract_from_docx(self, file) -> str:
        """Extrait le texte d'un document Word (paragraphes + tableaux)"""
        try:
            file_bytes = io.BytesIO(file.read())
            doc = Document(file_bytes)

            parts = []

            # Paragraphes
            for para in doc.paragraphs:
                if para.text.strip():
                    parts.append(para.text)

            # Tableaux
            for table in doc.tables:
                for row in table.rows:
                    row_cells = [cell.text.strip() for cell in row.cells]
                    row_text = " | ".join(c for c in row_cells if c)
                    if row_text:
                        parts.append(row_text)

            return "\n\n".join(parts)
        except Exception as e:
            st.error(f"Erreur DOCX: {str(e)}")
            return ""
    
    def _extract_from_txt(self, file) -> str:
        """Extrait le texte d'un fichier texte"""
        try:
            return file.read().decode('utf-8')
        except UnicodeDecodeError:
            # Essayer avec d'autres encodages
            try:
                file.seek(0)
                return file.read().decode('latin-1')
            except:
                return ""
    
    def _extract_from_image(self, file) -> str:
        """
        Extraction via GPT-4o Vision (1min.ai) — aucun OCR local requis.
        Envoie l'image en base64 directement à l'API.
        """
        import base64
        import requests as _requests
        file_bytes = file.read()

        # Encoder l'image en base64
        b64_image = base64.b64encode(file_bytes).decode('utf-8')

        try:
            url = f"{settings.MIN_AI_BASE_URL}/api/chat-with-ai"
            headers = {
                "Content-Type": "application/json",
                "API-KEY": settings.MIN_AI_API_KEY,
            }
            payload = {
                "type": "CHAT_WITH_AI",
                "model": settings.MIN_AI_MODEL,
                "promptObject": {
                    "prompt": (
                        "Transcris intégralement tout le texte et toutes les formules "
                        "mathématiques de cette image. "
                        "Garde la structure exacte de l'exercice : titres, numéros de questions, "
                        "tableaux, formules (notation standard : f(x) = ...). "
                        "Ne résous rien, transcris uniquement."
                    ),
                    "isMixed": True,
                    "imageList": [b64_image],
                    "webSearch": False,
                },
            }
            response = _requests.post(url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            text = data["aiRecord"]["aiRecordDetail"]["resultObject"][0]
            if text.strip():
                return text.strip()
        except Exception as e:
            st.warning(f"⚠️ Extraction vision GPT-4o échouée : {e}")

        return "[Image - extraction non disponible]"
    
    def _save_file(self, file):
        """Sauvegarde le fichier uploadé"""
        try:
            file_path = self.upload_dir / file.name
            with open(file_path, 'wb') as f:
                f.write(file.getbuffer())
        except Exception as e:
            st.warning(f"Sauvegarde échouée: {str(e)}")
    
    def get_file_stats(self) -> Dict[str, int]:
        """Retourne les statistiques des fichiers uploadés"""
        files = list(self.upload_dir.glob('*'))
        return {
            'total_files': len(files),
            'total_size_mb': sum(f.stat().st_size for f in files) / (1024 * 1024)
        }

# Instance globale
document_processor = DocumentProcessor()