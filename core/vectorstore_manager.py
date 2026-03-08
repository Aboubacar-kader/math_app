"""
Gestionnaire de la base vectorielle Qdrant - VERSION AMÉLIORÉE
Gère l'indexation et la recherche avec distinction knowledge_base / user_uploads
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
import uuid
from config.settings import settings
from core.llm_manager import llm_manager


class VectorStoreManager:
    """
    Gestionnaire de la base vectorielle avec distinction knowledge_base / user_uploads
    
    Deux types de documents :
    - knowledge_base : Documents permanents (cours, théorèmes) indexés au démarrage
    - user_upload : Documents temporaires uploadés par l'utilisateur
    """
    
    def __init__(self):
        self.client = QdrantClient(path=settings.QDRANT_PATH)
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Crée la collection si elle n'existe pas, ou la recrée si la dimension a changé"""
        # Dimension réelle du modèle d'embeddings actuel
        test_embedding = llm_manager.embeddings.embed_query("test")
        expected_size = len(test_embedding)

        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name in collection_names:
            # Vérifier que la dimension stockée correspond
            info = self.client.get_collection(self.collection_name)
            stored_size = info.config.params.vectors.size
            if stored_size != expected_size:
                # Dimension incompatible → supprimer et recréer
                self.client.delete_collection(self.collection_name)
                collection_names = []  # forcer la création ci-dessous

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=expected_size,
                    distance=Distance.COSINE
                )
            )
    
    # ════════════════════════════════════════════════════════
    # AJOUT DE DOCUMENTS
    # ════════════════════════════════════════════════════════
    
    def add_documents(
        self, 
        texts: List[str], 
        metadatas: List[Dict[str, Any]] = None,
        source_type: str = "user_upload"
    ):
        """
        Ajoute des documents à la base vectorielle.
        
        Args:
            texts: Liste de textes à indexer
            metadatas: Métadonnées associées à chaque texte
            source_type: "knowledge_base" (permanent) ou "user_upload" (temporaire)
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # IMPORTANT: Marquer le type de source
        for metadata in metadatas:
            if 'source_type' not in metadata:
                metadata['source_type'] = source_type
        
        # Découper les textes en chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        points = []
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            chunks = text_splitter.split_text(text)
            
            for chunk in chunks:
                # Générer l'embedding
                embedding = llm_manager.embeddings.embed_query(chunk)
                
                # Créer le point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "metadata": metadata,
                        "source_index": i
                    }
                )
                points.append(point)
        
        # Ajouter à la collection
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
    
    # ════════════════════════════════════════════════════════
    # RECHERCHE
    # ════════════════════════════════════════════════════════
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        source_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Recherche les documents les plus similaires à la requête.
        
        Args:
            query: La requête de recherche
            top_k: Nombre de résultats à retourner
            source_type: Filtrer par type ("knowledge_base", "user_upload", ou None pour tous)
            
        Returns:
            Liste des documents trouvés avec leur score
        """
        # Générer l'embedding de la requête
        query_embedding = llm_manager.embeddings.embed_query(query)
        
        # Préparer le filtre si nécessaire
        query_filter = None
        if source_type:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_type",
                        match=MatchValue(value=source_type)
                    )
                ]
            )
        
        # Rechercher dans Qdrant
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter
        )
        
        # Formatter les résultats
        documents = []
        for result in results:
            documents.append({
                "text": result.payload["text"],
                "score": result.score,
                "metadata": result.payload.get("metadata", {})
            })
        
        return documents
    
    # ════════════════════════════════════════════════════════
    # RECHERCHE SPÉCIFIQUE
    # ════════════════════════════════════════════════════════
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recherche UNIQUEMENT dans la knowledge base (documents permanents)
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats
        
        Returns:
            Documents de la knowledge base uniquement
        """
        return self.search(query, top_k, source_type="knowledge_base")
    
    def search_user_uploads(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Recherche UNIQUEMENT dans les uploads utilisateur (documents temporaires)
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats
        
        Returns:
            Documents uploadés par l'utilisateur uniquement
        """
        return self.search(query, top_k, source_type="user_upload")
    
    # ════════════════════════════════════════════════════════
    # GESTION
    # ════════════════════════════════════════════════════════
    
    def count_documents(self, source_type: Optional[str] = None) -> int:
        """
        Compte le nombre de documents dans la base
        
        Args:
            source_type: Filtrer par type ou None pour tous
        
        Returns:
            Nombre de documents
        """
        if source_type:
            # Compter avec filtre
            # Note: Qdrant ne permet pas de count avec filtre facilement
            # On fait une recherche avec limit élevé
            results = self.search("", top_k=10000, source_type=source_type)
            return len(results)
        else:
            # Compter tous
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
    
    def clear_user_uploads(self):
        """
        Supprime UNIQUEMENT les documents uploadés par l'utilisateur
        Garde la knowledge base intacte
        """
        # Qdrant ne permet pas de delete par filtre facilement
        # On doit d'abord récupérer les IDs puis les supprimer
        # Pour simplifier, on peut implémenter ça plus tard si nécessaire
        pass
    
    def clear_collection(self):
        """Vide complètement la collection (knowledge_base + user_uploads)"""
        self.client.delete_collection(self.collection_name)
        self._ensure_collection_exists()
    
    # ════════════════════════════════════════════════════════
    # STATISTIQUES
    # ════════════════════════════════════════════════════════
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur la base vectorielle
        
        Returns:
            Dict avec total, knowledge_base, user_uploads
        """
        total = self.count_documents()
        
        # Pour compter par type, on devrait faire une vraie requête
        # Pour l'instant, on retourne juste le total
        return {
            "total": total,
            "knowledge_base": None,  # Nécessite requête filtrée
            "user_uploads": None     # Nécessite requête filtrée
        }


# ════════════════════════════════════════════════════════
# INSTANCE GLOBALE
# ════════════════════════════════════════════════════════

vectorstore_manager = VectorStoreManager()
