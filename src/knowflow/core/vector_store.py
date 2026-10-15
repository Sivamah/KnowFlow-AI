"""
KnowFlow AI - Persistent Vector Store Manager
Manages ChromaDB persistence, batch embedding generation, document-level deletion,
and index cataloging. Guarantees that indexing survives application reboots.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

from src.knowflow.core.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME
)
from src.knowflow.core.chunker import DocumentChunk
from src.knowflow.core.document_loader import LoadedDocument

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Production ChromaDB manager supporting persistence, document deduplication,
    proper document removal, and catalog tracking.
    """

    def __init__(
        self,
        persist_dir: Optional[Union[str, Path]] = None,
        collection_name: str = CHROMA_COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME
    ):
        self.persist_dir = Path(persist_dir or CHROMA_PERSIST_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.model_name = model_name

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "KnowFlow AI persistent knowledge base"}
        )

        # Lazy-load embedding model
        self._embed_model: Optional[SentenceTransformer] = None
        self._manifest_path = self.persist_dir / "documents_manifest.json"
        self._manifest: Dict[str, dict] = self._load_manifest()

    @property
    def embed_model(self) -> SentenceTransformer:
        """Lazy-loaded sentence transformer."""
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self.model_name)
        return self._embed_model

    def _load_manifest(self) -> Dict[str, dict]:
        """Load document metadata catalog from disk."""
        if self._manifest_path.is_file():
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load manifest: {e}. Starting fresh.")
        return {}

    def _save_manifest(self) -> None:
        """Persist document metadata catalog to disk."""
        try:
            with open(self._manifest_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save document manifest: {e}")

    def is_document_indexed(self, doc_id: str, sha256: Optional[str] = None) -> bool:
        """Check if a document is already indexed by ID or content hash."""
        if doc_id in self._manifest:
            return True
        if sha256:
            for entry in self._manifest.values():
                if entry.get("sha256") == sha256:
                    return True
        return False

    def index_document(
        self,
        doc: LoadedDocument,
        chunks: List[DocumentChunk],
        batch_size: int = 64
    ) -> int:
        """
        Embed and persist document chunks into ChromaDB with metadata.
        Returns the number of newly indexed chunks.
        """
        if not chunks:
            return 0

        # Remove previous version if re-indexing document with same ID
        if doc.doc_id in self._manifest:
            self.delete_document(doc.doc_id)

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Generate normalized embeddings in batches
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeds = self.embed_model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            all_embeddings.extend(batch_embeds.tolist())

        # Insert into ChromaDB
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=all_embeddings,
            metadatas=metadatas
        )

        # Update and save catalog manifest
        self._manifest[doc.doc_id] = {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "file_size_bytes": doc.file_size_bytes,
            "sha256": doc.sha256,
            "total_pages": doc.total_pages,
            "chunk_count": len(chunks),
            "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._save_manifest()
        return len(chunks)

    def delete_document(self, doc_id: str) -> bool:
        """
        Completely delete all chunks associated with a document from ChromaDB
        and update the manifest catalog.
        """
        deleted = False
        try:
            # Query collection for chunks belonging to this doc_id
            existing = self.collection.get(where={"doc_id": doc_id})
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
                deleted = True
        except Exception as e:
            logger.error(f"Error deleting chunks for doc_id '{doc_id}' from ChromaDB: {e}")

        if doc_id in self._manifest:
            del self._manifest[doc_id]
            self._save_manifest()
            deleted = True

        return deleted

    def get_indexed_documents(self) -> List[dict]:
        """Return a list of all currently indexed documents with metadata."""
        return list(self._manifest.values())

    def get_collection_stats(self) -> dict:
        """Return summary metrics for the vector database."""
        total_chunks = self.collection.count()
        return {
            "total_chunks": total_chunks,
            "total_documents": len(self._manifest),
            "persist_dir": str(self.persist_dir),
            "collection_name": self.collection_name,
            "embedding_model": self.model_name
        }

    def clear_all(self) -> None:
        """Completely reset the collection and clear the catalog."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "KnowFlow AI persistent knowledge base"}
        )
        self._manifest.clear()
        self._save_manifest()
