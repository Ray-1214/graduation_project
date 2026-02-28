"""
Vector store — FAISS-backed similarity search with sentence-transformers.

Provides add/query interface for RAG retrieval and semantic memory.
Embedding model is loaded lazily to keep imports fast.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A retrieved document chunk."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class VectorStore:
    """FAISS index + sentence-transformers embedding model."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2") -> None:
        self.embedding_model_name = embedding_model_name
        self._embedder = None
        self._index = None
        self._texts: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._dim: Optional[int] = None

    # -- lazy init --

    def _ensure_embedder(self):
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from exc
        logger.info("Loading embedding model '%s' …", self.embedding_model_name)
        self._embedder = SentenceTransformer(self.embedding_model_name)
        self._dim = self._embedder.get_sentence_embedding_dimension()

    def _ensure_index(self):
        if self._index is not None:
            return
        self._ensure_embedder()
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            ) from exc
        self._index = faiss.IndexFlatIP(self._dim)  # inner-product (cosine after norm)

    # -- public API --

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Embed and index a batch of texts. Returns count added."""
        self._ensure_index()
        if not texts:
            return 0

        metas = metadatas or [{} for _ in texts]
        embeddings = self._embedder.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        self._texts.extend(texts)
        self._metadatas.extend(metas)
        logger.info("Added %d texts to vector store (total: %d).", len(texts), len(self._texts))
        return len(texts)

    def query(self, text: str, top_k: int = 3) -> List[Document]:
        """Return the top_k most similar documents."""
        self._ensure_index()
        if self._index.ntotal == 0:
            return []

        q_emb = self._embedder.encode([text], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype=np.float32)
        scores, indices = self._index.search(q_emb, min(top_k, self._index.ntotal))

        results: List[Document] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(Document(
                text=self._texts[idx],
                metadata=self._metadatas[idx],
                score=float(score),
            ))
        return results

    @property
    def size(self) -> int:
        return len(self._texts)

    def __repr__(self) -> str:
        return f"VectorStore(model={self.embedding_model_name!r}, docs={self.size})"
