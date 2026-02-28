"""
RAG Retriever — similarity search and context formatting.

Wraps the VectorStore to provide a clean retrieval interface that
returns both raw documents and a formatted context string ready
for prompt injection.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from memory.vector_store import Document, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Queries the vector store and formats results for prompt injection."""

    def __init__(self, vector_store: VectorStore, top_k: int = 3) -> None:
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """Return the most relevant documents for a query."""
        k = top_k or self.top_k
        docs = self.vector_store.query(query, top_k=k)
        logger.info(
            "Retrieved %d documents for query: %s",
            len(docs),
            query[:60],
        )
        return docs

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> str:
        """Return a formatted context string for prompt injection."""
        docs = self.retrieve(query, top_k)
        if not docs:
            return "(No relevant context found.)"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(
                f"[{i}] (source: {source}, score: {doc.score:.3f})\n{doc.text}"
            )
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return f"Retriever(store={self.vector_store!r}, top_k={self.top_k})"
