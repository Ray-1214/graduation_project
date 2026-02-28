"""
RAG Indexer — document chunking, embedding, and indexing.

Reads text files, splits them into overlapping chunks, and stores
them in a VectorStore for later retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from memory.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Indexer:
    """Indexes text documents into a vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _chunk_text(self, text: str, source: str = "") -> List[Dict]:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        for i in range(0, len(text), step):
            chunk_text = text[i : i + self.chunk_size]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": {
                        "source": source,
                        "char_offset": i,
                    },
                })
            if i + self.chunk_size >= len(text):
                break
        return chunks

    def index_file(self, path: str) -> int:
        """Read and index a single text file. Returns chunk count."""
        p = Path(path)
        if not p.exists():
            logger.warning("File not found: %s", path)
            return 0

        text = p.read_text(encoding="utf-8", errors="replace")
        chunks = self._chunk_text(text, source=str(p))
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        metas = [c["metadata"] for c in chunks]
        self.vector_store.add_texts(texts, metas)
        logger.info("Indexed %d chunks from %s", len(chunks), p.name)
        return len(chunks)

    def index_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None,
    ) -> int:
        """Index all matching files in a directory. Returns total chunk count."""
        exts = extensions or [".txt", ".md", ".py", ".json"]
        p = Path(dir_path)
        if not p.is_dir():
            logger.warning("Directory not found: %s", dir_path)
            return 0

        total = 0
        for filepath in sorted(p.rglob("*")):
            if filepath.is_file() and filepath.suffix in exts:
                total += self.index_file(str(filepath))

        logger.info("Indexed %d total chunks from %s", total, dir_path)
        return total
