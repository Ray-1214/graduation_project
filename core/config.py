"""
Global configuration for the cognitive architecture.

Centralizes model paths, generation parameters, and system defaults.
All components read from a shared Config instance.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


# Project root is two levels up from this file (core/config.py → AI/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """Immutable configuration for the cognitive architecture."""

    # --- LLM ---
    model_path: str = str(PROJECT_ROOT / "models" / "llm" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
    n_gpu_layers: int = 99          # offload all layers to GPU
    n_ctx: int = 4096               # context window (Mistral supports up to 8192)
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stop_tokens: list = field(default_factory=lambda: ["\n\n\n", "---"])

    # --- Embeddings (for RAG / vector store) ---
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # --- Memory ---
    short_term_capacity: int = 20   # max messages in working memory
    long_term_store_path: str = str(PROJECT_ROOT / "memory" / "reflections.json")

    # --- RAG ---
    chunk_size: int = 512           # characters per chunk
    chunk_overlap: int = 64         # overlap between chunks
    retrieval_top_k: int = 3

    # --- Reasoning ---
    tot_branch_factor: int = 3      # branches per node in ToT
    tot_max_depth: int = 3
    tot_beam_width: int = 2
    react_max_steps: int = 8

    # --- Experiments ---
    results_dir: str = str(PROJECT_ROOT / "experiments" / "results")

    # --- Derived ---
    @property
    def knowledge_base_dir(self) -> str:
        return str(PROJECT_ROOT / "rag" / "knowledge_base")

    @classmethod
    def default(cls) -> "Config":
        """Return a default config, respecting environment overrides."""
        overrides = {}
        if env_model := os.environ.get("COGARCH_MODEL_PATH"):
            overrides["model_path"] = env_model
        if env_ctx := os.environ.get("COGARCH_N_CTX"):
            overrides["n_ctx"] = int(env_ctx)
        if env_gpu := os.environ.get("COGARCH_N_GPU_LAYERS"):
            overrides["n_gpu_layers"] = int(env_gpu)
        if env_temp := os.environ.get("COGARCH_TEMPERATURE"):
            overrides["temperature"] = float(env_temp)
        return cls(**overrides)
