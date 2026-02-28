"""
Pluggable LLM interface for the cognitive architecture.

Defines a BaseLLM protocol and a concrete LlamaCppLLM implementation
backed by llama-cpp-python. Additional backends (e.g. vLLM, HF
Transformers) can be added by implementing BaseLLM.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseLLM(ABC):
    """Protocol that every LLM backend must implement."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate a completion for the given prompt.

        Returns the generated text (no special tokens, no prompt echo).
        """
        ...

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """Generate completions for multiple prompts (used by ToT)."""
        ...


# ---------------------------------------------------------------------------
# llama-cpp-python implementation
# ---------------------------------------------------------------------------

class LlamaCppLLM(BaseLLM):
    """LLM backed by a local GGUF model via llama-cpp-python."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config.default()
        self._model = None  # lazy init

    # -- lazy loading so import is cheap --
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required. "
                "Install with: pip install llama-cpp-python"
            ) from exc

        logger.info("Loading model from %s …", self.config.model_path)
        self._model = Llama(
            model_path=self.config.model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=self.config.n_ctx,
            verbose=False,
        )
        logger.info("Model loaded successfully.")

    # -- public API --

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        self._ensure_loaded()
        result = self._model(
            prompt,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=top_p or self.config.top_p,
            stop=stop or self.config.stop_tokens,
            echo=False,
        )
        text: str = result["choices"][0]["text"]
        return text.strip()

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """Sequential batch — llama.cpp doesn't do true batching easily."""
        return [self.generate(p, **kwargs) for p in prompts]

    def __repr__(self) -> str:
        return f"LlamaCppLLM(model={self.config.model_path!r})"
