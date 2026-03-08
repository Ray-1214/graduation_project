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


# ---------------------------------------------------------------------------
# CPU–Coprocessor Model Manager (hot-swap scheduler)
# ---------------------------------------------------------------------------

import gc
import time


class ModelManager:
    """Manages two LLM models with hot-swap scheduling.

    Only one model resides in VRAM at a time.  The manager lazily loads the
    requested role ("thinking" or "coding") and unloads the previous one,
    freeing GPU memory via ``gc.collect()`` and ``torch.cuda.empty_cache()``.

    Attributes:
        config: Global configuration (contains paths & generation params).
    """

    # Valid roles
    ROLE_THINKING: str = "thinking"
    ROLE_CODING: str = "coding"
    _VALID_ROLES = {ROLE_THINKING, ROLE_CODING}

    def __init__(self, config: Config) -> None:
        self.config = config
        self._current_role: str = "none"
        self._model: Any = None
        self._Llama: Optional[type] = None  # cached class reference

        # Telemetry
        self._swap_count: int = 0
        self._swap_total_ms: float = 0.0

    # ------------------------------------------------------------------
    # Lazy import
    # ------------------------------------------------------------------

    def _get_llama_class(self) -> type:
        """Lazily import and cache ``llama_cpp.Llama``."""
        if self._Llama is not None:
            return self._Llama
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required. "
                "Install with: pip install llama-cpp-python"
            ) from exc
        self._Llama = Llama
        return Llama

    # ------------------------------------------------------------------
    # Unload / Load
    # ------------------------------------------------------------------

    def _unload(self) -> None:
        """Release the current model from VRAM."""
        if self._model is None:
            self._current_role = "none"
            return

        prev = self._current_role
        logger.info("Unloading %s model …", prev)

        del self._model
        self._model = None
        gc.collect()

        # Clear CUDA cache if PyTorch is available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared.")
        except ImportError:
            pass

        self._current_role = "none"
        logger.info("Model %s unloaded.", prev)

    def _load(self, role: str) -> None:
        """Load the model for *role* into VRAM.

        Args:
            role: ``"thinking"`` or ``"coding"``.
        """
        if role not in self._VALID_ROLES:
            raise ValueError(
                f"Unknown role {role!r}; expected one of {self._VALID_ROLES}"
            )

        Llama = self._get_llama_class()

        # Select parameters by role
        if role == self.ROLE_THINKING:
            model_path = self.config.model_path
            n_ctx = self.config.n_ctx
        else:  # coding
            model_path = self.config.code_model_path
            n_ctx = self.config.code_n_ctx

        logger.info("Loading %s model from %s (n_ctx=%d) …", role, model_path, n_ctx)
        t0 = time.perf_counter()

        self._model = Llama(
            model_path=model_path,
            n_gpu_layers=self.config.n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._swap_count += 1
        self._swap_total_ms += elapsed_ms
        self._current_role = role

        logger.info(
            "Model %s loaded in %.0f ms (swap #%d).",
            role, elapsed_ms, self._swap_count,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_role(self, role: str) -> None:
        """Ensure the model for *role* is loaded (no-op if already active).

        If ``config.model_swap_enabled`` is ``False``, the thinking model
        is always used regardless of the requested role.

        Args:
            role: ``"thinking"`` or ``"coding"``.
        """
        # Fallback when swapping is disabled
        if not self.config.model_swap_enabled:
            role = self.ROLE_THINKING

        if self._current_role == role:
            return  # already loaded — no-op

        self._unload()
        self._load(role)

    def generate(
        self,
        prompt: str,
        role: str = "thinking",
        **kwargs: Any,
    ) -> str:
        """Generate text using the model associated with *role*.

        Default generation parameters (temperature, max_tokens, …) are
        chosen based on the role but can be overridden via **kwargs.

        Args:
            prompt: The input prompt.
            role: ``"thinking"`` or ``"coding"``.
            **kwargs: Overrides for generation parameters.

        Returns:
            The generated text, stripped of leading/trailing whitespace.
        """
        self.ensure_role(role)

        # Build defaults based on role
        if role == self.ROLE_CODING and self.config.model_swap_enabled:
            defaults: Dict[str, Any] = {
                "max_tokens": self.config.code_max_tokens,
                "temperature": self.config.code_temperature,
                "top_p": self.config.code_top_p,
                "stop": self.config.code_stop_tokens,
            }
        else:
            defaults = {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "stop": self.config.stop_tokens,
            }

        # kwargs take precedence
        defaults.update(kwargs)

        result = self._model(prompt, echo=False, **defaults)
        text: str = result["choices"][0]["text"]
        return text.strip()

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Return swap telemetry as a dict."""
        return {
            "current_role": self._current_role,
            "total_swaps": self._swap_count,
            "total_swap_ms": round(self._swap_total_ms, 1),
            "avg_swap_ms": round(
                self._swap_total_ms / max(1, self._swap_count), 1
            ),
        }

    def __repr__(self) -> str:
        return (
            f"ModelManager(role={self._current_role!r}, "
            f"swaps={self._swap_count})"
        )

