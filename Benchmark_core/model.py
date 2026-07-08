"""
benchmark_core/model.py

Core model wrapper for loading and running GGUF-quantized LLMs via
llama-cpp-python, used across the Q-Elasticity benchmarking framework.

This module is intentionally scoped to model loading, text generation,
and resource cleanup only. Benchmark-specific logic (prompt construction,
answer extraction, scoring) lives in the `benchmarks/` package and calls
into this module.

Output key convention:
    `generate()` always returns "generated_text" (not "text") as the key
    for the model's output string. This is the single canonical key used
    throughout the framework (benchmarks, csv_logger, evaluation) — do not
    introduce "text" elsewhere as an alias.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from llama_cpp import Llama


class BenchmarkModelError(Exception):
    """Raised when BenchmarkModel fails to load a model or generate text."""


class BenchmarkModel:
    """
    Thin, reusable wrapper around llama_cpp.Llama for chat-style generation.

    Designed to work uniformly across Llama-3 and Qwen2.5 instruct-tuned
    GGUF models, since both expose OpenAI-style chat templates that
    llama-cpp-python's `create_chat_completion` can consume directly.

    Attributes:
        model_path: Filesystem path to the GGUF model file.
        n_ctx: Context window size (tokens).
        n_gpu_layers: Number of layers to offload to GPU (-1 for all).
        seed: RNG seed for deterministic generation.
        verbose: Whether llama.cpp should print its own internal logs.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a BenchmarkModel instance.

        Note: this does NOT load the model into memory. Call `load_model()`
        explicitly before generating, so callers control when the (often
        slow, GPU-memory-consuming) load happens.

        Args:
            model_path: Path to a .gguf model file.
            n_ctx: Context window size in tokens.
            n_gpu_layers: Number of transformer layers to offload to GPU.
                Use -1 to offload all layers (full GPU inference).
            seed: Seed for reproducible sampling.
            verbose: If True, llama.cpp prints internal load/inference logs.
        """
        self.model_path = model_path
        self.model_name = Path(model_path).stem
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.verbose = verbose

        self._llm: Optional[Llama] = None

    @property
    def is_loaded(self) -> bool:
        """Return True if the underlying Llama model is currently loaded."""
        return self._llm is not None

    def load_model(self) -> None:
        """
        Load the GGUF model into memory (and GPU, if n_gpu_layers != 0).

        Raises:
            BenchmarkModelError: If the model file cannot be loaded.
        """
        if self.is_loaded:
            return

        if not Path(self.model_path).is_file():
            raise BenchmarkModelError(
                f"Model file not found: {self.model_path}"
            )

        try:
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                seed=self.seed,
                verbose=self.verbose,
            )
        except Exception as exc:
            raise BenchmarkModelError(
                f"Failed to load model at '{self.model_path}': {exc}"
            ) from exc

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: Optional[Union[str, List[str]]] = None,
        repeat_penalty: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generate a chat completion for a single user prompt.

        Args:
            prompt: The user message / task prompt.
            system_prompt: Optional system message to steer model behavior.
                If None, no system message is included.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic/greedy).
            top_p: Nucleus sampling probability mass.
            stop: Optional stop sequence(s). A single string or a list of
                strings; generation halts as soon as any is produced.
            repeat_penalty: Penalty applied to repeated tokens (1.0 = none).

        Returns:
            A dictionary with the following keys:
                - "generated_text": str, the generated assistant message
                  content, stripped of leading/trailing whitespace.
                - "prompt_tokens": int, tokens consumed by the prompt.
                - "completion_tokens": int, tokens generated in the response.
                - "total_tokens": int, prompt_tokens + completion_tokens.
                - "runtime_sec": float, wall-clock generation time in seconds.
                - "tokens_per_sec": float, completion_tokens / runtime_sec.

        Raises:
            BenchmarkModelError: If the model is not loaded, or generation
                fails for any reason (timeout, malformed response, etc).
        """
        if not self.is_loaded:
            raise BenchmarkModelError(
                "Model is not loaded. Call load_model() before generate()."
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start_time = time.perf_counter()
        try:
            response = self._llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                repeat_penalty=repeat_penalty,
            )
        except Exception as exc:
            raise BenchmarkModelError(f"Generation failed: {exc}") from exc
        runtime_sec = time.perf_counter() - start_time

        try:
            choices = response.get("choices", [])
            if not choices:
                raise BenchmarkModelError(
                    "create_chat_completion returned no choices."
                )
            message = choices[0].get("message", {})
            generated_text = message.get("content", "") or ""
            generated_text = generated_text.strip()

            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            # Some llama.cpp builds/bindings don't populate usage stats.
            # Fall back to direct tokenization so metrics are never zeroed
            # out silently in the CSV output.
            if completion_tokens == 0 and generated_text:
                completion_tokens = len(
                    self._llm.tokenize(generated_text.encode("utf-8"))
                )
            if prompt_tokens == 0:
                prompt_tokens = len(self._llm.tokenize(prompt.encode("utf-8")))

            total_tokens = usage.get(
                "total_tokens", prompt_tokens + completion_tokens
            )
        except BenchmarkModelError:
            raise
        except Exception as exc:
            raise BenchmarkModelError(
                f"Malformed response from create_chat_completion: {exc}"
            ) from exc

        tokens_per_sec = (
            completion_tokens / runtime_sec if runtime_sec > 0 else 0.0
        )

        return {
            "generated_text": generated_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "runtime_sec": runtime_sec,
            "tokens_per_sec": tokens_per_sec,
        }

    def close_model(self) -> None:
        """
        Release the underlying Llama model and free associated resources.

        Safe to call multiple times; no-op if the model is already closed
        or was never loaded. Forces garbage collection afterward to help
        release native (GPU/CPU) memory held by llama.cpp promptly, which
        matters when looping over many model/quant combinations in one
        process (e.g. via run.py).
        """
        if self._llm is not None:
            try:
                del self._llm
            finally:
                self._llm = None
                gc.collect()

    def __enter__(self) -> "BenchmarkModel":
        """Support `with BenchmarkModel(...) as m:` usage; loads the model."""
        self.load_model()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Ensure the model is closed when exiting a context manager block."""
        self.close_model()
