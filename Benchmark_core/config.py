"""
benchmark_core/config.py

Global configuration for all benchmark runs.
Never edit these values anywhere else — import this module instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    seed: int = 42
    n_ctx: int = 4096
    repeat_penalty: float = 1.0

    execution_timeout: float = 10.0  # seconds; used when running generated code

    # Passed directly to model.generate(); empty means no early stopping
    stop: tuple[str, ...] = ()

    # Forwarded to BenchmarkModel(verbose=...) — suppresses llama.cpp output by default
    verbose: bool = False

    # When True, each row is flushed and fsynced immediately (crash-safe but slower)
    csv_flush_every_row: bool = True


DEFAULT_CONFIG = BenchmarkConfig()
