"""Command-line entry point for the GGUF benchmark framework."""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import cast

from benchmark_core.config import DEFAULT_CONFIG, BenchmarkConfig
from benchmark_core.metadata import parse_model_metadata
from benchmark_core.model import BenchmarkModel, BenchmarkModelError

BenchmarkRun = Callable[[BenchmarkModel, Path, BenchmarkConfig, bool], dict[str, float]]

BENCHMARKS: frozenset[str] = frozenset(
    {
        "humaneval",
        "mbpp",
        "gsm8k",
        "mmlu",
        "truthfulqa",
        "safetybench",
    }
)


def build_config(
    *,
    ctx: int,
    seed: int,
    temperature: float,
    top_p: float,
) -> BenchmarkConfig:
    """Create a benchmark configuration with CLI overrides."""
    return replace(
        DEFAULT_CONFIG,
        n_ctx=ctx,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
    )


def build_output_path(
    *,
    output: Path | None,
    benchmark: str,
    model_name: str,
) -> Path:
    """Build the output CSV path and ensure its parent directory exists."""
    path = output or Path("results") / benchmark / f"{Path(model_name).stem}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_benchmark(name: str) -> BenchmarkRun:
    """Lazily import and return the selected benchmark run function."""
    module: ModuleType = importlib.import_module(f"benchmarks.{name}")
    run_func = getattr(module, "run", None)

    if not callable(run_func):
        raise TypeError(f"benchmarks.{name} does not expose a callable run()")

    return cast(BenchmarkRun, run_func)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run a benchmark against a quantized GGUF language model."
    )

    parser.add_argument("--model", required=True, type=Path, help="Path to GGUF model.")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(BENCHMARKS),
        help="Benchmark to run.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    parser.add_argument(
        "--ctx",
        type=int,
        default=DEFAULT_CONFIG.n_ctx,
        help="Context window size.",
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.seed,
        help="Random seed.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CONFIG.temperature,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_CONFIG.top_p,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from an existing output CSV when supported.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose llama-cpp-python logging.",
    )

    return parser


def print_run_header(
    *,
    model_name: str,
    model_family: object,
    model_size: object,
    quant: object,
    benchmark: str,
    output_path: Path,
) -> None:
    """Print the benchmark run details before execution."""
    print("==========================")
    print("Benchmark Run")
    print("==========================")
    print()
    print(f"Model      : {model_name}")
    print(f"Family     : {model_family}")
    print(f"Size       : {model_size}")
    print(f"Quant      : {quant}")
    print(f"Benchmark  : {benchmark}")
    print(f"Output CSV : {output_path}")
    print()


def print_summary(
    *,
    model_name: str,
    benchmark: str,
    output_path: Path,
    metrics: dict[str, float],
) -> None:
    """Print a polished benchmark completion summary."""
    print()
    print("==========================")
    print("Benchmark Complete")
    print("==========================")
    print()
    print(f"Model      : {model_name}")
    print(f"Benchmark  : {benchmark}")
    print(f"Output CSV : {output_path}")
    print()
    print(f"Score              : {metrics.get('score', 0.0):.4f}")
    print(f"Average Runtime    : {metrics.get('avg_runtime_sec', 0.0):.4f} s")
    print(f"Average Tokens/sec : {metrics.get('avg_tokens_per_sec', 0.0):.4f}")


def main() -> None:
    """Parse CLI arguments, run the selected benchmark, and close the model."""
    parser = build_parser()
    args = parser.parse_args()

    metadata = parse_model_metadata(args.model)
    model_name = str(metadata["model_name"])

    config = build_config(
        ctx=args.ctx,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    output_path = build_output_path(
        output=args.output,
        benchmark=args.benchmark,
        model_name=model_name,
    )

    model = BenchmarkModel(
        model_path=args.model,
        n_ctx=config.n_ctx,
        n_gpu_layers=args.gpu_layers,
        seed=config.seed,
        verbose=args.verbose,
    )

    model.model_name = metadata["model_name"]
    model.model_family = metadata["model_family"]
    model.model_size = metadata["model_size"]
    model.quant = metadata["quant"]
    model.quant_bits = metadata["quant_bits"]

    print_run_header(
        model_name=model_name,
        model_family=metadata["model_family"],
        model_size=metadata["model_size"],
        quant=metadata["quant"],
        benchmark=args.benchmark,
        output_path=output_path,
    )

    try:
        benchmark_run = load_benchmark(args.benchmark)
        model.load_model()
        metrics = benchmark_run(model, output_path, config, args.resume)
    except BenchmarkModelError as exc:
        print(f"Model error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception:
        print("Unexpected error while running benchmark:", file=sys.stderr)
        traceback.print_exc(limit=5, file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        try:
            model.close_model()
        except BenchmarkModelError as exc:
            print(f"Model cleanup error: {exc}", file=sys.stderr)

    print_summary(
        model_name=model_name,
        benchmark=args.benchmark,
        output_path=output_path,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()

