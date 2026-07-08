"""
benchmarks/mmlu.py

MMLU benchmark runner (cais/mmlu, "all" configuration).

Evaluates a fixed, deterministically-sampled subset of 1000 questions from
the test split. Each question is presented with its four answer choices
(A-D); the model is asked to respond with only the letter of the best
answer, which is parsed and compared against the dataset's official
answer index.

Writes one CSV row per question incrementally via CSVLogger.
Reports accuracy, average runtime, and tokens/sec at the end.

Usage (via run.py — do not call directly):
    python run.py --benchmark mmlu --model <path> ...
"""

from __future__ import annotations

import random
import re
import string
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark_core.config import DEFAULT_CONFIG, BenchmarkConfig
from benchmark_core.csv_logger import CSVLogger
from benchmark_core.gpu import get_gpu_memory_mb
from benchmark_core.model import BenchmarkModel

# Fixed sample size and seed — every model/quant combination must be
# evaluated on exactly the same 1000 questions in exactly the same order,
# so both are constants rather than parameters.
_SAMPLE_SIZE = 1000
_SAMPLE_SEED = 42

_LETTERS = string.ascii_uppercase[:4]  # "ABCD" — MMLU is always 4-way MC


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset() -> List[Dict[str, Any]]:
    """
    Return a deterministic 1000-question sample from cais/mmlu
    ("all" configuration, "test" split).

    Sampling uses a local `random.Random(_SAMPLE_SEED)` instance (rather
    than seeding the global `random` module) so the same 1000 questions
    (in the same order) are selected on every run and for every
    model/quant combination — required for apples-to-apples comparisons
    across evaluations — without side effects on any other code in the
    process that also uses `random`.

    Each returned question dict contains:
        question         str        the question text
        choices          List[str]  exactly 4 answer choices
        answer           int        index (0-3) of the correct choice
        subject          str        MMLU subject/category
        _dataset_index   int        the row's index in the original,
                                     unsampled test split — kept so each
                                     prompt_id can be traced back to the
                                     exact source question
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets package not found. Install it with: pip install datasets"
        ) from exc

    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load cais/mmlu (all) from HuggingFace: {exc}"
        ) from exc

    if len(ds) < _SAMPLE_SIZE:
        raise RuntimeError(
            f"cais/mmlu test split has only {len(ds)} rows; "
            f"cannot sample {_SAMPLE_SIZE}."
        )

    rng = random.Random(_SAMPLE_SEED)
    sampled_indices = rng.sample(range(len(ds)), _SAMPLE_SIZE)

    questions = []
    for dataset_index in sampled_indices:
        item = dict(ds[dataset_index])
        item["_dataset_index"] = dataset_index
        questions.append(item)

    print(
        f"[mmlu] Loaded cais/mmlu 'all' test split ({len(ds)} rows) — "
        f"sampled {len(questions)} questions (seed={_SAMPLE_SEED})"
    )
    return questions


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(question: str, choices: List[str]) -> str:
    """
    Build the 4-way multiple-choice prompt for a single MMLU question.

    NOTE: the instruction line intentionally avoids re-listing the answer
    letters (e.g. "A\\nB\\nC\\nor D") after the choices. Some models pick up
    on that trailing letter sequence as the nearest salient token and
    default to echoing it (observed as a severe, quantization-independent
    "always answer A" collapse on Mistral-7B-Instruct-v0.3 — confirmed via
    cross-benchmark comparison: SafetyBench and TruthfulQA, which use
    similar-but-not-identical instruction phrasing without a trailing
    letter list, showed no such collapse for the same model/files).
    """
    lettered = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(_LETTERS, choices)
    )
    return (
        f"Question:\n{question}\n\n"
        f"Choices:\n{lettered}\n\n"
        "Respond with only the letter of the correct choice. "
        "Do not provide explanations."
    )


# ---------------------------------------------------------------------------
# Answer extraction / scoring
# ---------------------------------------------------------------------------

def _expected_letter(answer_index: int) -> Optional[str]:
    """Map MMLU's 0-3 answer index to a letter, or None if out of range."""
    if not isinstance(answer_index, int) or not (0 <= answer_index < len(_LETTERS)):
        return None
    return _LETTERS[answer_index]


def _extract_letter(raw_output: str) -> Optional[str]:
    """
    Parse the first valid capital letter (A-D) out of the model's raw
    output.

    Tries a series of increasingly permissive patterns — an explicit
    "answer: X" style phrase, a parenthesized or dotted letter, then a
    bare standalone letter — and returns the first match found. Returns
    None if no A-D letter can be found anywhere in the output.
    """
    text = raw_output.strip().upper()
    if not text:
        return None

    patterns = [
        r"ANSWER\s*(?:IS|:)?\s*([A-D])\b",
        r"OPTION\s*([A-D])\b",
        r"\(([A-D])\)",
        r"^([A-D])[\.\):]",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    model: BenchmarkModel,
    output_path: Path,
    config: BenchmarkConfig = DEFAULT_CONFIG,
    resume: bool = True,
) -> Dict[str, float]:
    """
    Run the MMLU benchmark (1000-question deterministic sample).

    Args:
        model:       Initialised BenchmarkModel (already loaded).
        output_path: Path for the output CSV (created or appended to).
        config:      BenchmarkConfig; defaults to DEFAULT_CONFIG.
        resume:      Skip prompt IDs already present in output_path.

    Returns:
        {
            "score":              float,  # accuracy over this session's questions only
            "avg_runtime_sec":    float,
            "avg_tokens_per_sec": float,
        }

    NOTE on resume: when resuming a partial run, the returned score covers
    only questions completed in this session. Compute the final accuracy
    from the complete CSV after all questions finish:
        rows = CSVLogger(output_path).read_rows()
        score = sum(int(r["correct"]) for r in rows) / len(rows)
    """
    questions = _load_dataset()

    logger = CSVLogger(output_path, overwrite=not resume)
    already_done = logger.read_logged_prompt_ids(benchmark="mmlu") if resume else set()
    if already_done:
        print(
            f"[mmlu] Resuming — skipping {len(already_done)} already-logged questions.\n"
            f"       Returned score covers this session only; "
            f"compute final accuracy from the complete CSV."
        )

    total = correct = successful_generations = 0
    total_runtime = total_tps = 0.0

    for item in questions:
        prompt_id = f"MMLU/{item['_dataset_index']}"
        if prompt_id in already_done:
            continue

        question_text: str = item.get("question", "")
        choices: List[str] = item.get("choices", [])
        answer_index = item.get("answer")

        expected_letter = _expected_letter(answer_index)
        prompt_text = _build_prompt(question_text, choices)

        if expected_letter is None or len(choices) != 4:
            # Malformed dataset row — nothing sensible to grade against.
            print(f"[mmlu] ✗ {prompt_id:<20} malformed_ground_truth")
            row = CSVLogger.make_row(
                model=model.model_name,
                model_family=model.model_family,
                model_size=model.model_size,
                quant=model.quant,
                quant_bits=model.quant_bits,
                benchmark="mmlu",
                capability="reasoning",
                prompt_id=prompt_id,
                prompt=textwrap.shorten(prompt_text, width=500, placeholder="..."),
                correct=False,
                fail_reason="malformed_ground_truth",
                score_overall=0.0,
                seed=config.seed,
                n_ctx=config.n_ctx,
                temperature=config.temperature,
                top_p=config.top_p,
                repeat_penalty=config.repeat_penalty,
            )
            logger.log_row(row)
            total += 1
            continue

        try:
            result = model.generate(
                prompt=prompt_text,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                repeat_penalty=config.repeat_penalty,
                stop=list(config.stop) if config.stop else None,
            )
        except Exception as exc:
            print(f"[mmlu] ✗ {prompt_id:<20} generation_error: {exc}")
            # Unspecified columns (model_output, tokens, runtime, etc.)
            # are filled as "" by make_row() — schema is satisfied.
            row = CSVLogger.make_row(
                model=model.model_name,
                model_family=model.model_family,
                model_size=model.model_size,
                quant=model.quant,
                quant_bits=model.quant_bits,
                benchmark="mmlu",
                capability="reasoning",
                prompt_id=prompt_id,
                prompt=textwrap.shorten(prompt_text, width=500, placeholder="..."),
                expected_answer=expected_letter,
                correct=False,
                fail_reason=f"generation_error: {exc}",
                score_overall=0.0,
                seed=config.seed,
                n_ctx=config.n_ctx,
                temperature=config.temperature,
                top_p=config.top_p,
                repeat_penalty=config.repeat_penalty,
            )
            logger.log_row(row)
            total += 1
            continue

        # Captured immediately after generation so it reflects actual
        # inference memory usage rather than a pre-generation snapshot.
        gpu_mb: Optional[float] = get_gpu_memory_mb()

        raw_output: str = result["generated_text"]
        prompt_tokens: int = result.get("prompt_tokens", 0)
        completion_tokens: int = result.get("completion_tokens", 0)
        total_tokens: int = result.get("total_tokens", prompt_tokens + completion_tokens)
        runtime: float = result["runtime_sec"]
        tps: float = result["tokens_per_sec"]

        predicted_letter = _extract_letter(raw_output)

        if predicted_letter is None:
            is_correct, fail_reason = False, "invalid_option"
        else:
            is_correct = predicted_letter == expected_letter
            fail_reason = (
                ""
                if is_correct
                else f"expected_{expected_letter}_predicted_{predicted_letter}"
            )

        total += 1
        if is_correct:
            correct += 1
        successful_generations += 1
        total_runtime += runtime
        total_tps += tps

        row = CSVLogger.make_row(
            model=model.model_name,
            model_family=model.model_family,
            model_size=model.model_size,
            quant=model.quant,
            quant_bits=model.quant_bits,
            benchmark="mmlu",
            capability="reasoning",
            prompt_id=prompt_id,
            prompt=textwrap.shorten(prompt_text, width=500, placeholder="..."),
            model_output=textwrap.shorten(raw_output, width=2000, placeholder="..."),
            expected_answer=expected_letter,
            correct=bool(is_correct),
            fail_reason=fail_reason,
            score_overall=1.0 if is_correct else 0.0,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            runtime_sec=round(runtime, 4),
            tokens_per_sec=round(tps, 4),
            gpu_memory_mb=gpu_mb if gpu_mb is not None else "",
            seed=config.seed,
            n_ctx=config.n_ctx,
            temperature=config.temperature,
            top_p=config.top_p,
            repeat_penalty=config.repeat_penalty,
        )
        logger.log_row(row)

        status = "✓" if is_correct else "✗"
        print(
            f"[mmlu] {status} {prompt_id:<20} "
            f"runtime={runtime:.2f}s  tps={tps:.1f}  "
            f"accuracy={correct}/{total}"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    score = correct / total if total > 0 else 0.0
    avg_runtime = total_runtime / successful_generations if successful_generations > 0 else 0.0
    avg_tps = total_tps / successful_generations if successful_generations > 0 else 0.0

    print(
        f"\n[mmlu] DONE\n"
        f"Total questions: {total}\n"
        f"Correct: {correct}\n"
        f"Accuracy: {score:.4f}\n"
        f"Average runtime: {avg_runtime:.2f}s\n"
        f"Average tokens/sec: {avg_tps:.1f}"
    )

    return {
        "score": score,
        "avg_runtime_sec": avg_runtime,
        "avg_tokens_per_sec": avg_tps,
    }
