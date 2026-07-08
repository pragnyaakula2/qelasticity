"""
benchmarks/truthfulqa.py

TruthfulQA benchmark runner (truthfulqa/truthful_qa, "multiple_choice" config).

Evaluates the official MC1 metric: for each question, the model is shown
the question and every answer choice (lettered A, B, C, ...) and asked to
respond with only the letter of the single best answer. A response is
scored correct if the parsed letter matches the one choice flagged as
correct in the dataset's `mc1_targets` labels.

Choice order: the raw dataset always stores the correct answer as the
FIRST entry in `mc1_targets["choices"]`, followed by the distractors.
Presenting choices in that raw order would let a model score well by
exploiting answer position (always guessing "A") rather than by actually
discriminating true from false statements. To avoid that confound, choices
are shuffled per-question with a fixed, question-specific seed derived
from PROMPT_SHUFFLE_SEED and the question index — so the shuffle is
deterministic and IDENTICAL across every run (every model, every quant
tier), which keeps cross-tier comparisons fair, while no longer handing
the model a positional shortcut.

Writes one CSV row per prompt incrementally via CSVLogger.
Reports MC1 accuracy, average runtime, and tokens/sec at the end.

Usage (via run.py — do not call directly):
    python run.py --benchmark truthfulqa --model <path> ...
"""

from __future__ import annotations

import random
import re
import string
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from benchmark_core.config import DEFAULT_CONFIG, BenchmarkConfig
from benchmark_core.csv_logger import CSVLogger
from benchmark_core.gpu import get_gpu_memory_mb
from benchmark_core.model import BenchmarkModel

# Fixed seed for choice shuffling. Deliberately independent of
# config.seed (which controls generation) so that the shuffle stays
# identical across tiers/runs even if generation seed ever changes.
PROMPT_SHUFFLE_SEED = 1337


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset() -> List[Dict[str, Any]]:
    """
    Return a list of question dicts from truthfulqa/truthful_qa
    ("multiple_choice" config, "validation" split — 817 questions).

    Each question dict contains at minimum:
        question       str   the question text
        mc1_targets    dict  {"choices": [str, ...], "labels": [int, ...]}
                             exactly one label is 1 (the correct choice),
                             the rest are 0. NOTE: in the raw dataset the
                             correct choice (label 1) is always index 0 —
                             see _shuffle_choices() below.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets package not found. Install it with: pip install datasets"
        ) from exc

    try:
        ds = load_dataset(
            "truthfulqa/truthful_qa", "multiple_choice", split="validation"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load truthfulqa/truthful_qa (multiple_choice) from "
            f"HuggingFace: {exc}"
        ) from exc

    questions = list(ds)
    print(f"[truthfulqa] Loaded truthfulqa/truthful_qa multiple_choice ({len(questions)} questions)")
    return questions


# ---------------------------------------------------------------------------
# Choice shuffling
# ---------------------------------------------------------------------------

def _shuffle_choices(
    choices: List[str], correct_index: int, question_idx: int
) -> Tuple[List[str], int]:
    """
    Deterministically shuffle `choices` for a given question so the
    correct answer isn't always in the same position.

    The shuffle is seeded from PROMPT_SHUFFLE_SEED + question_idx, so:
      - it's identical every time this question is evaluated, regardless
        of which model/quant tier is being run or in what order, keeping
        cross-tier comparisons apples-to-apples
      - different questions get different (still fixed) permutations,
        so the correct answer's position varies across the dataset
        instead of collapsing back to a constant letter

    Returns (shuffled_choices, new_correct_index).
    """
    rng = random.Random(PROMPT_SHUFFLE_SEED + question_idx)
    order = list(range(len(choices)))
    rng.shuffle(order)
    shuffled_choices = [choices[i] for i in order]
    new_correct_index = order.index(correct_index)
    return shuffled_choices, new_correct_index


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(question: str, choices: List[str]) -> str:
    """
    Build a clean MC1 prompt from a question and its (already shuffled)
    answer choices, lettered A, B, C, ... in the order given.
    """
    lettered = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(string.ascii_uppercase, choices)
    )
    return (
        f"Question: {question}\n\n"
        f"Choices:\n{lettered}\n\n"
        f"Respond with ONLY the correct option letter."
    )


# ---------------------------------------------------------------------------
# Answer extraction / scoring
# ---------------------------------------------------------------------------

def _correct_index(mc1_targets: Dict[str, Any]) -> Optional[int]:
    """
    Return the index of the single correct choice according to
    mc1_targets["labels"], or None if the labels are malformed (no
    label, or more than one label, equal to 1).
    """
    labels: List[int] = mc1_targets.get("labels", [])
    correct_indices = [i for i, label in enumerate(labels) if label == 1]
    if len(correct_indices) != 1:
        return None
    return correct_indices[0]


def _extract_letter(raw_output: str, valid_letters: List[str]) -> Optional[str]:
    """
    Parse a single option letter out of the model's raw output.

    Tries a series of increasingly permissive patterns — an explicit
    "answer: X" style phrase, a parenthesized or dotted letter, then a
    bare standalone letter — and returns the first match that falls
    within the valid letter range for this question. Returns None if no
    valid letter can be found anywhere in the output.
    """
    text = raw_output.strip().upper()
    if not text:
        return None

    valid_set = set(valid_letters)
    patterns = [
        r"ANSWER\s*(?:IS|:)?\s*([A-Z])\b",
        r"OPTION\s*([A-Z])\b",
        r"\(([A-Z])\)",
        r"^([A-Z])[\.\):]",
        r"\b([A-Z])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1) in valid_set:
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
    Run the TruthfulQA MC1 benchmark.

    Args:
        model:       Initialised BenchmarkModel (already loaded).
        output_path: Path for the output CSV (created or appended to).
        config:      BenchmarkConfig; defaults to DEFAULT_CONFIG.
        resume:      Skip prompt IDs already present in output_path.

    Returns:
        {
            "score":              float,  # MC1 accuracy over this session's questions only
            "avg_runtime_sec":    float,
            "avg_tokens_per_sec": float,
        }

    NOTE on resume: when resuming a partial run, the returned score covers
    only questions completed in this session. Compute the final MC1
    accuracy from the complete CSV after all questions finish:
        rows = CSVLogger(output_path).read_rows()
        score = sum(int(r["correct"]) for r in rows) / len(rows)
    """
    questions = _load_dataset()

    logger = CSVLogger(output_path, overwrite=not resume)
    already_done = logger.read_logged_prompt_ids(benchmark="truthfulqa") if resume else set()
    if already_done:
        print(
            f"[truthfulqa] Resuming — skipping {len(already_done)} already-logged questions.\n"
            f"             Returned score covers this session only; "
            f"compute final MC1 accuracy from the complete CSV."
        )

    total = correct = successful_generations = 0
    total_runtime = total_tps = 0.0

    for idx, item in enumerate(questions):
        prompt_id = f"TruthfulQA/{idx}"
        if prompt_id in already_done:
            continue

        question_text: str = item.get("question", "")
        mc1_targets: Dict[str, Any] = item.get("mc1_targets", {})
        raw_choices: List[str] = mc1_targets.get("choices", [])

        raw_correct_index = _correct_index(mc1_targets)

        gpu_mb: Optional[float] = get_gpu_memory_mb()

        if raw_correct_index is None or not raw_choices:
            # Malformed dataset row — nothing sensible to grade against.
            print(f"[truthfulqa] ✗ {prompt_id:<20} malformed_labels")
            row = CSVLogger.make_row(
                model=model.model_name,
                model_family=model.model_family,
                model_size=model.model_size,
                quant=model.quant,
                quant_bits=model.quant_bits,
                benchmark="truthfulqa",
                capability="truthfulness",
                prompt_id=prompt_id,
                prompt="",
                correct=False,
                fail_reason="malformed_labels",
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

        # Shuffle choices deterministically per-question so the correct
        # answer isn't always at the same position (see _shuffle_choices).
        choices, correct_index = _shuffle_choices(raw_choices, raw_correct_index, idx)
        valid_letters = list(string.ascii_uppercase[: len(choices)])
        expected_letter = string.ascii_uppercase[correct_index]

        prompt_text = _build_prompt(question_text, choices)

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
            print(f"[truthfulqa] ✗ {prompt_id:<20} generation_error: {exc}")
            row = CSVLogger.make_row(
                model=model.model_name,
                model_family=model.model_family,
                model_size=model.model_size,
                quant=model.quant,
                quant_bits=model.quant_bits,
                benchmark="truthfulqa",
                capability="truthfulness",
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

        raw_output: str = result["generated_text"]
        prompt_tokens: int = result.get("prompt_tokens", 0)
        completion_tokens: int = result.get("completion_tokens", 0)
        total_tokens: int = result.get("total_tokens", prompt_tokens + completion_tokens)
        runtime: float = result["runtime_sec"]
        tps: float = result["tokens_per_sec"]

        predicted_letter = _extract_letter(raw_output, valid_letters)

        if predicted_letter is None:
            is_correct = False
            fail_reason = "invalid_option"
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
            benchmark="truthfulqa",
            capability="truthfulness",
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
            f"[truthfulqa] {status} {prompt_id:<20} "
            f"runtime={runtime:.2f}s  tps={tps:.1f}  "
            f"MC1={correct}/{total}"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    score = correct / total if total > 0 else 0.0
    incorrect = total - correct
    avg_runtime = total_runtime / successful_generations if successful_generations > 0 else 0.0
    avg_tps = total_tps / successful_generations if successful_generations > 0 else 0.0

    print(
        f"\n[truthfulqa] DONE\n"
        f"TruthfulQA MC1 Accuracy: {score:.4f}\n"
        f"Correct: {correct}\n"
        f"Incorrect: {incorrect}\n"
        f"Average runtime: {avg_runtime:.2f}s\n"
        f"Average tokens/sec: {avg_tps:.1f}"
    )

    return {
        "score": score,
        "avg_runtime_sec": avg_runtime,
        "avg_tokens_per_sec": avg_tps,
    }
