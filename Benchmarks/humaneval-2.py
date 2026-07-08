"""
benchmarks/humaneval.py

HumanEval benchmark runner (openai/openai_humaneval).

Evaluates pass@1 via sandboxed execution with a configurable timeout.
Writes one CSV row per prompt incrementally via CSVLogger.
Reports pass@1, average runtime, and tokens/sec at the end.

NOTE: This runner uses the original HumanEval benchmark, not HumanEval+.
      Results are directly comparable with published pass@1 numbers.
      If HumanEval+ is needed later, use the official EvalPlus evaluator
      rather than reconstructing its test harness manually.

Usage (via run.py — do not call directly):
    python run.py --benchmark humaneval --model <path> ...
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

from benchmark_core.config import DEFAULT_CONFIG, BenchmarkConfig
from benchmark_core.csv_logger import CSVLogger
from benchmark_core.execution import run_python_tests
from benchmark_core.extraction import extract_code
from benchmark_core.gpu import get_gpu_memory_mb
from benchmark_core.model import BenchmarkModel
from benchmark_core.prompts import (
    humaneval_prompt,
    CODING_SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_dataset() -> Dict[str, Any]:
    """
    Return {task_id: problem_dict} from openai/openai_humaneval.

    Each problem_dict contains at minimum:
        task_id       str   e.g. "HumanEval/0"
        prompt        str   the function signature + docstring
        test          str   the check() function source
        entry_point   str   name of the function under test
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets package not found. Install it with: pip install datasets"
        ) from exc

    try:
        ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load openai/openai_humaneval from HuggingFace: {exc}"
        ) from exc

    problems = {row["task_id"]: row for row in ds}
    print(f"[humaneval] Loaded openai/openai_humaneval ({len(problems)} problems)")
    return problems


# ---------------------------------------------------------------------------
# Test harness extraction
# ---------------------------------------------------------------------------

def _build_test_code(problem: Dict[str, Any]) -> str:
    """
    Return a runnable test string from a HumanEval problem dict.

    The dataset provides:
        problem["test"]         — source of a check() function
        problem["entry_point"]  — name of the function under test

    check() is defined but never called in the raw dataset, so we
    append the call ourselves if it's missing.
    """
    test_str: str = problem.get("test", "").strip()
    entry: str = problem.get("entry_point", "").strip()

    if not test_str:
        return ""

    # Append the check() call if the test source doesn't already include it
    if entry and f"check({entry})" not in test_str:
        test_str += f"\ncheck({entry})"

    return test_str


# ---------------------------------------------------------------------------
# Model output cleaning
# ---------------------------------------------------------------------------

def clean_model_output(text: str) -> str:
    """
    Strip markdown code fences and common conversational prefixes that
    instruct-tuned models tend to prepend before the actual code, e.g.
    "Here is the completed function:\n```python\n...". extract_code()
    is a code-extraction step, not a prose-stripping step, so this
    cleanup happens first and the cleaned text is what both extraction
    and logging operate on.
    """
    text = text.strip()

    # Remove common English prefixes first — they often appear *before*
    # a markdown fence (e.g. "Here is the completed function:\n```python"),
    # so stripping fences first would leave a dangling opening fence.
    prefixes = [
        r"here is the completed function:?",
        r"here is the completed python function:?",
        r"here is the complete python function:?",
        r"sure,?\s*here.*?:",
        r"this function.*?:",
    ]
    pattern = r"^\s*(?:" + "|".join(prefixes) + r")\s*"
    text = re.sub(
        pattern,
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove markdown code fences
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Executable program construction (prompt + completion, official protocol)
# ---------------------------------------------------------------------------
#
# extract_code() (benchmark_core/extraction.py) is a *code-extraction*
# step, not a *program-assembly* step: it hands back a string containing
# one or more complete top-level function definitions (its final guard
# stage explicitly rejects anything that doesn't parse into at least one
# ast.FunctionDef/AsyncFunctionDef), or "" if nothing usable was found.
# It never returns a bare function body, and it never returns partial /
# signature-less fragments -- so `solution` here is always either "" or
# one-or-more complete `def ...` blocks.
#
# HumanEval prompts, in turn, are usually valid standalone Python: any
# helper functions the target depends on (e.g. `poly()` before
# `find_zero()`, or `encode_cyclic()` before `decode_cyclic()`) are fully
# defined, and the entry-point function itself is present as a stub with
# only a docstring body (a string-literal expression statement is a
# syntactically valid function body, so the whole prompt parses cleanly
# on its own).
#
# The official HumanEval protocol executes prompt + completion + test
# together, precisely so those helper definitions are in scope. The
# previous implementation here only executed `solution` + test_code,
# silently dropping any helper the model didn't reproduce itself --
# hence NameError: poly/encode_cyclic/encode_shift is not defined.
#
# The one thing we must not do is naively concatenate prompt_text +
# solution: since instruct models are prompted to emit the *complete*
# function (not just a body continuation), `solution` typically restates
# the entry-point signature itself, and sometimes restates a helper too.
# Naive concatenation wouldn't error (a later `def` simply shadows an
# earlier one of the same name at module scope), but it would violate
# the "don't duplicate" requirement and leave dead, confusing code in
# the executed program. So we parse both sides, and only keep the
# prompt's top-level function/class definitions whose name isn't already
# (re)defined in `solution`.


def _build_executable_program(prompt_text: str, solution: str) -> str:
    """
    Combine the original HumanEval prompt with the model's extracted
    completion into one executable program, following the official
    HumanEval evaluation procedure (prompt + completion), while avoiding
    duplicate definitions when the completion already restates a
    signature or helper function from the prompt.

    Args:
        prompt_text: The original, unmodified HumanEval prompt string
            (problem["prompt"]) — i.e. exactly what was preserved and
            sent to the model via humaneval_prompt(). Not the model's
            output.
        solution: The output of extract_code(clean_model_output(...)):
            either "" or one-or-more complete top-level function
            definitions.

    Returns:
        A single string of Python source: the prompt's helper
        definitions (imports, constants, and any function/class the
        completion does NOT already redefine) followed by `solution`.
        Exact prompt source text is preserved verbatim via
        ast.get_source_segment, so indentation and formatting inside
        kept helper functions are untouched.

        Falls back to returning `solution` unchanged if the prompt or
        the solution doesn't parse as standalone Python — this matches
        the previous (pre-fix) execution behaviour for that edge case
        rather than raising, since a prompt/solution that doesn't parse
        was already going to fail the test either way.
    """
    try:
        prompt_tree = ast.parse(prompt_text)
    except SyntaxError:
        return solution

    if not solution:
        # Nothing was extracted; there's no completion to append, and
        # running the prompt alone (docstring-only stub) will simply
        # fail the tests as before -- no NameError risk either way.
        return solution

    try:
        solution_tree = ast.parse(solution)
    except SyntaxError:
        return solution

    # Names the completion already defines in full — the prompt's
    # versions of these (its entry-point stub, and/or any helper the
    # model chose to restate) must be left out to avoid duplication.
    solution_names = {
        node.name
        for node in solution_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    kept_segments = []
    for node in prompt_tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in solution_names
        ):
            continue
        segment = ast.get_source_segment(prompt_text, node)
        if segment is not None:
            kept_segments.append(segment)

    return "\n\n".join(kept_segments + [solution])


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
    Run HumanEval pass@1 benchmark.

    Args:
        model:       Initialised BenchmarkModel (already loaded).
        output_path: Path for the output CSV (created or appended to).
        config:      BenchmarkConfig; defaults to DEFAULT_CONFIG.
        resume:      Skip prompt IDs already present in output_path.

    Returns:
        {
            "score":             float,  # pass@1 over this session's tasks only
            "avg_runtime_sec":   float,
            "avg_tokens_per_sec": float,
        }

    NOTE on resume: when resuming a partial run, the returned score covers
    only tasks completed in this session. Compute the final pass@1 from
    the complete CSV after all tasks finish:
        rows = CSVLogger(output_path).read_rows()
        score = sum(int(r["correct"]) for r in rows) / len(rows)
    """
    problems = _load_dataset()
    logger = CSVLogger(output_path, overwrite=not resume)
    already_done = logger.read_logged_prompt_ids(benchmark="humaneval") if resume else set()

    if already_done:
        print(
            f"[humaneval] Resuming — skipping {len(already_done)} already-logged problems.\n"
            f"            Returned score covers this session only; "
            f"compute final pass@1 from the complete CSV."
        )

    total = correct = successful_generations = 0
    total_runtime = total_tps = 0.0

    for task_id, problem in problems.items():
        if task_id in already_done:
            continue

        prompt_text: str = problem.get("prompt", "")
        test_code: str = _build_test_code(problem)
        gpu_mb: Optional[float] = get_gpu_memory_mb()

        try:
            	result = model.generate(
   			prompt=humaneval_prompt(prompt_text),
    			system_prompt=CODING_SYSTEM_PROMPT,
    			max_tokens=config.max_tokens,
    			temperature=config.temperature,
    			top_p=config.top_p,
   		 	repeat_penalty=config.repeat_penalty,
    			stop=list(config.stop) or None,
		)
        except Exception as exc:
            print(f"[humaneval] ✗ {task_id:<30} generation_error: {exc}")
            # Unspecified columns (model_output, tokens, runtime, etc.)
            # are filled as "" by make_row() — schema is satisfied.
            row = CSVLogger.make_row(
                model=model.model_name,
                model_family=model.model_family,
                model_size=model.model_size,
                quant=model.quant,
                quant_bits=model.quant_bits,
                benchmark="humaneval",
                capability="coding",
                prompt_id=task_id,
                prompt=textwrap.shorten(prompt_text, width=500, placeholder="..."),
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

        cleaned_output = clean_model_output(raw_output)
        solution = extract_code(cleaned_output)

        # Official HumanEval protocol executes prompt + completion
        # together, so helper functions the prompt defines (e.g. poly(),
        # encode_cyclic(), encode_shift()) that the target function
        # depends on are in scope, without duplicating anything the
        # completion already redefines itself.
        executable_program = _build_executable_program(prompt_text, solution)

        if test_code:
            try:
                passed, fail_reason = run_python_tests(executable_program, test_code, config.execution_timeout)
            except Exception as exc:
                passed, fail_reason = False, f"execution_error: {exc}"
        else:
            passed, fail_reason = False, "no_test_code"

        total += 1
        if passed:
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
            benchmark="humaneval",
            capability="coding",
            prompt_id=task_id,
            prompt=textwrap.shorten(prompt_text, width=500, placeholder="..."),
            model_output=cleaned_output[:5000],
            expected_answer="",
            correct=bool(passed),
            fail_reason=fail_reason,
            score_overall=1.0 if passed else 0.0,
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

        status = "✓" if passed else "✗"
        print(
            f"[humaneval] {status} {task_id:<30} "
            f"runtime={runtime:.2f}s  tps={tps:.1f}  "
            f"pass@1={correct}/{total}"
        )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    score = correct / total if total > 0 else 0.0
    avg_runtime = total_runtime / successful_generations if successful_generations > 0 else 0.0
    avg_tps = total_tps / successful_generations if successful_generations > 0 else 0.0

    print(
        f"\n[humaneval] DONE — "
        f"pass@1={score:.4f} ({correct}/{total})  "
        f"avg_runtime={avg_runtime:.2f}s  "
        f"avg_tok/s={avg_tps:.1f}"
    )

    return {
        "score": score,
        "avg_runtime_sec": avg_runtime,
        "avg_tokens_per_sec": avg_tps,
    }
