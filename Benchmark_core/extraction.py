"""
benchmark_core/extraction.py

Utilities for extracting clean, executable Python code from raw chat
completions produced by instruction-tuned models (Llama-3-Instruct,
Qwen2.5-Instruct, etc).

Instruct models routinely wrap code in markdown fences, prepend or append
natural-language explanations, restate the function signature, or repeat
the same code block twice (once "shown" and once "explained"). None of
that should reach the benchmark evaluator — only the final, executable
Python code should.

Public API:
    extract_code(raw_text: str) -> str
        The single entry point benchmarks should call. Internally chains
        the smaller helpers below in a fixed order.

Smaller helpers (exposed individually for testing / reuse):
    strip_code_fences(text: str) -> str
    remove_trailing_explanation(text: str) -> str
    remove_duplicate_blocks(text: str) -> str
    dedent_preserve_relative_indentation(text: str) -> str
"""

from __future__ import annotations

import ast
import re
from textwrap import dedent
from typing import List

# Matches a fenced code block, optionally tagged with a language
# (```python, ```py, or bare ```), capturing the inner content.
_FENCE_RE = re.compile(
    r"```\s*(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Lines that are almost certainly prose, not Python, e.g.
# "Here is the function:", "This solution works by...", "Explanation:"
#
# Broadened (vs. the original anchor-only version) to also catch common
# unfenced lead-ins like "Here is the completed Python function:" or
# "Sure! Here is the implementation:", which the original pattern missed
# because it required an exact "here's"/"here is" immediately followed
# by one of a fixed set of nouns. This only matters for RAW (unfenced)
# completions -- when a ```python fence is present, strip_code_fences()
# already discards everything outside the fence, so this regex never
# even sees that text.
_PROSE_LINE_RE = re.compile(
    r"""^\s*
    (?:
        here.*\b(?:function|implementation|solution|code)\b.* |
        sure.* |
        explanation\s*: |
        note\s*: |
        the\s+(?:function|code|solution|implementation)\s+(?:above|below)\b |
        i\s+(?:hope|used|implemented)\b |
        ```
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Heuristics for "this line is clearly natural language, not Python".
# Used to trim trailing explanation paragraphs once we're past the last
# real code line.
_NL_SENTENCE_RE = re.compile(
    r"^[A-Z][a-zA-Z ,'\"]*\b(the|this|it|you|we|note|in)\b.*[.!]\s*$"
)


def strip_code_fences(text: str) -> str:
    """
    Extract the contents of markdown code fences, if any are present.

    If one or more ```python ... ``` (or bare ``` ... ```) blocks exist,
    their contents are concatenated (separated by a blank line) and
    returned. If no fences are found, the original text is returned
    unchanged, on the assumption the model emitted raw code directly.

    Examples:
        >>> strip_code_fences("```python\\ndef f():\\n    return 1\\n```")
        'def f():\\n    return 1'

        >>> strip_code_fences(
        ...     "Sure!\\n```python\\ndef f():\\n    return 1\\n```\\nDone."
        ... )
        'def f():\\n    return 1'

        >>> strip_code_fences("def f():\\n    return 1")
        'def f():\\n    return 1'
    """
    matches = _FENCE_RE.findall(text)
    if not matches:
        return text.strip("\n")

    blocks = [m.strip("\n") for m in matches if m.strip()]
    return "\n\n".join(blocks).strip("\n")


def remove_trailing_explanation(text: str) -> str:
    """
    Drop prose that follows the last line of actual code.

    Walks the text line by line and keeps track of the last line that
    looks like Python (i.e. not an obvious natural-language sentence or
    a "Here is...", "Explanation:", etc. lead-in). Everything after the
    last code-like line is discarded.

    This is intentionally conservative: blank lines and lines that look
    code-like (start with whitespace, end with ':', contain '=', '(', etc.)
    are kept; only clearly prose-like trailing lines are dropped.

    Examples:
        >>> remove_trailing_explanation(
        ...     "def f():\\n    return 1\\n\\nThis function returns one."
        ... )
        'def f():\\n    return 1'

        >>> remove_trailing_explanation(
        ...     "Here is the solution:\\ndef f():\\n    return 1"
        ... )
        'def f():\\n    return 1'

        >>> remove_trailing_explanation("def f():\\n    return 1")
        'def f():\\n    return 1'
    """
    lines = text.split("\n")

    # Drop leading prose lines (e.g. "Here is the function:")
    start = 0
    while start < len(lines) and _PROSE_LINE_RE.match(lines[start]):
        start += 1
    lines = lines[start:]

    # Find the last line index that looks like code.
    last_code_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _looks_like_code_line(stripped):
            last_code_idx = i
        elif _NL_SENTENCE_RE.match(stripped) and last_code_idx != -1:
            # Clearly a trailing natural-language sentence after code
            # has already started; don't extend last_code_idx for this.
            continue

    if last_code_idx == -1:
        # Nothing looked code-like; return as-is (caller's problem).
        return text.strip()

    return "\n".join(lines[: last_code_idx + 1]).rstrip()


def _looks_like_code_line(stripped_line: str) -> bool:
    """
    Heuristic: does a single stripped line look like Python source?

    Used by remove_trailing_explanation to find where code ends and
    prose begins. Errs toward classifying ambiguous lines as code,
    since false negatives (treating code as prose) are more destructive
    than false positives.
    """
    code_markers = (
        "def ", "class ", "return", "import ", "from ", "if ", "elif ",
        "else", "for ", "while ", "try", "except", "finally", "with ",
        "yield", "raise", "pass", "break", "continue", "assert ",
        "@", "#", "lambda ",
    )
    if stripped_line.startswith(code_markers):
        return True
    if stripped_line.endswith((":", "(", ")", "]", "}", ",", "\\")):
        return True
    if any(tok in stripped_line for tok in ("=", "(", ")", "[", "]")):
        # Avoid misclassifying normal prose that happens to contain a
        # parenthetical, e.g. "This works (in most cases)." — require
        # absence of terminal sentence punctuation typical of prose.
        if not stripped_line.endswith((".", "!", "?")):
            return True
    return False


def _split_top_level_units(text: str) -> List[str]:
    """
    Split text into "top-level units": chunks starting at each line whose
    first character is non-whitespace (e.g. `def `, `class `, a top-level
    `#` comment, or a bare statement). Indented lines and blank lines are
    attached to the preceding unit, since they're part of its body.

    This is the key fix that makes deduplication safe for real Python:
    a function's internal blank lines (e.g. between two nested defs, or
    for PEP 8 spacing) never cause it to be split into multiple "blocks",
    so a single top-level function maps to exactly one unit regardless of
    how many blank lines it contains.
    """
    indices = [m.start() for m in re.finditer(r"(?m)^(?=\S)", text)]
    if not indices:
        return [text] if text.strip() else []
    indices.append(len(text))
    return [text[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]


def remove_duplicate_blocks(text: str) -> str:
    """
    Collapse repeated top-level code units (functions, classes, or
    standalone statements) into a single occurrence, including when an
    entire multi-function snippet is duplicated wholesale.

    Splitting happens on *top-level* boundaries (lines starting at column
    0 with a non-whitespace character) rather than on blank lines, so a
    function's internal blank lines never fragment it into separate
    "blocks" that fail to match their duplicate counterpart. This also
    means that if a model repeats an entire snippet of several functions
    back-to-back, each function is deduped independently and the whole
    repeated snippet collapses correctly — not just exact single-block
    repeats.

    Examples:
        >>> remove_duplicate_blocks(
        ...     "def f():\\n    return 1\\n\\ndef f():\\n    return 1"
        ... )
        'def f():\\n    return 1'

        >>> # Blank line WITHIN a function body does not cause it to be
        >>> # split into separate dedup units (functionally unchanged,
        >>> # though exact blank-line count between units is normalized):
        >>> remove_duplicate_blocks(
        ...     "def helper():\\n    return 1\\n\\n\\n"
        ...     "def solve():\\n    return helper()"
        ... )
        'def helper():\\n    return 1\\n\\ndef solve():\\n    return helper()'

        >>> # Whole multi-function snippet duplicated wholesale collapses
        >>> # correctly, not just single repeated blocks:
        >>> doubled = (
        ...     "def helper():\\n    return 1\\n\\n"
        ...     "def solve():\\n    return helper()\\n\\n"
        ...     "def helper():\\n    return 1\\n\\n"
        ...     "def solve():\\n    return helper()"
        ... )
        >>> remove_duplicate_blocks(doubled)
        'def helper():\\n    return 1\\n\\ndef solve():\\n    return helper()'
    """
    units = _split_top_level_units(text.strip("\n"))
    seen: List[str] = []
    deduped: List[str] = []
    for unit in units:
        normalized = unit.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.append(normalized)
        deduped.append(unit.strip("\n"))
    return "\n\n".join(deduped).strip("\n")


def strip_main_guard(text: str) -> str:
    """
    Remove a trailing `if __name__ == "__main__":` block and its body.

    Optional helper — NOT called by default inside extract_code, since
    benchmarks differ on whether a `__main__` guard is unwanted noise
    (HumanEval+/MBPP, which call the function directly) or legitimate
    structure (a future script-style benchmark). Call this explicitly
    when you know the target benchmark wants a bare function/class only.

    Only strips a `__main__` guard that appears at the top level
    (column 0) and removes everything from that line to the end of the
    text, since in practice models place it last and nothing meaningful
    follows it.

    Examples:
        >>> strip_main_guard(
        ...     'def add(a, b):\\n    return a + b\\n\\n'
        ...     'if __name__ == "__main__":\\n    print(add(1, 2))'
        ... )
        'def add(a, b):\\n    return a + b'

        >>> strip_main_guard("def add(a, b):\\n    return a + b")
        'def add(a, b):\\n    return a + b'
    """
    match = re.search(r'(?m)^if\s+__name__\s*==\s*["\']__main__["\']\s*:', text)
    if not match:
        return text.strip("\n")
    return text[: match.start()].rstrip("\n")


def dedent_preserve_relative_indentation(text: str) -> str:
    """
    Remove any common leading whitespace while preserving the relative
    indentation structure of the code (i.e. nested blocks stay nested).

    This guards against models that indent an entire snippet by a
    constant offset (e.g. because it was pasted inside a markdown list
    or a quoted block), which would otherwise cause an IndentationError
    at the top level.

    Examples:
        >>> dedent_preserve_relative_indentation(
        ...     "    def f():\\n        return 1"
        ... )
        'def f():\\n    return 1'

        >>> dedent_preserve_relative_indentation(
        ...     "def f():\\n    return 1"
        ... )
        'def f():\\n    return 1'
    """
    return dedent(text).strip()


def _has_top_level_function(text: str) -> bool:
    """
    Return True only if `text` parses as valid Python AND contains at
    least one top-level function definition (sync or async).

    This guards against the case where every earlier stage happily
    passes through a syntactically-plausible-looking *fragment* that is
    not actually a complete, executable unit -- e.g. a lone
    `return number - int(number)` line. Such a fragment starts with a
    recognized code marker (`return`), so `_looks_like_code_line` keeps
    it, `remove_trailing_explanation` doesn't touch it, and nothing else
    in the pipeline notices anything is wrong. It's only a `def` scan
    that actually catches this class of bug, since a bare `return`
    outside a function is not caught by `ast.parse` at parse time -- it
    only fails at compile time -- so it cannot be relied upon to raise
    SyntaxError.

    Returns False (rather than raising) on any SyntaxError, since the
    caller treats "unparseable" and "no function" the same way: discard.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in tree.body
    )


def extract_code(raw_text: str, remove_main_guard: bool = False) -> str:
    """
    Full extraction pipeline: turn a raw model completion into clean,
    executable Python source.

    Pipeline order:
        1. strip_code_fences          - pull code out of ``` blocks
        2. remove_trailing_explanation - drop prose before/after code
        3. remove_duplicate_blocks    - collapse repeated blocks
        4. dedent_preserve_relative_indentation - normalize indentation
        5. _has_top_level_function guard - reject fragments with no def

    This order matters: fences must be stripped first so that
    remove_trailing_explanation operates on code-only text (otherwise a
    line like "```" or "Explanation:" outside the fence would be
    misread as the end of the snippet). Deduping runs after trailing
    explanation removal so identical code copies introduced via
    multiple fences are caught. Dedent runs last (before the function
    guard) since join operations above can reintroduce blank lines but
    never change indentation.

    The final guard step exists because none of steps 1-4 verify that
    what survives is a *complete* function -- only that it looks
    line-by-line like code. A bare `return number - int(number)` (no
    enclosing `def`) passes every earlier stage untouched and previously
    reached the evaluator as-is, causing
    `SyntaxError: 'return' outside function` at exec() time. Rather than
    letting that meaningless fragment through, extract_code now returns
    "" for any completion that doesn't parse into at least one top-level
    function definition. This does not repair incomplete generations
    (that's not fixable after the fact) -- it just makes the failure
    show up as an empty extraction instead of a confusing runtime
    SyntaxError, and prevents a fragment from silently being scored.

    Works for both Llama-3-Instruct and Qwen2.5-Instruct outputs: the
    two families differ mainly in *how* they wrap explanations (Llama-3
    tends to prepend "Here's the solution:", Qwen tends to append
    "This function works by..."), and the pipeline's prose-line and
    trailing-explanation heuristics are not family-specific.

    Args:
        raw_text: The raw `generated_text` string returned by
            BenchmarkModel.generate().
        remove_main_guard: If True, strip a trailing
            `if __name__ == "__main__":` block before returning. Default
            False, since whether this is unwanted noise (HumanEval+,
            MBPP) or legitimate structure depends on the benchmark.

    Returns:
        A string containing only executable Python code, with no
        markdown fences, no natural-language explanation, no duplicate
        blocks, and normalized (non-offset) indentation. Returns an
        empty string if no code-like content was found, or if what
        remains after cleaning does not contain at least one top-level
        function definition.

    Examples:
        >>> extract_code(
        ...     "Here's the function:\\n"
        ...     "```python\\n"
        ...     "def add(a, b):\\n"
        ...     "    return a + b\\n"
        ...     "```\\n"
        ...     "This adds two numbers together."
        ... )
        'def add(a, b):\\n    return a + b'

        >>> extract_code(
        ...     "```python\\n"
        ...     "def add(a, b):\\n"
        ...     "    return a + b\\n"
        ...     "```\\n"
        ...     "```python\\n"
        ...     "def add(a, b):\\n"
        ...     "    return a + b\\n"
        ...     "```"
        ... )
        'def add(a, b):\\n    return a + b'

        >>> extract_code(
        ...     "    def add(a, b):\\n"
        ...     "        return a + b"
        ... )
        'def add(a, b):\\n    return a + b'

        >>> extract_code("Sorry, I can't help with that.")
        ''

        >>> # Fragment with no enclosing def -- previously returned
        >>> # verbatim and blew up with SyntaxError at exec() time.
        >>> extract_code("return number - int(number)")
        ''
    """
    text = strip_code_fences(raw_text)
    text = remove_trailing_explanation(text)
    text = remove_duplicate_blocks(text)
    text = dedent_preserve_relative_indentation(text)

    if remove_main_guard:
        text = strip_main_guard(text)

    if not any(_looks_like_code_line(ln.strip()) for ln in text.split("\n") if ln.strip()):
        return ""

    if not _has_top_level_function(text):
        return ""

    return text


# ---------------------------------------------------------------------------
# Inline unit-test examples (not executed automatically; run via pytest if
# this module is wrapped in a test file, e.g. tests/test_extraction.py).
# ---------------------------------------------------------------------------
#
# def test_basic_fence_extraction():
#     raw = "Here's the code:\n```python\ndef add(a, b):\n    return a + b\n```"
#     assert extract_code(raw) == "def add(a, b):\n    return a + b"
#
# def test_trailing_explanation_removed():
#     raw = (
#         "```python\n"
#         "def square(x):\n"
#         "    return x * x\n"
#         "```\n"
#         "This function squares the input and returns the result."
#     )
#     assert extract_code(raw) == "def square(x):\n    return x * x"
#
# def test_duplicate_block_collapsed():
#     raw = (
#         "```python\n"
#         "def double(x):\n"
#         "    return x * 2\n"
#         "```\n\n"
#         "```python\n"
#         "def double(x):\n"
#         "    return x * 2\n"
#         "```"
#     )
#     assert extract_code(raw) == "def double(x):\n    return x * 2"
#
# def test_no_fence_raw_code_passthrough():
#     raw = "def identity(x):\n    return x"
#     assert extract_code(raw) == "def identity(x):\n    return x"
#
# def test_qwen_style_lead_in_and_trailing_summary():
#     raw = (
#         "Sure, here is the implementation:\n"
#         "```python\n"
#         "def is_even(n):\n"
#         "    return n % 2 == 0\n"
#         "```\n"
#         "In summary, this checks whether n is divisible by two."
#     )
#     assert extract_code(raw) == "def is_even(n):\n    return n % 2 == 0"
#
# def test_offset_indentation_normalized():
#     raw = "```python\n    def f():\n        return 1\n```"
#     assert extract_code(raw) == "def f():\n    return 1"
#
# def test_refusal_returns_empty_string():
#     raw = "I'm sorry, but I can't help with that request."
#     assert extract_code(raw) == ""
#
# def test_multiline_function_with_helper_preserves_nesting():
#     raw = (
#         "```python\n"
#         "def helper(x):\n"
#         "    if x > 0:\n"
#         "        return x\n"
#         "    else:\n"
#         "        return -x\n"
#         "\n"
#         "def main(x):\n"
#         "    return helper(x) + 1\n"
#         "```"
#     )
#     expected = (
#         "def helper(x):\n"
#         "    if x > 0:\n"
#         "        return x\n"
#         "    else:\n"
#         "        return -x\n"
#         "\n"
#         "def main(x):\n"
#         "    return helper(x) + 1"
#     )
#     assert extract_code(raw) == expected
#
# def test_bare_return_fragment_no_def_rejected():
#     # This is the exact bug from the FP16/Q8 SafetyBench analysis:
#     # a bare `return` with no enclosing `def` passes every line-level
#     # heuristic but is not valid, complete Python.
#     raw = "return number - int(number)"
#     assert extract_code(raw) == ""
#
# def test_unfenced_verbose_leadin_stripped():
#     # Broadened _PROSE_LINE_RE catches lead-ins the original anchor
#     # missed, but only matters for UNFENCED raw completions -- fenced
#     # text never reaches this regex since strip_code_fences() already
#     # discards everything outside the fence.
#     raw = (
#         "Here is the completed Python function:\n"
#         "def add(a, b):\n"
#         "    return a + b"
#     )
#     assert extract_code(raw) == "def add(a, b):\n    return a + b"
