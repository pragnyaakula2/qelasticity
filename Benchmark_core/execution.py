"""
benchmark_core/execution.py

Reusable sandboxed Python execution utilities for coding benchmarks.

Provides run_python_tests(), which concatenates a generated solution with a
test harness, syntax-checks the result, and executes it inside an isolated
namespace under a configurable timeout.

Used by HumanEval, MBPP, and any other benchmark that evaluates generated
Python code. Contains no benchmark-specific logic, no dataset loading, and
no model code.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import signal
import traceback
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    """Raised by the SIGALRM handler when execution exceeds the time limit."""


def _sigalrm_handler(signum: int, frame: Any) -> None:
    """SIGALRM signal handler — raises _TimeoutError to interrupt execution."""
    raise _TimeoutError()


def _syntax_check(code: str) -> tuple[bool, str]:
    """
    Parse *code* with ast.parse() and report whether it is syntactically valid.

    Args:
        code: Python source code to check.

    Returns:
        ``(True, "")`` if the code parses without error.
        ``(False, "SyntaxError: <message>")`` otherwise.
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"


def _arm_timeout(timeout: float) -> None:
    """
    Install the SIGALRM handler and schedule an alarm for *timeout* seconds.

    The alarm duration is rounded up to the nearest whole second because
    signal.alarm() accepts only integer seconds.

    On platforms where SIGALRM is unavailable (e.g. Windows), this function
    silently does nothing. Execution will proceed without a hard timeout.

    Args:
        timeout: Maximum allowed execution time in seconds.
    """
    try:
        signal.signal(signal.SIGALRM, _sigalrm_handler)
        signal.alarm(max(1, math.ceil(timeout)))
    except (AttributeError, OSError):
        pass


def _cancel_timeout() -> None:
    """
    Cancel any pending SIGALRM alarm.

    Safe to call even when SIGALRM is unavailable or no alarm is active.
    All exceptions are suppressed so this can be called unconditionally
    from a ``finally`` block.
    """
    with contextlib.suppress(Exception):
        signal.alarm(0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_python_tests(
    solution: str,
    test_code: str,
    timeout: float = 10.0,
) -> tuple[bool, str]:
    """
    Execute *solution* + *test_code* in an isolated namespace and return the result.

    The full source executed is::

        <solution>
        <test_code>

    Execution steps:

    1. Syntax-check the combined source with :func:`ast.parse`. Return
       immediately with a descriptive error if parsing fails.
    2. Arm a SIGALRM timeout (POSIX only; silently skipped on Windows).
    3. Compile and :func:`exec` the source inside an isolated namespace
       with stdout captured and discarded.
    4. Cancel the alarm unconditionally in the ``finally`` block.

    The isolated namespace is seeded with Python builtins so generated code
    can use built-in functions (``len``, ``range``, ``print``, etc.) normally.

    Args:
        solution:  Generated Python code to evaluate (function definitions,
                   helper functions, etc.).
        test_code: Test harness to run against the solution. Typically
                   contains a ``check()`` function and a call to invoke it.
        timeout:   Maximum wall-clock seconds allowed for execution.
                   Defaults to 10.0.

    Returns:
        ``(True, "")``
            All tests passed without error.
        ``(False, "SyntaxError: ...")``
            The combined source failed to parse.
        ``(False, "TimeoutError: exceeded <N>s")``
            Execution did not complete within *timeout* seconds.
        ``(False, "AssertionError[: ...]")``
            A test assertion failed.
        ``(False, "<ExcType>: ...\\n<traceback>")``
            Any other runtime exception, including up to three traceback frames.

    Notes:
        - stdout is captured and discarded; it does not appear in the terminal.
        - SIGALRM is unavailable on Windows. On those platforms no hard
          timeout is enforced. Use a subprocess-based approach if a hard
          timeout is required on Windows.
        - ``exec()`` runs in an isolated dict namespace, preventing pollution
          of the module-level scope, but this is **not** a full security
          sandbox. Do not use this to run untrusted code from unknown sources.
    """
    combined = solution + "\n" + test_code

    # Step 1: syntax check — fast, no timeout needed
    ok, fail_reason = _syntax_check(combined)
    if not ok:
        return False, fail_reason

    # Step 2: arm timeout
    _arm_timeout(timeout)

    # Step 3: compile and execute
    namespace: Dict[str, Any] = {"__builtins__": __builtins__}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(  # noqa: S102
                compile(combined, "<generated_code>", "exec"),
                namespace,
            )
        return True, ""

    except _TimeoutError:
        return False, f"TimeoutError: exceeded {timeout:.1f}s"

    except AssertionError as exc:
        msg = str(exc)
        return False, f"AssertionError: {msg}" if msg else "AssertionError"

    except Exception:
        return False, traceback.format_exc(limit=3).strip()

    # Step 4: cancel alarm — runs even when an exception was raised
    finally:
        _cancel_timeout()
