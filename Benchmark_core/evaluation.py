"""
benchmark_core/execution.py

Reusable sandboxed Python execution utilities for coding benchmarks.

Used by HumanEval, MBPP, and any other benchmark that needs to run
generated code against a test harness.

No benchmark-specific logic, no dataset loading, no model code.
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
    raise _TimeoutError()


def _syntax_check(code: str) -> tuple[bool, str]:
    """
    Parse code with ast.parse().

    Returns:
        (True, "")                      on success
        (False, "SyntaxError: <msg>")   on failure
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"


def _arm_timeout(timeout: float) -> None:
    """
    Install the SIGALRM handler and schedule an alarm.
    On platforms where SIGALRM is unavailable (e.g. Windows),
    this function silently does nothing.
    """
    try:
        signal.signal(signal.SIGALRM, _sigalrm_handler)
        signal.alarm(max(1, math.ceil(timeout)))
    except (AttributeError, OSError):
        pass


def _cancel_timeout() -> None:
    """Cancel any pending SIGALRM alarm, ignoring errors."""
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
    Execute solution + test_code in an isolated namespace and return the result.

    The full code executed is::

        <solution>
        <test_code>

    Steps:
        1. Syntax-check the combined source with ast.parse().
        2. Arm a SIGALRM timeout (POSIX only; skipped silently on Windows).
        3. exec() inside an isolated namespace with stdout captured.
        4. Cancel the alarm.

    Args:
        solution:  Generated code to evaluate (function definition, etc.).
        test_code: Test harness to run against the solution (check() call, etc.).
        timeout:   Maximum wall-clock seconds allowed for execution.

    Returns:
        (True,  "")                         — all tests passed.
        (False, "SyntaxError: ...")         — code did not parse.
        (False, "TimeoutError: ...")        — execution exceeded timeout.
        (False, "AssertionError: ...")      — a test assertion failed.
        (False, "<ExcType>: ...\n<tb>")     — any other runtime exception.

    Notes:
        - stdout is captured and discarded; it does not appear in the terminal.
        - SIGALRM is unavailable on Windows. On those platforms the function
          still runs but enforces no hard timeout. Use a subprocess-based
          approach if a hard timeout is required on Windows.
        - exec() runs in an isolated dict namespace, preventing pollution of
          the module-level scope, but it is NOT a full security sandbox.
          Do not use this to execute untrusted code from unknown sources.
    """
    combined = solution + "\n" + test_code

    # --- Step 1: syntax check (fast, no timeout needed) ---
    ok, fail_reason = _syntax_check(combined)
    if not ok:
        return False, fail_reason

    # --- Step 2: arm timeout ---
    _arm_timeout(timeout)

    # --- Step 3: execute ---
    namespace: Dict[str, Any] = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(combined, "<generated_code>", "exec"), namespace)  # noqa: S102
        return True, ""

    except _TimeoutError:
        return False, f"TimeoutError: exceeded {timeout:.1f}s"

    except AssertionError as exc:
        msg = str(exc)
        return False, f"AssertionError: {msg}" if msg else "AssertionError"

    except Exception:
        tb = traceback.format_exc(limit=3).strip()
        return False, tb

    # --- Step 4: cancel alarm ---
    finally:
        _cancel_timeout()
