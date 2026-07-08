"""
benchmark_core/csv_logger.py

Reusable, schema-locked CSV logger for benchmark results.
"""

from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

CSV_SCHEMA_VERSION: int = 2

CSV_COLUMNS: List[str] = [
    "timestamp", "model", "model_family", "model_size", "quant",
    "quant_bits", "benchmark", "capability", "prompt_id", "prompt",
    "model_output", "expected_answer", "correct", "fail_reason",
    "score_overall", "prompt_tokens", "completion_tokens", "total_tokens",
    "runtime_sec", "tokens_per_sec", "gpu_memory_mb", "seed", "n_ctx",
    "temperature", "top_p", "repeat_penalty",
]

_COLUMN_TYPES: Dict[str, tuple] = {
    "correct": (bool, int), "prompt_tokens": (int,), "completion_tokens": (int,),
    "total_tokens": (int,), "runtime_sec": (float, int), "tokens_per_sec": (float, int),
    "score_overall": (float, int), "quant_bits": (float, int), "seed": (int,),
    "n_ctx": (int,), "temperature": (float, int), "top_p": (float, int),
    "repeat_penalty": (float, int),
}


class CSVLoggerError(Exception):
    """Raised when CSVLogger fails to write, read, or validate a row."""


class CSVLogger:
    """Append-only, incrementally-flushed CSV writer with a fixed schema."""

    def __init__(self, path: Union[str, Path], overwrite: bool = False) -> None:
        self.path = Path(path)
        self.overwrite = overwrite
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not overwrite and self.path.exists() and self.path.stat().st_size > 0:
            self._validate_existing_header()
        else:
            self._write_header()

    def _validate_existing_header(self) -> None:
        try:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                try:
                    existing_header = next(reader)
                except StopIteration:
                    existing_header = []
        except OSError as exc:
            raise CSVLoggerError(f"Failed to read existing file '{self.path}': {exc}") from exc

        if existing_header != CSV_COLUMNS:
            raise CSVLoggerError(
                f"Existing file '{self.path}' has a header that does not match the "
                f"canonical schema (CSV_SCHEMA_VERSION={CSV_SCHEMA_VERSION}).\n"
                f"Expected: {CSV_COLUMNS}\nFound:    {existing_header}\n"
                f"Pass overwrite=True to start this file fresh instead."
            )

    def _write_header(self) -> None:
        try:
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                f.flush()
                os.fsync(f.fileno())
        except OSError as exc:
            raise CSVLoggerError(f"Failed to write header to '{self.path}': {exc}") from exc

    def log_row(self, row: Dict[str, Any]) -> None:
        row = dict(row)
        if "prompt_id" in row:
            row["prompt_id"] = str(row["prompt_id"])
        self._validate_row(row)
        with self._lock:
            try:
                with self.path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                    writer.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                raise CSVLoggerError(f"Failed to write row to '{self.path}': {exc}") from exc

    def log_rows(self, rows: Iterable[Dict[str, Any]]) -> None:
        validated: List[Dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            if "prompt_id" in row:
                row["prompt_id"] = str(row["prompt_id"])
            self._validate_row(row)
            validated.append(row)

        with self._lock:
            try:
                with self.path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                    for row in validated:
                        writer.writerow(row)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                raise CSVLoggerError(f"Failed to write rows to '{self.path}': {exc}") from exc

    def _validate_row(self, row: Dict[str, Any]) -> None:
        row_keys = set(row.keys())
        expected_keys = set(CSV_COLUMNS)
        missing = expected_keys - row_keys
        extra = row_keys - expected_keys
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing columns: {sorted(missing)}")
            if extra:
                problems.append(f"unexpected columns: {sorted(extra)}")
            raise CSVLoggerError(
                f"Row schema mismatch ({'; '.join(problems)}). Expected exactly: {CSV_COLUMNS}"
            )
        for column, expected_types in _COLUMN_TYPES.items():
            value = row.get(column, "")
            if value == "" or value is None:
                continue
            if not isinstance(value, expected_types):
                raise CSVLoggerError(
                    f"Column '{column}' expected type(s) {[t.__name__ for t in expected_types]}, "
                    f"got {type(value).__name__} ({value!r})"
                )

    @staticmethod
    def make_row(**kwargs: Any) -> Dict[str, Any]:
        unexpected = set(kwargs.keys()) - set(CSV_COLUMNS)
        if unexpected:
            raise CSVLoggerError(
                f"Unexpected column(s) in make_row(): {sorted(unexpected)}. Valid columns: {CSV_COLUMNS}"
            )
        if "timestamp" not in kwargs:
            kwargs["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if "prompt_id" in kwargs:
            kwargs["prompt_id"] = str(kwargs["prompt_id"])
        return {col: kwargs.get(col, "") for col in CSV_COLUMNS}

    def read_rows(self) -> List[Dict[str, str]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except OSError as exc:
            raise CSVLoggerError(f"Failed to read rows from '{self.path}': {exc}") from exc

    def read_logged_prompt_ids(self, benchmark: Optional[str] = None) -> Set[str]:
        if not self.path.exists():
            return set()
        seen: Set[str] = set()
        try:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if benchmark is not None and row.get("benchmark") != benchmark:
                        continue
                    seen.add(row.get("prompt_id", ""))
        except OSError as exc:
            raise CSVLoggerError(f"Failed to read prompt IDs from '{self.path}': {exc}") from exc
        return seen

    @property
    def row_count(self) -> int:
        """Return the number of data rows logged (excluding the header)."""
        if not self.path.exists():
            return 0
        try:
            with self.path.open("r", newline="", encoding="utf-8") as f:
                # Subtract 1 for the header row
                return max(0, sum(1 for _ in f) - 1)
        except OSError as exc:
            raise CSVLoggerError(f"Failed to count rows in '{self.path}': {exc}") from exc
