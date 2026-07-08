"""
benchmark_core/metadata.py

Extract standardised metadata from GGUF model filenames.

Supports the naming conventions used by the models in this project:
    Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
    Meta-Llama-3-8B-Instruct.Q8_0.gguf
    Meta-Llama-3-8B-Instruct.F16.gguf
    Qwen2.5-7B-Instruct.Q4_K_M.gguf
    Qwen2.5-14B-Instruct.IQ2_XS.gguf

As well as sharded GGUF filenames, which use hyphens instead of a dot
before the quant tag and carry a trailing shard suffix:
    qwen2.5-14b-instruct-fp16-00001-of-00008.gguf
    qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Maps quantisation name (upper-case) -> approximate bit-width
_QUANT_BITS: dict[str, float] = {
    "F16":     16.0,
    "FP16":    16.0,
    "Q8_0":     8.0,
    "Q6_K":     6.0,
    "Q5_K_M":   5.0,
    "Q5_K_S":   5.0,
    "Q5_0":     5.0,
    "Q4_K_M":   4.0,
    "Q4_K_S":   4.0,
    "Q4_0":     4.0,
    "Q3_K_M":   3.0,
    "Q3_K_S":   3.0,
    "Q3_K_L":   3.0,
    "Q2_K":     2.0,
    "IQ2_XS":   2.0,
    "IQ2_XXS":  2.0,
    "IQ1_S":    1.0,
}

# Maps a substring found in the stem (lower-case) -> canonical family name
_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("llama",  "Llama"),
    ("qwen",   "Qwen"),
    ("mistral","Mistral"),
    ("gemma",  "Gemma"),
    ("phi",    "Phi"),
]

# Regex that matches a model-size token like 7B, 8B, 14B, 70B, 0.5B, 1.5B
_SIZE_RE = re.compile(r"\b(\d+(?:\.\d+)?B)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_model_metadata(model_path: str) -> dict[str, object]:
    """
    Parse standardised metadata from a GGUF model filename.

    The filename is expected to follow the pattern::

        <ModelName>.<QuantisationTag>.gguf

    where <ModelName> contains the model family and parameter size, and
    <QuantisationTag> is one of the known quantisation identifiers (e.g.
    Q4_K_M, Q8_0, F16, IQ2_XS).

    Args:
        model_path: Path to the GGUF file (absolute, relative, or filename only).

    Returns:
        A dict with the following keys:

        ``model_name``   (str)   Filename stem without the quant suffix,
                                 e.g. "Meta-Llama-3-8B-Instruct".
        ``model_family`` (str)   Detected family, e.g. "Llama" or "Qwen".
                                 "Unknown" if the family cannot be determined.
        ``model_size``   (str)   Parameter count string, e.g. "8B" or "14B".
                                 "Unknown" if not found in the filename.
        ``quant``        (str)   Quantisation tag as it appears in the filename,
                                 e.g. "Q4_K_M".
        ``quant_bits``   (float) Approximate bits per weight, e.g. 4.0.

    Raises:
        ValueError: If the filename does not end in .gguf, the quant tag
                    cannot be identified, or the quant tag is not in the
                    known lookup table.

    Examples:
        >>> parse_model_metadata("Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
        {
            'model_name':   'Meta-Llama-3-8B-Instruct',
            'model_family': 'Llama',
            'model_size':   '8B',
            'quant':        'Q4_K_M',
            'quant_bits':   4.0,
        }
    """
    path = Path(model_path)

    # --- Validate extension ---
    if path.suffix.lower() != ".gguf":
        raise ValueError(
            f"Expected a .gguf file, got: '{model_path}'"
        )

    # Full stem, e.g.:
    #   "Meta-Llama-3-8B-Instruct.Q4_K_M"                      (dot format)
    #   "qwen2.5-14b-instruct-fp16-00001-of-00008"             (sharded)
    full_stem = path.stem  # strips .gguf

    # Locate a known quantisation tag anywhere in the stem, bounded by
    # '.', '-', '_', or the start/end of the string. Checking against the
    # known tag list (rather than splitting on a fixed separator) lets one
    # code path handle both dot-separated and hyphen-separated/sharded
    # filenames, since the quant tag is the one part of the filename we
    # can identify with certainty. Tags are tried longest-first so e.g.
    # "Q4_K_M" is matched before a hypothetical shorter prefix of it.
    match = None
    for tag in sorted(_QUANT_BITS, key=len, reverse=True):
        m = re.search(rf"(?<![A-Za-z0-9])({re.escape(tag)})(?![A-Za-z0-9])",
                       full_stem, re.IGNORECASE)
        if m:
            match = m
            break

    if match is None:
        raise ValueError(
            f"Cannot identify a known quantisation tag in filename: '{path.name}'. "
            f"Supported tags: {sorted(_QUANT_BITS)}"
        )

    # Model name is everything before the quant tag, with the trailing
    # separator stripped. Anything after the tag (e.g. a shard suffix
    # like "-00001-of-00008") is simply not part of the model name.
    model_name = full_stem[:match.start()].rstrip(".-_")
    quant_raw = match.group(1)

    if not model_name:
        raise ValueError(
            f"Cannot determine model name in filename: '{path.name}'"
        )

    # --- Validate and normalise quant tag ---
    quant_upper = quant_raw.upper()
    if quant_upper not in _QUANT_BITS:
        raise ValueError(
            f"Unrecognised quantisation tag '{quant_raw}' in '{path.name}'. "
            f"Supported tags: {sorted(_QUANT_BITS)}"
        )

    quant_bits: float = _QUANT_BITS[quant_upper]
    # Preserve the original casing from the filename for the quant field
    quant: str = quant_raw

    # --- Detect model family ---
    stem_lower = model_name.lower()
    model_family = "Unknown"
    for pattern, canonical in _FAMILY_PATTERNS:
        if pattern in stem_lower:
            model_family = canonical
            break

    # --- Detect model size ---
    size_match = _SIZE_RE.search(model_name)
    model_size = size_match.group(1).upper() if size_match else "Unknown"

    return {
        "model_name":   model_name,
        "model_family": model_family,
        "model_size":   model_size,
        "quant":        quant,
        "quant_bits":   quant_bits,
    }
