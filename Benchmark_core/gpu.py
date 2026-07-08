"""
benchmark_core/gpu.py

Lightweight GPU memory monitoring utilities for benchmark logging.

Safe to call repeatedly. Never raises. No side effects beyond lazy NVML
initialisation on the first successful call.
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# Lazy NVML initialisation
# ---------------------------------------------------------------------------

# None  = not yet attempted
# False = attempted and failed (pynvml unavailable or no GPU)
# True  = initialised successfully
_nvml_ready: Optional[bool] = None


def _ensure_nvml() -> bool:
    """
    Initialise pynvml on first call and cache the result.

    Returns True if NVML is available and initialised, False otherwise.
    Never raises.
    """
    global _nvml_ready

    if _nvml_ready is not None:
        return _nvml_ready

    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_ready = True
    except Exception:
        _nvml_ready = False

    return _nvml_ready


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_gpu_memory_mb() -> Optional[float]:
    """
    Return the currently used GPU memory on device 0, in megabytes.

    Uses pynvml (part of the nvidia-ml-py package) to query the first
    available NVIDIA GPU. NVML is initialised lazily on the first call
    and the result is cached, so repeated calls incur no re-initialisation
    overhead.

    Returns:
        Used GPU memory in MB as a float, or None if:
            - pynvml is not installed,
            - no NVIDIA GPU is present,
            - NVML initialisation fails for any reason,
            - the memory query itself fails.

    Never raises an exception.
    """
    if not _ensure_nvml():
        return None

    try:
        import pynvml
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return None
