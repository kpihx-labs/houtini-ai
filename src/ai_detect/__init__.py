"""ai-detect — per-sentence AI text detection with pattern diagnostics.

Public API:
    from ai_detect import classify_text, get_detector
"""

from .detector import (
    DEFAULT_MODEL,
    MODELS,
    classify_text,
    get_detector,
)
from .patterns import diagnose_sentence

__all__ = [
    "classify_text",
    "get_detector",
    "diagnose_sentence",
    "MODELS",
    "DEFAULT_MODEL",
]

from importlib.metadata import version

__version__ = version("houtini-ai")
