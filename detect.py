#!/usr/bin/env python3
"""
Backwards-compatible entry point.

The implementation now lives in the `ai_detect` package. This shim keeps the
old shim usable through `uv run --extra xpu python detect.py ...`.
Prefer the installed console script `houtini-ai` after `make uv-link`.
"""

import sys

from ai_detect.cli import main

if __name__ == "__main__":
    sys.exit(main())
