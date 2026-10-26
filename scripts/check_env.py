#!/usr/bin/env python3
"""
Environment doctor for ai-detect.

Reports Python, PyTorch accelerator, Transformers and ONNX Runtime availability
so install issues are easy to diagnose.

    python scripts/check_env.py
"""

import importlib
import sys


def line(label, value):
    print(f"  {label:22} {value}")


def main():
    print("houtini-ai environment check")
    print("-" * 40)
    line("Python", sys.version.split()[0])
    line("Executable", sys.executable)

    try:
        import torch
        line("torch", torch.__version__)
        line("CUDA available", torch.cuda.is_available())
        line("XPU available", torch.xpu.is_available())
        if torch.cuda.is_available():
            line("GPU", torch.cuda.get_device_name(0))
    except ImportError:
        line("torch", "NOT INSTALLED (make install-xpu or make install-cpu)")

    try:
        import transformers
        line("transformers", transformers.__version__)
    except ImportError:
        line("transformers", "NOT INSTALLED (uv sync)")

    try:
        ort = importlib.import_module("onnxruntime")
        line("onnxruntime", f"{ort.__version__}  providers={ort.get_available_providers()}")
    except ImportError:
        line("onnxruntime", "not installed (only needed for --model en-light)")

    line("Interface", "CLI only")

    print("-" * 40)
    from ai_detect.models import DEFAULT_MODEL, catalogue
    print(f"Default selection: {DEFAULT_MODEL}. Only requested missing models download.")
    for entry in catalogue():
        line(entry["key"], f"{entry['repo']} ({','.join(entry['languages'])}; cached={entry['cached']})")


if __name__ == "__main__":
    main()
