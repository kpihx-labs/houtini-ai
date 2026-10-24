"""Device routing tests without requiring installed accelerator hardware."""

import sys
from types import SimpleNamespace

from ai_detect.detector import resolve_torch_device


def _torch(xpu_available=False, cuda_available=False):
    return SimpleNamespace(
        device=lambda name: name,
        xpu=SimpleNamespace(is_available=lambda: xpu_available),
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
    )


def test_auto_prefers_cuda_over_xpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(xpu_available=True, cuda_available=True))
    assert resolve_torch_device("auto") == "cuda"


def test_explicit_xpu_falls_back_to_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch())
    assert resolve_torch_device("xpu") == "cpu"


def test_auto_uses_cuda_without_xpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(cuda_available=True))
    assert resolve_torch_device("auto") == "cuda"


def test_auto_uses_xpu_without_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _torch(xpu_available=True))
    assert resolve_torch_device("auto") == "xpu"
