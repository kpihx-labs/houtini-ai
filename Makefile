# AI-Detectors / houtini-ai — Makefile
# Standard targets for AI detection tool with hardware profiles
# Usage: make install-xpu / make install-cpu / make check / make test

SHELL := /bin/zsh
UV := uv
PKG := houtini-ai
PKG_DIR := $(CURDIR)

.PHONY: help install-cpu install-xpu install-cuda install-light sync test check clean

help:
	@echo "Targets:"
	@echo "  install-cpu     Install with CPU torch backend"
	@echo "  install-xpu     Install with Intel Arc XPU torch backend (default for this machine)"
	@echo "  install-cuda    Install with NVIDIA CUDA torch backend"
	@echo "  install-light   Install with ONNX light model only (no torch)"
	@echo "  sync            Sync dependencies (respects UV_EXTRA env or --extra)"
	@echo "  test            Run tests"
	@echo "  check           Lint + type-check"
	@echo "  clean           Remove .venv and build artifacts"

# Default: XPU on this machine
UV_EXTRA ?= xpu

install-cpu: UV_EXTRA = cpu
install-cpu: sync

install-xpu: UV_EXTRA = xpu
install-xpu: sync

install-cuda: UV_EXTRA = cuda
install-cuda: sync

install-light: UV_EXTRA = light
install-light: sync

sync:
	@echo ">>> Syncing $(PKG) with extra: $(UV_EXTRA)"
	$(UV) sync --extra $(UV_EXTRA)

test:
	@echo ">>> Running tests"
	$(UV) run --extra $(UV_EXTRA) --with pytest python -m pytest tests -v

check: test
	$(UV) run --extra $(UV_EXTRA) python -m compileall -q src
	ruff check .
	pyright --pythonpath .venv/bin/python .

clean:
	rm -rf .venv dist build *.egg-info
	$(UV) cache prune

# Install as editable tool with specific extra
tool-install-cpu:
	$(UV) tool install --editable ".[cpu]"

tool-install-xpu:
	$(UV) tool install --editable ".[xpu]"

tool-install-cuda:
	$(UV) tool install --editable ".[cuda]"

tool-install-light:
	$(UV) tool install --editable ".[light]"

# Verification that torch is on the right backend
verify-xpu:
	$(UV) run --extra xpu python -c "import torch; print('torch:', torch.__version__, '| xpu:', torch.xpu.is_available())"

verify-cpu:
	$(UV) run --extra cpu python -c "import torch; print('torch:', torch.__version__, '| cpu ok')"

.PHONY: push puhsb uv-build uv-publish uv-link

push:
	@set -e; branch=$$(git branch --show-current); for remote in github gitlab; do git push "$$remote" "$$branch"; done

puhsb: push

uv-build:
	@if [ -d dist ]; then rm dist; fi
	uv build --no-sources

uv-publish: uv-build
	uv publish dist/*

uv-link:
	uv tool install --editable ".[$(UV_EXTRA)]" --force
