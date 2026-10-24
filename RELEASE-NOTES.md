# KpihX-Labs 1.1.0

- Add explicit `--device xpu` support for DeBERTa inference.
- Set `auto` device selection to CUDA, then Intel XPU, then CPU.
- Verify on the HW1 10-page PDF: XPU produces the same labels as CPU with a
  maximum probability delta of 0.0001 and is 2.07x faster end-to-end.

## 1.0.3

- Include the unchanged original MIT notice and model-license notices in both archives.
- Add byte-for-byte distribution-notice tests and exclude editor backups.
- Update documentation and runtime hints to UV and the current CLI-only interface.

## 1.0.2

- Preserve AutoConfig checkpoint dispatch at runtime while correcting its static typing.

## 1.0.1

- Preserve upstream MIT attribution and full Git history.
- Publish and expose `houtini-ai` with the `src/ai_detect` package and uv_build.
- Keep CPU, XPU and CUDA profiles separate with explicit PyTorch indexes.
- Remove the MCP server and its direct dependencies from the CLI distribution.
- Fix optional ONNX imports and type validation without downloading model weights.
- Retain network-free tests, Ruff and Pyright in `make check`.
- Add two-remote push targets and wheel/sdist publication targets.
