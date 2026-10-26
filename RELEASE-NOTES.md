# KpihX-Labs 2.0.0

- Add declarative language/model registry: en, en-light, fr and default en+fr;
  fr+en is an alias of the bilingual XLM-R checkpoint, not an ensemble.
- Separate model lifecycle, inference adapters and report orchestration. Complete
  local caches make no network request; imports/help/listing never pull weights.
- Add Typer/Rich scan, compare, models list/info and recursive --help-all with
  parameter help and concrete examples. Retain direct file/options invocations.
- Report distinct mean scores, AI-label fractions, abstentions and phase timings.
  Window long sentences rather than silently dropping their tail.
- Add French-scoped style diagnostics without changing neural probabilities.
- Add offline lifecycle/CLI/overflow/metric tests and a reproducible two-book
  benchmark script retaining every JSON, stderr, input hash and wall time.

## KpihX-Labs 1.1.2

- Add the standard `houtini-ai --version` CLI introspection flag.

## 1.1.1

- Remove residual MCP branding and the obsolete upstream requirements manifest.
- Align all public runtime documentation with CUDA, XPU and CPU support.

## 1.1.0

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
