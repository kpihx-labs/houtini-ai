<div align="center">
  <img src="https://raw.githubusercontent.com/houtini-ai/ai-detect/main/assets/logo.png" width="120" height="120" alt="ai-detect" />
</div>

# houtini-ai - local AI text detection

## KpihX-Labs CLI distribution

This MIT-licensed adaptation retains Richard Baxter / Houtini's implementation and
history: https://github.com/houtini-ai/ai-detect.
The PyPI distribution and executable are `houtini-ai`; the Python package is `ai_detect`.
The build backend is `uv_build`, the source layout is `src/ai_detect`, and no MCP
server is shipped by this adaptation.

From a checkout, use `make uv-link` (XPU wheels by default) or
`make uv-link UV_EXTRA=cpu`. Run `houtini-ai document.txt --json`.
The default CLI device policy is `auto`: CUDA, then Intel XPU, then CPU.
For a generic CPU installation from PyPI, use
`uv tool install 'houtini-ai[cpu]' --index https://download.pytorch.org/whl/cpu`.
For Intel wheels from PyPI, use
`uv tool install 'houtini-ai[xpu]' --index https://download.pytorch.org/whl/xpu`.
Explicit wheel selection is necessary because uv's source-index configuration is
not embedded in wheel dependency metadata. With the XPU wheel installed,
`houtini-ai` auto-selects XPU when CUDA is absent; use `--device cpu`,
`--device xpu` or `--device cuda` to override it.

`make check` runs network-free tests, compilation, Ruff and Pyright.
`make uv-build` builds a wheel and sdist, `make uv-publish` publishes them, and
`make push` (also `make puhsb`) pushes the current branch to `github` and `gitlab`.
Model files are downloaded from Hugging Face on first inference; they are not
bundled in the PyPI wheel. Results go to stdout and load messages to stderr.
Scores and pattern diagnostics are not proof of authorship or plagiarism.

The detector description below retains upstream context; operational examples
use UV and the CLI-only `houtini-ai` command.

[![MCP](https://img.shields.io/badge/Model_Context_Protocol-server-8b5cf6?style=flat-square)](https://modelcontextprotocol.io)
[![Model](https://img.shields.io/badge/model-DeBERTa--v3-5b5fff?style=flat-square)](https://huggingface.co/desklib/ai-text-detector-v1.01)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Known Vulnerabilities](https://snyk.io/test/github/houtini-ai/ai-detect/badge.svg)](https://snyk.io/test/github/houtini-ai/ai-detect)

**Is the copy you're buying handwritten? Find out, free and at scale.**

houtini-ai is a free, open source AI text detector that runs on your own hardware. It scores writing sentence by sentence using [desklib/ai-text-detector-v1.01](https://huggingface.co/desklib/ai-text-detector-v1.01). There is no API key, per-word pricing or text upload: after the initial model download it can work offline. This adaptation ships a CLI and an importable Python package under the original MIT licence, not an MCP server.

That combination is the point. Every mainstream AI checker - GPTZero, Originality.ai, Copyleaks, Winston - is a hosted service you paste your client's unpublished draft into. This one never sends a byte anywhere.

It isn't Pangram Labs, and no detector is a lie detector - a confident writer who contracts their verbs will sail through. But where there's smoke there's fire, and across a batch of copy the signal is real.

**Quick answers**

| Question | Answer |
|---|---|
| Does it cost anything? | No. MIT licensed, no API key, no usage limits. |
| Does my text leave the machine? | No. Inference is local; the only network call is the first model download. |
| Which model? | DeBERTa-v3-large (304M), RAID benchmark #1. A 126 MB RoBERTa ONNX alternative ships too. |
| Can I integrate it into another service? | Invoke the CLI with `--json` or import its Python package; no API daemon is installed. |
| Does it need a GPU? | No. Auto selects CUDA, then Intel XPU, then CPU. |
| How accurate is it? | On paired tests, formal phrasing scored 92.6% AI against 0.03% for the same fact written conversationally. Treat it as a signal across a batch, not a verdict on one sentence. |

## What it tells you

Better than a single percentage, it tells you *why* a line reads as machine-written. The patterns are mapped from 70+ paired sentence tests, formal version against a conversational rewrite of the same information:

- **Missing contractions are the #1 signal.** "The software has been significantly improved" scores 92.6% AI; "software's got a lot better" scores 0.03%.
- **"Provides / offers / delivers" are detector magnets.** So are the **"Furthermore" / "Additionally" / "In conclusion"** openers.
- **"For [group] who [condition]"** framing, vague abstractions where a specific name would do, and, oddly, the word **"noticeably"**.

Each flag comes with a plain-English rewrite suggestion, so the tool doubles as an editing pass. The full pattern write-up is in [`docs/detection-patterns.md`](docs/detection-patterns.md).

```
  Model AI score: 87.1% - LIKELY AI   [desklib]
  Sentences: 3 AI / 0 Human / 3 scored
  SDSL: mean=7.3 words, stddev=1.9, CV=0.26 (very uniform (AI-like))
  Patterns: 1 flaggy adverb, 1 formal verb, 1 formal transition

  >> AI  (0.92) The software has been significantly improved.
              ^ flaggy_adverb: 'significantly' — try 'a lot' or 'massively' or cut it
  >> AI  (0.73) Additionally, the QR2 provides excellent stability.
              ^ formal_verb: 'provides' — try 'gives you' or 'has'
              ^ formal_transition: 'Additionally' — cut it, or use 'And' / 'Plus'
```

## Two detectors

| | `desklib` (default) | `light` (`--model light`) |
|---|---|---|
| Model | DeBERTa-v3-large, 304M | RoBERTa-base int8, ONNX |
| Download | **~1.7 GB**, once | **~126 MB**, once |
| Runtime | PyTorch (CPU or CUDA) | ONNX Runtime (CPU, or GPU via DirectML/CUDA) |
| Best for | the calibrated reference score | a fast, small triage pass |

`desklib` is the one the pattern research was built on and the one I trust for an absolute score. `light` is smaller and quicker to install and it runs hotter — it over-flags a bit — so treat it as a fast first look rather than the final word. Same CLI, same MCP tools, just pass `--model light`.

## Why a local detector rather than a hosted one

The hosted AI checkers - GPTZero, Originality.ai, Copyleaks, Winston AI - all work the same way: you paste the text into their site, their server scores it, you pay per word or per month. For a lot of jobs that's fine. For three, it isn't.

**Client confidentiality.** If you're vetting commissioned copy, that draft is unpublished and often under NDA. Pasting it into a third-party scoring service is a disclosure, whatever the privacy policy says. Running the model locally makes the question moot.

**Volume.** Checking sixty articles from an agency costs nothing here beyond electricity. Per-word pricing turns the same batch into a purchase order.

**Reproducibility.** A hosted model can be retrained on a Tuesday and score your archive differently on Wednesday, with no changelog. A pinned local checkpoint gives you the same number in six months, which matters if the score is going in a report.

The trade is real: you give up a polished dashboard, team accounts and a support contract, and you spend 1.7 GB of disk. As of August 2026 this is beta software and the honest positioning is a signal across a batch, not a verdict you'd take to arbitration.

## Install

```bash
git clone git@github.com:kpihx-labs/houtini-ai.git
cd houtini-ai
make uv-link                          # Intel XPU wheel profile
make uv-link UV_EXTRA=cpu             # alternative CPU profile
uv tool install --editable '.[xpu,light]' --force  # optional ONNX backend
```

This exposes one command, `houtini-ai`. Project-local execution is also possible
with `uv run --extra xpu python -m ai_detect.cli ...`.

### Looking for detect.py?

`detect.py` remains a small compatibility shim. Invoke it through
`uv run --extra xpu python detect.py ...`. The implementation lives in `src/ai_detect/`:

| File | What's in it |
|------|--------------|
| `src/ai_detect/detector.py` | Model loading and scoring - both backends live here |
| `src/ai_detect/patterns.py` | Pattern diagnostics (formal verbs, missing contractions, SDSL) |
| `src/ai_detect/cli.py` | The command-line interface |
| `detect.py` | Backwards-compatible shim - imports and calls `ai_detect.cli` |

Start at `src/ai_detect/detector.py` to inspect scoring. The removed MCP implementation
is retained in the upstream Git history, not installed by this distribution.

Check your setup any time:

```bash
uv run --extra xpu python scripts/check_env.py
```

## Run it

An accelerator helps but is not required. The default order is CUDA, then Intel XPU, then CPU.

```bash
houtini-ai --file draft.txt              # score a file
houtini-ai --text "your text here"       # score a string
houtini-ai --compare a.txt b.txt         # compare two versions
houtini-ai --json --file draft.txt       # machine-readable output
houtini-ai --model light --file draft.txt   # small/fast model
houtini-ai --device cpu --file draft.txt    # force CPU (auto | xpu | cpu | cuda)
```

Try it on the bundled examples — one obviously machine-written, one not:

```bash
houtini-ai --compare examples/ai-sample.txt examples/human-sample.txt
```

Only the copy-heavy sentences get scored: anything under five words (headings, fragments) is skipped rather than guessed at.

## Integrate through the CLI

Use `houtini-ai --file draft.txt --json` from a shell or subprocess.
Structured results go to stdout; diagnostic messages go to stderr.
No daemon, MCP client configuration or background service is required.

## The model & the download

First run of the default model pulls **~1.7 GB** from the Hugging Face Hub and caches it under `~/.cache/huggingface`. After that it's instant and offline. Nothing about your text is sent anywhere — the model runs on your hardware.

The first CLI inference waits for the required model files. Later invocations
reuse the Hugging Face cache but still pay process-start and model-load costs;
they are not guaranteed to be instantaneous. `--model light` uses the smaller
ONNX checkpoint when its optional dependencies are installed.

Why local and not a hosted API? desklib registers a custom model architecture, so it isn't served by Hugging Face's hosted inference — and even where a hosted detector exists, using it would mean shipping your unpublished copy to someone else's server. For a tool you point at commissioned work, local is the right default. If 1.7 GB is too much, `--model light` is a ~126 MB stand-in.

## Status

Beta, and honest about it. It's a CLI I run on my own commissioned copy: the model does the classification, the pattern diagnostics and rewrite hints are mine. Expect rough edges. If it's useful to you, that's a bonus.

Built by [Houtini](https://houtini.com).
