# Two-book multilingual benchmark, 2026-10-26

## Method and artifacts

All six runs used the installed CLI in a fresh process, complete local model
caches and the same two unmodified UTF-8 texts. FR and bilingual inference used
Intel XPU; ONNX used CPUExecutionProvider with four intra-op threads. No text
was uploaded. The script is `scripts/benchmark_books.py`.

Local artifacts: `$HOME/RAW/detectors/houtini-multilingual-20261026T190237Z/`
contains full per-sentence JSON, each stderr transcript and `summary.json`.
The corpora are NOT distributed with the repository or package.

| Corpus | Input SHA256 | Scored / skipped |
|---|---|---|
| C'etait ca ou mourir (Thelyson Orelien) | `a07e265fd546c1e975def25896b5adef8dc94c58f0d29a3be4feee9f617fdb53` | 4,309 / 1,505 |
| Mais qui a attrape le bison de Higgs ? (David Louapre) | `ebe6d8c6539e8860e13c121f4699f17f674f2e10aa067aa51526edf2d0681474` | 2,175 / 228 |

Inputs came from EPUB-to-PDF conversion and native pdftotext; front matter,
page debris and line wrapping remain. Bison dates from 2016, before the current
mass-market generative-AI era. Operator-reported allegations about the novel
are not verified labels. Neither is an experimental AI-positive control.

## Results and timings

The mean score is NOT the percentage of AI-labelled sentences, nor a probability
that the author used AI. Fragment abstentions are excluded from both denominators.

| Model | Corpus | Mean score | AI / scored | Load s | Scoring s | CLI wall s |
|---|---|---:|---:|---:|---:|---:|
| fr | C'etait | 0.1% | 2 / 4,309 | 7.262 | 48.702 | 59.692 |
| fr | Bison | 2.0% | 31 / 2,175 | 5.269 | 32.289 | 39.196 |
| fr+en | C'etait | 0.2% | 10 / 4,309 | 8.419 | 31.294 | 42.341 |
| fr+en | Bison | 2.3% | 48 / 2,175 | 8.784 | 19.537 | 30.499 |
| en-light | C'etait | 60.9% | 3,001 / 4,309 | 4.717 | 88.569 | 95.760 |
| en-light | Bison | 72.9% | 1,804 / 2,175 | 4.559 | 58.434 | 64.774 |

Scoring seconds include tokenization, batching, transfers and synchronized neural
execution, not isolated accelerator kernel time. Load seconds include first
library imports and checkpoint loading. CLI wall includes process startup and
all report work. No download time is included in these warm-disk-cache runs.
The first explicit cache fills took 73.40 s (FR), 52.60 s (bilingual) and 8.91 s
(ONNX), separately from the benchmark.

| Selection | Upstream repository | Resolved revision |
|---|---|---|
| fr | almanach/camemberta-chatgptdetect-noisy | `0978fcac434142dd903be6920bbf57f8634438b9` |
| en+fr (= fr+en) | almanach/xlmr-chatgptdetect-noisy | `e7bc8107ddd1486b43cc9fe76ca647e70fb365f0` |
| en-light | onnx-community/tmr-ai-text-detector-ONNX | `b9aa251e5bcda7e429fcc936767d921435945b60` |

## Investigation, not threshold tuning

`scripts/verify_backends.py` compared adapter outputs against a direct checkpoint
forward pass using independent tokenization/feed construction. It read class
names from checkpoint metadata: HUMAN/human is class 0; CHATGPT/ai is class 1.
Maximum probability differences on four plumbing samples were 3.61e-9 (FR,
XPU vs CPU), 1.55e-7 (bilingual, XPU vs CPU) and 0 (ONNX). Results are archived
in `$HOME/RAW/detectors/houtini-backend-verification.json`.

No class inversion, accelerator drift or adapter discrepancy was found in that
check. All full-book runs had exactly one token window per scored sentence, so
overflow handling did not change these results. No threshold was adjusted to
fit either book or the allegations.

The English ONNX detector heavily flags a pre-LLM-era French book. The FR and
bilingual checkpoints give low scores to both books and also to some explicitly
synthetic plumbing samples. Agreement with direct inference does not establish
accuracy: these are old ChatGPT/HC3 classifiers, used sentence by sentence on
literary/scientific prose outside their training task. Their model cards explicitly
warn about out-of-domain and adversarial failure. Short synthetic samples test
wiring, not recall on an AI benchmark. A proper accuracy study needs independently
labelled native French controls and multiple modern generators.

The FR runtime emits an upstream `torch.jit.script` deprecation FutureWarning
during transformer import. It is preserved in stderr, not suppressed. All runs
completed successfully; no missing/mismatched checkpoint weights were reported.

## Language research: other local tools

Lynote's installed CLI is the lightweight heuristic engine, not the separate
XLM-R/ModernBERT production training pipeline. Its marker and transition lists
are English. ASCII word matching splits accented French tokens; transitions
are counted as substrings. Its book scores must not be read as calibrated
French authorship probabilities. The Typer/Rich CLI upgrade leaves core scoring
unchanged and makes the limitations inspectable through `lynote-ai info`.

Uncanny's compression algorithm can process arbitrary bytes, but its AI/human
reference preludes are English and stylistically contrasting. Its optional
perplexity analyzer uses English GPT-2. Language-neutral arithmetic is not
language-neutral validation. A reasonable future extension would use per-language
reference corpora, Unicode-aware segmentation, configurable LM backends and
language/domain-specific calibration with an abstention state.

Sources, consulted 2026-10-26:

- https://huggingface.co/almanach/camemberta-chatgptdetect-noisy
- https://huggingface.co/almanach/xlmr-chatgptdetect-noisy
- https://huggingface.co/datasets/almanach/hc3_french_ood
- https://github.com/lynote-ai/ai-text-detector
- https://github.com/dunkinfrunkin/uncanny
- https://github.com/thinkst/zippy (explicit non-English prelude caveat)
