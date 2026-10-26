"""Dependency-light model catalogue. Adding a standard HF classifier is data-only.

Listing never imports torch/transformers/ONNX or contacts the Hub. Revisions are
recorded in reports; checkpoint licences and calibration remain upstream concerns.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    languages: tuple[str, ...]
    backend: str
    approx_mb: int
    weights: str = "model.safetensors"
    ai_index: int = 1
    max_length: int = 512
    revision: str = "main"
    batch_size: int = 8
    tokenizer_files: tuple[str, ...] = ("tokenizer_config.json", "tokenizer.json")
    caveat: str = "Research signal, not a calibrated authorship probability."

    @property
    def required_files(self):
        # Tokenizer completeness matters: a weights-only cache is not ready.
        return ("config.json", *self.tokenizer_files, self.weights)


MODELS = {
    spec.key: spec for spec in (
        ModelSpec("en", "desklib/ai-text-detector-v1.01", ("en",), "desklib", 1736),
        ModelSpec("en-light", "onnx-community/tmr-ai-text-detector-ONNX", ("en",),
                  "onnx", 126, weights="onnx/model_int8.onnx"),
        ModelSpec("fr", "almanach/camemberta-chatgptdetect-noisy", ("fr",),
                  "sequence", 450, caveat="2023 HC3 French ChatGPT detector; translated training data, domain shift; fine-tune licence unspecified."),
        ModelSpec("en+fr", "almanach/xlmr-chatgptdetect-noisy", ("en", "fr"),
                  "sequence", 1110, caveat="2023 HC3 EN/FR ChatGPT detector; domain shift and adversarial fragility."),
    )
}
ALIASES = {"fr+en": "en+fr"}
DEFAULT_MODEL = "en+fr"


def model_spec(name):
    key = ALIASES.get(name, name)
    if key not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {', '.join(MODELS)}")
    return MODELS[key]


def cached_path(spec):
    """Resolve a complete local snapshot, with no network request or download."""
    from pathlib import Path

    from huggingface_hub import try_to_load_from_cache

    paths = [try_to_load_from_cache(spec.repo, f, revision=spec.revision)
             for f in spec.required_files]
    if not all(isinstance(p, str) and Path(p).is_file() for p in paths):
        return None
    root = Path(str(paths[0])).parent
    # Never combine files from two revisions of a repository.
    if not all((root / f).is_file() for f in spec.required_files):
        return None
    return root


def is_model_cached(name):
    try:
        return cached_path(model_spec(name)) is not None
    except (ImportError, ValueError, OSError):
        return False


def ensure_model(spec):
    """Only a requested, incomplete model may trigger a download."""
    import sys

    from huggingface_hub import snapshot_download

    root = cached_path(spec)
    if root is not None:
        return root, False
    print(f"Downloading requested model {spec.key}: {spec.repo} (~{spec.approx_mb} MB).",
          file=sys.stderr)
    root = snapshot_download(
        spec.repo, revision=spec.revision,
        allow_patterns=["config.json", "*token*", "*.model", "*.spm", "vocab*",
                        "merges*", "added_tokens.json", spec.weights],
    )
    from pathlib import Path
    root = Path(root)
    missing = [f for f in spec.required_files if not (root / f).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete model {spec.repo}: missing {missing}")
    return root, True


def catalogue():
    return [{**asdict(s), "cached": is_model_cached(s.key),
             "aliases": [a for a, k in ALIASES.items() if k == s.key]}
            for s in MODELS.values()]
