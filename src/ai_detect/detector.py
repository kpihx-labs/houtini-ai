"""Model-independent orchestration and explicit lazy lifecycle.

No model is prefetched or loaded at import. Catalogue data lives in models.py;
runtime-specific inference lives in backends.py. Diagnostics never change scores.
"""

import math
import os
import threading
from time import perf_counter

from .backends import BACKENDS, resolve_torch_device, synchronize  # noqa: F401
from .models import DEFAULT_MODEL, MODELS, ensure_model, is_model_cached, model_spec  # noqa: F401
from .patterns import calculate_sdsl, count_pattern_totals, diagnose_sentence, sent_tokenize

MIN_WORDS = 5
_DETECTOR_CACHE = {}
_DETECTOR_LOCKS = {}
_DETECTOR_LOCKS_GUARD = threading.Lock()


def verdict(pct):
    return "LIKELY AI" if pct > 50 else "LIKELY HUMAN" if pct < 40 else "MIXED"


def preload_dependencies():
    """Compatibility no-op: importing every runtime violates lazy selection."""


def prefetch_model(model_name=DEFAULT_MODEL):
    """Explicit opt-in helper only; never invoked by imports or listing."""
    ensure_model(model_spec(model_name))
    return True


def _device_key(device):
    key = (device or os.environ.get("AI_DETECT_DEVICE") or "auto").strip().lower()
    return "cuda" if key == "gpu" else key


def get_detector(model=DEFAULT_MODEL, device=None):
    spec = model_spec(model)
    key = (spec.key, _device_key(device))
    with _DETECTOR_LOCKS_GUARD:
        lock = _DETECTOR_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if key not in _DETECTOR_CACHE:
            factory = BACKENDS.get(spec.backend)
            if factory is None:
                raise ValueError(f"No backend registered for {spec.backend}")
            _DETECTOR_CACHE[key] = factory(spec, device=device)
    return _DETECTOR_CACHE[key]


def classify_text(text, model=DEFAULT_MODEL, device=None, detector=None):
    """Return sentence scores, denominators, model identity and phase timings.

    model_ai_pct is the mean model probability, not the AI-labelled fraction.
    Short fragments abstain (null probabilities); empty/short-only input loads
    no model. Both metrics are descriptive, not an authorship determination.
    """
    started = perf_counter()
    spec = model_spec(model)
    sentences = sent_tokenize(text)
    results = []
    scoreable, indices = [], []
    for sentence in sentences:
        if len(sentence.split()) < MIN_WORDS:
            results.append({"sentence": sentence, "label": "Skipped", "confidence": None,
                            "ai_prob": None, "human_prob": None, "diagnostics": [],
                            "skip_reason": f"fewer than {MIN_WORDS} whitespace words"})
        else:
            indices.append(len(results))
            scoreable.append(sentence)
            results.append(None)

    load_seconds = score_seconds = 0.0
    probs = []
    if scoreable:
        t = perf_counter()
        if detector is None:
            detector = get_detector(spec.key, device)
        load_seconds = perf_counter() - t
        synchronize(getattr(detector, "device", None))
        t = perf_counter()
        probs = detector.score(scoreable)
        synchronize(getattr(detector, "device", None))
        score_seconds = perf_counter() - t
        if len(probs) != len(scoreable):
            raise ValueError("Backend score count does not match sentence count")
        for sentence, prob, index in zip(scoreable, probs, indices, strict=True):
            prob = float(prob)
            if not math.isfinite(prob) or not 0 <= prob <= 1:
                raise ValueError(f"Invalid model probability: {prob}")
            results[index] = {
                "sentence": sentence, "label": "AI" if prob >= 0.5 else "Human",
                "confidence": round(max(prob, 1-prob), 4), "ai_prob": round(prob, 4),
                "human_prob": round(1-prob, 4),
                "diagnostics": diagnose_sentence(sentence, languages=spec.languages),
            }
    ai_count = sum(p >= 0.5 for p in probs)
    metadata = {
        "repo": spec.repo, "languages": list(spec.languages), "backend": spec.backend,
        "revision_requested": spec.revision, "caveat": spec.caveat,
        "device": str(getattr(detector, "device", "not loaded")),
        "resolved_revision": getattr(getattr(detector, "path", None), "name", None),
        "downloaded_this_load": getattr(detector, "downloaded", False),
    }
    return {
        "sentences": results, "model_ai_pct": round(sum(probs)/len(probs)*100, 1) if probs else None,
        "ai_sentence_pct": round(ai_count/len(probs)*100, 1) if probs else None,
        "ai_count": ai_count, "human_count": len(probs)-ai_count,
        "skipped_count": len(sentences)-len(probs), "scored_count": len(probs),
        "total": len(sentences), "model": spec.key, "model_info": metadata,
        "text_metrics": calculate_sdsl(scoreable), "pattern_totals": count_pattern_totals(results),
        "token_windows": getattr(detector, "window_count", len(probs)),
        "timings": {"load_seconds": round(load_seconds, 6),
                    "score_seconds": round(score_seconds, 6),
                    "total_seconds": round(perf_counter()-started, 6)},
        "caveats": ["Scores are not proof of authorship.",
                    "Diagnostics are lexical style observations, not causal model explanations.",
                    "Short sentences abstain; long sentences use averaged token-window scores."],
    }
