"""Lazy inference adapters: generic HF classifiers, Desklib head and ONNX.

All loaders use the selected local snapshot only. No remote code is executed.
Long sentences use non-overlapping token windows instead of silent truncation.
Window scores are averaged per sentence, keeping the report denominator stable.
"""

import os
import sys
import threading
from typing import Any, cast

from .models import ensure_model


def resolve_torch_device(pref=None):
    import torch

    pref = (pref or os.environ.get("AI_DETECT_DEVICE") or "auto").lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref in ("xpu", "cuda", "gpu"):
        name = "cuda" if pref == "gpu" else pref
        module = getattr(torch, name, None)
        if module is not None and module.is_available():
            return torch.device(name)
        print(f"Requested {name.upper()} is unavailable; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    if pref != "auto":
        raise ValueError(f"Unknown device: {pref}")
    for name in ("cuda", "xpu"):
        module = getattr(torch, name, None)
        if module is not None and module.is_available():
            return torch.device(name)
    return torch.device("cpu")


def _onnx_providers(ort, device):
    available = ort.get_available_providers()
    pref = (device or os.environ.get("AI_DETECT_DEVICE") or "auto").lower()
    if pref == "cpu":
        return ["CPUExecutionProvider"]
    if pref == "xpu":
        print("ONNX XPU unavailable; using CPUExecutionProvider.", file=sys.stderr)
        return ["CPUExecutionProvider"]
    gpu = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider")
           if p in available]
    if pref in ("cuda", "gpu"):
        gpu = [p for p in gpu if p == "CUDAExecutionProvider"]
        if not gpu:
            print("ONNX CUDA unavailable; using CPUExecutionProvider.", file=sys.stderr)
    return gpu + ["CPUExecutionProvider"]


def synchronize(device):
    if getattr(device, "type", None) in ("xpu", "cuda"):
        import torch
        getattr(torch, device.type).synchronize()


class WindowedBackend:
    """Shared overflow accounting and lock across inference runtimes."""

    tokenizer: Any

    def _infer(self, enc):
        raise NotImplementedError

    def _setup(self, spec):
        self.spec = spec
        self.name = spec.key
        self.lock = threading.Lock()
        self.window_count = 0

    def score(self, sentences):
        scores = []
        windows = 0
        with self.lock:
            batch_size = self.spec.batch_size
            for start in range(0, len(sentences), batch_size):
                batch = sentences[start:start + batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.spec.max_length,
                    return_overflowing_tokens=True, stride=0,
                )
                mapping = enc.pop("overflow_to_sample_mapping")
                totals, counts = [0.0] * len(batch), [0] * len(batch)
                for offset in range(0, len(mapping), batch_size):
                    inputs = {k: v[offset:offset+batch_size] for k, v in enc.items()}
                    probs = self._infer(inputs)
                    indices = mapping[offset:offset+batch_size]
                    if len(probs) != len(indices):
                        raise RuntimeError("Backend returned the wrong number of window scores")
                    for index, probability in zip(indices, probs, strict=True):
                        totals[index] += float(probability)
                        counts[index] += 1
                scores.extend(t / c for t, c in zip(totals, counts, strict=True))
                windows += len(mapping)
            self.window_count = windows
        return scores


class TorchBackend(WindowedBackend):
    def __init__(self, spec, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._setup(spec)
        self.path, self.downloaded = ensure_model(spec)
        self.device = resolve_torch_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model: Any
        if spec.backend == "desklib":
            self.model = _desklib_model(self.path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.path, local_files_only=True, dtype=torch.float32,
            )
            labels = self.model.config.id2label
            if len(labels) != 2 or spec.ai_index not in labels:
                raise ValueError(f"Expected binary classifier with AI class {spec.ai_index}: {labels}")
        cast(Any, self.model).to(self.device).eval()
        synchronize(self.device)
        print(f"Loaded {spec.key}: {spec.repo} on {self.device}", file=sys.stderr)

    def _infer(self, enc):
        import torch

        inputs = {k: torch.tensor(v, dtype=torch.long, device=self.device)
                  for k, v in enc.items() if k in self.tokenizer.model_input_names}
        with torch.inference_mode():
            output = self.model(**inputs)
            logits = output["logits"]
            if self.spec.backend == "desklib":
                return torch.sigmoid(logits).squeeze(-1).cpu().tolist()
            return torch.softmax(logits, dim=-1)[:, self.spec.ai_index].cpu().tolist()


def _desklib_model(path):
    """Upstream checkpoint uses mean pooling + one sigmoid logit, not AutoModelForSequenceClassification."""
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel, PreTrainedModel

    class DesklibModel(PreTrainedModel):
        config_class = cast(Any, AutoConfig)
        all_tied_weights_keys = {}

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.init_weights()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            output = self.model(input_ids, attention_mask=attention_mask)[0]
            mask = attention_mask.unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return {"logits": self.classifier(pooled)}

    return DesklibModel.from_pretrained(path, local_files_only=True, dtype=torch.float32)


class OnnxBackend(WindowedBackend):
    def __init__(self, spec, device=None):
        import importlib

        ort = importlib.import_module("onnxruntime")
        from transformers import AutoTokenizer

        self._setup(spec)
        self.path, self.downloaded = ensure_model(spec)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        options = ort.SessionOptions()
        # Avoid oversubscription on laptops with many logical cores.
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        self.session = ort.InferenceSession(
            str(self.path / spec.weights), sess_options=options,
            providers=_onnx_providers(ort, device),
        )
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.device = self.session.get_providers()[0]
        print(f"Loaded {spec.key}: {spec.repo} on {self.device}", file=sys.stderr)

    def _infer(self, enc):
        import numpy as np

        feeds = {k: np.asarray(v, dtype=np.int64) for k, v in enc.items() if k in self.inputs}
        if "token_type_ids" in self.inputs and "token_type_ids" not in feeds:
            feeds["token_type_ids"] = np.zeros_like(feeds["input_ids"])
        logits = self.session.run(None, feeds)[0]
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected two-class ONNX logits, got {logits.shape}")
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (exp / exp.sum(axis=1, keepdims=True))[:, self.spec.ai_index].tolist()


BACKENDS = {"desklib": TorchBackend, "sequence": TorchBackend, "onnx": OnnxBackend}
