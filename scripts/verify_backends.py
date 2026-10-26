"""Offline integration cross-check: adapters vs direct checkpoint inference.

Use a monitored tmux pane. No accuracy claim: synthetic samples are plumbing
controls only. Class labels come from the checkpoint, not the adapter's index.
"""

import json
from pathlib import Path
from time import perf_counter

from ai_detect.backends import OnnxBackend, TorchBackend
from ai_detect.models import model_spec

SAMPLES = [
    "Le voisin a oublié ses clés dans la boîte aux lettres hier soir.",
    "En tant qu'intelligence artificielle, je ne possède ni émotions ni expériences personnelles.",
    "It is important to note that this comprehensive framework provides a robust solution.",
    "I left my notebook on the kitchen table and went back to fetch it.",
]


def main():
    import numpy as np
    import torch

    rows = []
    for name in ("fr", "en+fr", "en-light"):
        start = perf_counter()
        spec = model_spec(name)
        backend = OnnxBackend(spec, "cpu") if name == "en-light" else TorchBackend(spec, "xpu")
        got = backend.score(SAMPLES)
        config = json.loads((backend.path / "config.json").read_text())
        labels = config["id2label"]
        ai_indices = [int(k) for k, value in labels.items() if value.casefold() in ("ai", "chatgpt", "artificial")]
        assert ai_indices == [spec.ai_index], labels
        if isinstance(backend, OnnxBackend):
            enc = backend.tokenizer(SAMPLES, padding=True, truncation=True,
                                    max_length=512, return_tensors="np")
            logits = backend.session.run(None, {k: v for k, v in enc.items() if k in backend.inputs})[0]
            exp = np.exp(logits-logits.max(axis=1, keepdims=True))
            reference = (exp/exp.sum(axis=1, keepdims=True))[:, ai_indices[0]].tolist()
        else:
            # A direct CPU forward pass, independent of the adapter's window/feed code.
            backend.model.to("cpu")
            inputs = backend.tokenizer(SAMPLES, padding=True, truncation=True,
                                       max_length=512, return_tensors="pt")
            with torch.inference_mode():
                logits = backend.model(**inputs).logits
                reference = logits.softmax(dim=-1)[:, ai_indices[0]].tolist()
        delta = max(abs(a-b) for a, b in zip(got, reference, strict=True))
        assert delta < 0.001, (name, got, reference, delta)
        row = {"model": name, "labels": labels, "adapter": got, "direct_cpu_reference": reference,
               "max_delta": delta, "elapsed_seconds": perf_counter()-start}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del backend
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
    output = Path.home()/"RAW"/"detectors"/"houtini-backend-verification.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"VERIFIED: {output}", flush=True)


if __name__ == "__main__":
    main()
