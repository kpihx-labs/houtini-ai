"""Offline lifecycle, language routing and metric contracts."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ai_detect import detector
from ai_detect.backends import WindowedBackend
from ai_detect.cli import main
from ai_detect.models import DEFAULT_MODEL, MODELS, cached_path, ensure_model, model_spec
from ai_detect.patterns import diagnose_sentence


def test_registry_and_alias():
    assert set(MODELS) == {"en", "en-light", "fr", "en+fr"}
    assert DEFAULT_MODEL == "en+fr"
    assert model_spec("fr+en") is model_spec("en+fr")
    with pytest.raises(ValueError):
        model_spec("desklib")


def test_import_does_not_import_ml_runtimes():
    result = subprocess.run([sys.executable, "-c", ("import ai_detect,sys; "
                             "assert not {'torch','transformers','onnxruntime'} & sys.modules.keys()")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_catalogue_never_downloads(monkeypatch, capsys):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("download"))
    assert main(["models", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 4
    assert all(row["repo"] and row["languages"] for row in rows)


def test_cache_requires_tokenizer_and_weights(monkeypatch, tmp_path):
    import huggingface_hub
    spec = model_spec("fr")
    for name in spec.required_files:
        (tmp_path / name).touch()
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, name, **kw: str(tmp_path/name))
    assert cached_path(spec) == tmp_path
    (tmp_path / "tokenizer.json").unlink()
    assert cached_path(spec) is None


def test_complete_cache_never_calls_snapshot_download(monkeypatch, tmp_path):
    from ai_detect import models
    monkeypatch.setattr(models, "cached_path", lambda spec: tmp_path)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: pytest.fail("download"))
    assert ensure_model(model_spec("fr")) == (tmp_path, False)


def test_missing_cache_downloads_only_requested_model(monkeypatch, tmp_path):
    from ai_detect import models
    spec = model_spec("fr")
    for name in spec.required_files:
        (tmp_path / name).touch()
    monkeypatch.setattr(models, "cached_path", lambda spec: None)
    calls = []
    def download(repo, **kwargs):
        calls.append((repo, kwargs))
        return str(tmp_path)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    assert ensure_model(spec) == (tmp_path, True)
    assert calls[0][0] == spec.repo
    assert "pytorch_model.bin" not in calls[0][1]["allow_patterns"]


def test_empty_or_short_input_does_not_load(monkeypatch):
    monkeypatch.setattr(detector, "get_detector", lambda *a, **k: pytest.fail("model loaded"))
    for text in ("", "Bonjour."):
        result = detector.classify_text(text)
        assert result["model_ai_pct"] is None
        assert result["scored_count"] == 0
        if result["sentences"]:
            assert result["sentences"][0]["ai_prob"] is None


def test_average_score_is_not_ai_sentence_fraction():
    fake = SimpleNamespace(score=lambda s: [0.51, 0.99, 0.49])
    text = "First sentence has enough words. Second sentence has enough words. Third sentence has enough words."
    result = detector.classify_text(text, model="fr", detector=fake)
    assert result["model_ai_pct"] == 66.3
    assert result["ai_sentence_pct"] == 66.7
    assert result["ai_count"] == 2
    assert result["human_count"] == 1
    assert result["timings"]["score_seconds"] >= 0


@pytest.mark.parametrize("bad", [[float("nan")], [1.5], []])
def test_invalid_backend_scores_rejected(bad):
    with pytest.raises(ValueError):
        detector.classify_text("This sentence has enough words.", detector=SimpleNamespace(score=lambda s: bad))


def test_alias_shares_one_loaded_backend(monkeypatch):
    monkeypatch.setattr(detector, "_DETECTOR_CACHE", {})
    calls = []
    def factory(spec, device=None):
        calls.append(spec.key)
        return object()
    monkeypatch.setitem(detector.BACKENDS, "sequence", factory)
    assert detector.get_detector("fr+en") is detector.get_detector("en+fr")
    assert calls == ["en+fr"]


def test_french_diagnostics_do_not_apply_english_rules():
    findings = diagnose_sentence("En outre, il convient de vérifier le résultat.", ("fr",))
    assert findings and all(f["language"] == "fr" for f in findings)
    assert not diagnose_sentence("It is important to note this result.", ("fr",))
    assert diagnose_sentence("Additionally this result matters.", ("en",))


def test_overflow_windows_averaged_per_sentence():
    class FakeBackend(WindowedBackend):
        def __init__(self):
            self._setup(model_spec("fr"))
            self.tokenizer = lambda *a, **k: {"input_ids": [[1], [2], [3]],
                                             "overflow_to_sample_mapping": [0, 0, 1]}
        def _infer(self, enc):
            return [0.2, 0.8, 0.9]
    backend = FakeBackend()
    assert backend.score(["long", "short"]) == [0.5, 0.9]
    assert backend.window_count == 3
