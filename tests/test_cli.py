"""CLI surface tests that do not load a detection model."""

import json
from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

from ai_detect import __version__
from ai_detect.cli import app, main

runner = CliRunner()


def test_version_prints_installed_distribution_version(capsys):
    try:
        main(["--version"])
    except SystemExit as result:
        assert result.code == 0
    assert capsys.readouterr().out.strip() == f"houtini-ai {__version__}"


@pytest.mark.parametrize("args,expected", [
    (["--help"], "Model catalogue"),
    (["scan", "--help"], "Examples"),
    (["compare", "--help"], "second"),
    (["models", "--help"], "info"),
    (["models", "info", "--help"], "full name"),
    (["--help-all"], "houtini-ai models info"),
])
def test_recursive_help(args, expected):
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert expected in result.output


def test_model_info_alias_json():
    result = runner.invoke(app, ["models", "info", "fr+en", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["key"] == "en+fr"


@pytest.mark.parametrize("args", [
    ["scan", "--text", "test", "--model", "INVALID"],
    ["scan", "/does/not/exist.txt"],
    ["scan", "--text", "hello", "--file", "another.txt"],
    ["models", "info", "unknown"],
    ["scan", "--text", "a complete sentence has many words.", "--device", "invalid"],
])
def test_cli_errors_are_concise(args):
    result = runner.invoke(app, args)
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_short_scan_json_does_not_load(monkeypatch):
    from ai_detect import detector
    monkeypatch.setattr(detector, "get_detector", lambda *a, **k: pytest.fail("loaded"))
    result = runner.invoke(app, ["scan", "--text", "Bonjour.", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["model_ai_pct"] is None


@pytest.mark.parametrize("width", [60, 80, 110])
def test_catalogue_preserves_full_identifiers_at_terminal_width(monkeypatch, width):
    from ai_detect import cli
    from ai_detect.models import MODELS
    stream = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=stream, width=width, force_terminal=False))
    cli._list_models(False)
    output = stream.getvalue()
    assert "…" not in output
    recovered = {}
    current = None
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.split("│")[1:-1]]
        if len(cells) != 5:
            continue
        if cells[0] in MODELS:
            current = cells[0]
            recovered[current] = ""
        if current is not None:
            recovered[current] += cells[-1]
    assert recovered == {key: spec.repo for key, spec in MODELS.items()}
