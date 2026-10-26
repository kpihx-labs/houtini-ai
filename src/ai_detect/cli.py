"""Typer/Rich CLI; help and catalogue operations never initialize inference."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typer.main import get_command

from . import __version__
from .detector import classify_text
from .models import DEFAULT_MODEL, catalogue, model_spec

console = Console()
app = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="rich",
                  help="[bold]Houtini[/bold] - local EN/FR text detection, with explicit model provenance and timings.",
                  epilog="Models: en, en-light, fr, en+fr (default; alias fr+en). Explore: scan --help, compare --help, models --help. Full reference: --help-all. Scores are not proof of authorship.")
models_app = typer.Typer(invoke_without_command=True, no_args_is_help=False,
                         rich_markup_mode="rich")
app.add_typer(models_app, name="models", rich_help_panel="Model catalogue")


def _version(value: bool):
    if value:
        typer.echo(f"houtini-ai {__version__}")
        raise typer.Exit()


def _help_all(value: bool):
    if not value:
        return
    root = get_command(app)
    def render(command, name, parent=None):
        context = typer.Context(command, info_name=name, parent=parent)
        typer.echo(command.get_help(context))
        if hasattr(command, "commands"):
            for key, child in command.commands.items():
                render(child, key, context)
    render(root, "houtini-ai")
    raise typer.Exit()


@app.callback()
def root(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True,
                                         help="Installed distribution version.")] = False,
    help_all: Annotated[bool, typer.Option("--help-all", callback=_help_all, is_eager=True,
                                          help="Render every command's help, parameters and examples.")] = False,
):
    """Model selection: en, en-light, fr, en+fr (default); fr+en is an alias."""


def _model(value: str):
    try:
        return model_spec(value).key
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


Model = Annotated[str, typer.Option("--model", "-m", callback=_model,
                                   help="en | en-light | fr | en+fr; fr+en aliases en+fr. Selected model downloads only if absent.",
                                   rich_help_panel="Inference")]
Device = Annotated[str, typer.Option("--device", "-d", help="auto | xpu | cpu | cuda. Auto: CUDA, then XPU, then CPU.",
                                    rich_help_panel="Inference")]
Json = Annotated[bool, typer.Option("--json", "-j", help="Emit full machine-readable JSON only on stdout.", rich_help_panel="Output")]


def _device(value):
    if value not in ("auto", "xpu", "cpu", "cuda"):
        raise typer.BadParameter("Device must be auto, xpu, cpu or cuda", param_hint="--device")
    return value


def print_results(data, label="", sentences=False):
    """Compact default output; opt in to potentially thousands of sentence rows."""
    info = data["model_info"]
    score = data["model_ai_pct"]
    table = Table(title=label or "Detection summary", show_header=False)
    for key, value in (
        ("Model", f"{data['model']} - {info['repo']}"),
        ("Languages / device", f"{', '.join(info['languages'])} / {info['device']}"),
        ("Mean model score", "Not scored" if score is None else f"{score:.1f}% (not an authorship probability)"),
        ("AI-labelled fraction", f"{data['ai_sentence_pct']}% of scored sentences"),
        ("Sentences", f"{data['ai_count']} AI / {data['human_count']} human / {data['skipped_count']} skipped"),
        ("Token windows", data["token_windows"]),
        ("Load / scoring / total", " / ".join(f"{data['timings'][k]:.3f}s" for k in ("load_seconds", "score_seconds", "total_seconds"))),
    ):
        table.add_row(str(key), str(value))
    console.print(table)
    console.print(Panel(info["caveat"] + " Lexical diagnostics are not causal explanations.", title="Limitations"))
    if data["pattern_totals"]:
        console.print("Style markers:", data["pattern_totals"])
    if sentences:
        for row in data["sentences"]:
            console.print(row, markup=False)


def _read(path):
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise typer.BadParameter(f"Cannot read UTF-8 file {path}: {exc}") from exc


def _classify(text, model, device):
    if not text.strip():
        raise typer.BadParameter("No text provided. Pass a file, --text or UTF-8 stdin.")
    try:
        return classify_text(text, model=model, device=_device(device))
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(f"Detection failed: {exc}") from exc


@app.command(rich_help_panel="Analysis")
def scan(
    path: Annotated[Path | None, typer.Argument(help="UTF-8 file; omit to read stdin. Use --text for literal prose.")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Explicit UTF-8 file path.", rich_help_panel="Input")] = None,
    text: Annotated[str | None, typer.Option("--text", "-t", help="Literal text; mutually exclusive with a file.", rich_help_panel="Input")] = None,
    model: Model = DEFAULT_MODEL, device: Device = "auto", json_output: Json = False,
    sentences: Annotated[bool, typer.Option("--sentences", help="Print every scored/skipped sentence (full JSON always includes them).", rich_help_panel="Output")] = False,
):
    """Score a document locally; preserve skipped fragments and model identity.

    [bold]Examples[/bold]

    houtini-ai scan draft.txt --model fr --json

    houtini-ai scan --text 'This is a complete example sentence.' --model en-light

    houtini-ai scan bilingual.txt --model fr+en --device xpu

    [bold]Output[/bold]: mean score, AI-labelled fraction, counts, lexical markers,
    snapshot revision and load/scoring/total seconds. Long sentences are windowed.
    The first uncached run may download weights; cached runs load locally.
    """
    if sum(x is not None for x in (path, file, text)) > 1:
        raise typer.BadParameter("Choose exactly one of PATH, --file or --text")
    source = file or path
    content = _read(source) if source else text if text is not None else sys.stdin.read()
    data = _classify(content, model, device)
    if json_output:
        typer.echo(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print_results(data, str(source or "Literal text / stdin"), sentences)


@app.command(rich_help_panel="Analysis")
def compare(
    first: Annotated[Path, typer.Argument(help="First UTF-8 file.")],
    second: Annotated[Path, typer.Argument(help="Second UTF-8 file.")],
    model: Model = DEFAULT_MODEL, device: Device = "auto", json_output: Json = False,
):
    """Score two files sequentially with one lazily loaded model.

    Examples: houtini-ai compare first.txt second.txt --model fr --json

    Output: two full reports with separate timings. The second load normally
    reuses the in-process backend. This is not evidence of relative authorship.
    """
    # Read both before initializing a potentially expensive model.
    sources = [(p, _read(p)) for p in (first, second)]
    reports = [{"file": str(p), **_classify(text, model, device)} for p, text in sources]
    if json_output:
        typer.echo(json.dumps(reports, ensure_ascii=False, indent=2))
    else:
        for report in reports:
            print_results(report, report["file"])


def _list_models(json_output):
    rows = catalogue()
    if json_output:
        typer.echo(json.dumps(rows, indent=2))
        return
    table = Table(title=f"Houtini models - default {DEFAULT_MODEL}")
    for header in ("Key", "Languages", "Backend", "Cache", "Full Hub name"):
        table.add_column(header)
    for row in rows:
        table.add_row(row["key"], ", ".join(row["languages"]), row["backend"],
                      "ready" if row["cached"] else "not cached", row["repo"])
    console.print(table)
    console.print("fr+en = en+fr (one bilingual checkpoint, not an ensemble). No downloads on listing.")


@models_app.callback()
def models(ctx: typer.Context, json_output: Json = False):
    """Discover model readiness and full Hub identifiers. No weight downloads.

    [bold]Examples[/bold]

    houtini-ai models

    houtini-ai models --json

    houtini-ai models info fr

    [bold]Output[/bold]: languages, backend, cache readiness and the exact upstream
    repository. Use list/info --help for details. Nothing is downloaded or loaded.
    """
    if ctx.invoked_subcommand is None:
        _list_models(json_output)


@models_app.command("list")
def list_models(json_output: Json = False):
    """List every supported model without network access.

    Example: houtini-ai models list --json
    """
    _list_models(json_output)


@models_app.command("info")
def model_info(name: Annotated[str, typer.Argument(help="Model key or fr+en alias.")], json_output: Json = False):
    """Inspect one model's full name, languages, files and limitations.

    Example: houtini-ai models info fr --json

    Output: declarative specification and complete-cache readiness. No download.
    """
    key = _model(name)
    row = next(row for row in catalogue() if row["key"] == key)
    if json_output:
        typer.echo(json.dumps(row, indent=2))
    else:
        console.print(Panel(json.dumps(row, ensure_ascii=False, indent=2), title=key), markup=False)


def main(argv=None):
    """Entry point plus narrow compatibility for the historical file/options CLI."""
    # Keep the Windows UTF-8 safety guard from the original CLI: arbitrary user
    # prose must not fail because cp1252/cp850 cannot render a diagnostic glyph.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] not in {"scan", "compare", "models", "--help", "-h", "--help-all", "--version"}:
        if "--compare" in args or "-c" in args:
            index = args.index("--compare") if "--compare" in args else args.index("-c")
            args = ["compare", *args[:index], *args[index+1:]]
        else:
            args.insert(0, "scan")
    try:
        result = app(args=args, prog_name="houtini-ai", standalone_mode=True)
        return result if isinstance(result, int) else 0
    except SystemExit as exc:
        return int(exc.code or 0)


if __name__ == "__main__":
    sys.exit(main())
