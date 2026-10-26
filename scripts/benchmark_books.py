"""Reproducible cold-process CLI timings; keep all JSON, stderr and input hashes.

Run in a monitored tmux pane with the installed Houtini interpreter. This script
never modifies inputs, never uploads text and never overwrites an existing run.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--models", nargs="+", default=["fr", "fr+en", "en-light"])
    parser.add_argument("--output-parent", type=Path, required=True)
    args = parser.parse_args()
    for source in args.files:
        if not source.is_file():
            parser.error(f"Missing input: {source}")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_parent / f"houtini-multilingual-{timestamp}"
    output.mkdir(parents=False, exist_ok=False)
    rows = []
    print(f"Artifacts: {output}", flush=True)
    for model in args.models:
        for source in args.files:
            stem = f"{model.replace('+', '-')}_{source.stem}"
            start = perf_counter()
            print(f"START {model} {source.name}", flush=True)
            result = subprocess.run(
                [sys.executable, "-m", "ai_detect.cli", "scan", str(source),
                 "--model", model, "--json"], capture_output=True, text=True,
            )
            elapsed = perf_counter() - start
            (output / f"{stem}.stderr.txt").write_text(result.stderr, encoding="utf-8")
            (output / f"{stem}.json").write_text(result.stdout, encoding="utf-8")
            if result.returncode:
                print(result.stderr, file=sys.stderr, flush=True)
                raise RuntimeError(f"Benchmark failed ({result.returncode}): {model} {source}")
            report = json.loads(result.stdout)
            row = {"requested_model": model, "file": str(source),
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                   "process_wall_seconds": round(elapsed, 6),
                   **{key: report[key] for key in (
                       "model", "model_info", "model_ai_pct", "ai_sentence_pct", "timings",
                       "ai_count", "human_count", "skipped_count", "scored_count", "total", "token_windows")}}
            rows.append(row)
            (output / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    print(f"DONE: {len(rows)} successful runs in {output}", flush=True)


if __name__ == "__main__":
    main()
