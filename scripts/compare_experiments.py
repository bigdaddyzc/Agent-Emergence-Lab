"""Re-score an experiment manifest and print the baseline comparison.

Usage:
    python scripts/compare_experiments.py logs/experiments/latest.json [--baseline single_agent]

Useful after changing emergence patterns in config: the transcripts are
re-scored from their JSONL logs, so old runs can be compared with new rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment import EXPERIMENT_MODES, format_comparison, score_session  # noqa: E402


def rescore(manifest: list[dict], emergence_config: dict | None = None) -> list[dict]:
    rescored = []
    for item in manifest:
        path = Path(item["jsonl_path"])
        if not path.exists():
            print(f"warning: missing log {path}, keeping stored scores", file=sys.stderr)
            rescored.append(item)
            continue
        rescored.append({**item, "scores": score_session(path, emergence_config)})
    return rescored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare experiment modes from a manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--baseline", choices=EXPERIMENT_MODES)
    parser.add_argument("--config", type=Path,
                        help="Config whose emergence patterns to score with (default: built-in)")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    emergence = None
    if args.config:
        import yaml
        emergence = (yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}).get("emergence")
    print(format_comparison(rescore(manifest, emergence), args.baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
