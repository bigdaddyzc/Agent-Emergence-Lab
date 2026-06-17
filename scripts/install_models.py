"""Install Ollama models declared in config or passed on the command line."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_models_from_config(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    agents = data.get("agents", {})
    models = []
    for key in ("agent_a", "agent_b"):
        model = agents.get(key, {}).get("model")
        if model:
            models.append(model)
    return _dedupe(models)


def list_installed_models() -> set[str]:
    proc = subprocess.run(
        ["ollama", "list"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Cannot run 'ollama list'. Start Ollama with 'ollama serve' "
            f"or check installation.\n{proc.stderr.strip()}"
        )
    installed = set()
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            installed.add(parts[0])
    return installed


def pull_model(model: str) -> None:
    print(f"Installing {model} via Ollama...")
    proc = subprocess.run(["ollama", "pull", model], check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to install model: {model}")


def install_models(models: list[str]) -> list[str]:
    installed = list_installed_models()
    pulled = []
    for model in _dedupe(models):
        if model in installed:
            print(f"Already installed: {model}")
            continue
        pull_model(model)
        pulled.append(model)
    final = list_installed_models()
    missing = [m for m in models if m not in final]
    if missing:
        raise RuntimeError(f"Models still missing after install: {', '.join(missing)}")
    return pulled


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install Ollama models for Agent Emergence Lab")
    parser.add_argument("--config", type=Path, help="Config file to read models from")
    parser.add_argument("--models", nargs="*", default=[], help="Explicit model names")
    args = parser.parse_args(argv)

    models = list(args.models)
    if args.config:
        models.extend(load_models_from_config(args.config))
    models = _dedupe(models)
    if not models:
        print("No models requested. Use --config or --models.", file=sys.stderr)
        return 2

    try:
        pulled = install_models(models)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if pulled:
        print("Installed: " + ", ".join(pulled))
    else:
        print("All requested models were already installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
