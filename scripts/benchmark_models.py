"""Benchmark small Ollama models for low-resource CPU use."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ollama_client import OllamaClient
from src.model_roles import required_runtime_models


PROMPTS = [
    "用三句话解释什么是长期记忆。",
    "把记忆系统类比成图书馆，并指出这个类比哪里不成立。",
    "回应对方观点，并提出一个具体质疑。",
    "按 类型|新鲜度|内容 输出两条观察。",
]


def load_models_from_config(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return required_runtime_models(data)


def benchmark_model(client: OllamaClient, model: str, rounds: int, max_tokens: int) -> dict:
    runs = []
    for idx in range(rounds):
        prompt = PROMPTS[idx % len(PROMPTS)]
        started = time.perf_counter()
        error = ""
        try:
            resp = client.generate(
                model=model,
                prompt=prompt,
                temperature=0.3,
                max_tokens=max_tokens,
                context_window=2048,
            )
            elapsed = time.perf_counter() - started
            runs.append({
                "prompt": prompt,
                "elapsed_s": round(elapsed, 3),
                "tokens_generated": resp.tokens_generated,
                "tokens_prompt": resp.tokens_prompt,
                "tokens_per_second": resp.tokens_per_second,
                "parseable_structured_output": _is_parseable_structured_output(resp.text)
                if "类型|新鲜度|内容" in prompt else None,
                "sample": resp.text.strip()[:300],
                "error": "",
            })
        except Exception as exc:
            elapsed = time.perf_counter() - started
            error = str(exc)
            runs.append({
                "prompt": prompt,
                "elapsed_s": round(elapsed, 3),
                "tokens_generated": 0,
                "tokens_prompt": 0,
                "tokens_per_second": 0,
                "parseable_structured_output": None,
                "sample": "",
                "error": error,
            })
    return _summarize(model, runs)


def _summarize(model: str, runs: list[dict]) -> dict:
    successful = [r for r in runs if not r["error"]]
    speeds = [r["tokens_per_second"] for r in successful if r["tokens_per_second"] > 0]
    elapsed = [r["elapsed_s"] for r in successful]
    max_elapsed = max(elapsed) if elapsed else 0
    avg_speed = statistics.mean(speeds) if speeds else 0
    structured = [r["parseable_structured_output"] for r in runs if r["parseable_structured_output"] is not None]
    structured_ok = sum(1 for v in structured if v)
    grade = _grade(runs, avg_speed, max_elapsed, structured_ok, len(structured))
    return {
        "model": model,
        "grade": grade,
        "avg_tokens_per_second": round(avg_speed, 2),
        "max_elapsed_s": round(max_elapsed, 2),
        "successes": len(successful),
        "runs": runs,
        "recommendation": _recommendation(grade),
    }


def _grade(runs: list[dict], avg_speed: float, max_elapsed: float,
           structured_ok: int, structured_total: int) -> str:
    if any(r["error"] for r in runs):
        return "D"
    if max_elapsed > 90 or avg_speed < 1.5:
        return "D"
    if structured_total and structured_ok < max(1, structured_total):
        return "C"
    if avg_speed >= 4 and max_elapsed <= 45:
        return "A"
    return "B"


def _recommendation(grade: str) -> str:
    return {
        "A": "recommended",
        "B": "optional",
        "C": "smoke-test only",
        "D": "not recommended",
    }[grade]


def _is_parseable_structured_output(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    return sum(1 for line in lines if line.count("|") >= 2) >= 1


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Ollama models on this machine")
    parser.add_argument("--config", type=Path, help="Config file to read models from")
    parser.add_argument("--models", nargs="*", default=[], help="Explicit model names")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("docs/model_benchmarks.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("docs/model_benchmarks.md"))
    parser.add_argument("--host", default="http://localhost:11434")
    args = parser.parse_args(argv)

    models = list(args.models)
    if args.config:
        models.extend(load_models_from_config(args.config))
    models = _dedupe(models)
    if not models:
        print("No models requested. Use --config or --models.", file=sys.stderr)
        return 2

    client = OllamaClient(base_url=args.host, default_timeout=600, keep_alive="5m")
    results = [benchmark_model(client, model, args.rounds, args.max_tokens) for model in models]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_format_markdown(results), encoding="utf-8")
    for result in results:
        print(
            f"{result['model']}: grade={result['grade']} "
            f"avg_tps={result['avg_tokens_per_second']} "
            f"recommendation={result['recommendation']}"
        )
    print(f"Wrote benchmark report: {args.output}")
    print(f"Wrote benchmark markdown: {args.markdown_output}")
    return 0


def _format_markdown(results: list[dict]) -> str:
    lines = [
        "# Model Benchmarks",
        "",
        "This file is generated by `scripts/benchmark_models.py`.",
        "",
        "| Model | Grade | Avg tok/s | Max elapsed | Recommendation |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['model']} | {result['grade']} | "
            f"{result['avg_tokens_per_second']} | {result['max_elapsed_s']}s | "
            f"{result['recommendation']} |"
        )
    lines.append("")
    for result in results:
        lines.append(f"## {result['model']}")
        lines.append("")
        for run in result["runs"]:
            status = "ok" if not run["error"] else f"error: {run['error']}"
            sample = run["sample"].replace("\n", " ")[:200]
            lines.append(f"- {run['prompt']} -> {status}, {run['tokens_per_second']} tok/s, sample: {sample}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
