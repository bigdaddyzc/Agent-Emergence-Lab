"""Human calibration for the automatic scores.

1. Export a blind sample of utterances to label::

       python scripts/human_calibration.py export logs/experiments/latest.json \
           --out logs/calibration/labels.csv --n 30 --seed 0

   The CSV has one row per utterance with an empty ``human_score`` column (1-5:
   how novel, responsive and non-repetitive is this utterance?). The mode is
   *not* in the CSV (it is kept in ``labels.key.json``) so labels stay blind.

2. Fill in ``human_score`` (spreadsheet or editor), optionally ``human_score_2``
   for a second rater.

3. Report agreement::

       python scripts/human_calibration.py report logs/calibration/labels.csv

   Prints Spearman correlation between the human score and each automatic
   score (``auto_score`` = per-utterance overall score, ``judge_score`` when the
   LLM judge was run), and inter-rater agreement when two raters labelled.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import instruction_corpus_from_config  # noqa: E402
from src.evaluator import (  # noqa: E402
    EmergenceSignalEvaluator,
    build_reference_ngrams,
    overall_score,
    repetition_score,
)
from src.experiment import load_turns  # noqa: E402

FIELDS = ["sample_id", "session_id", "turn", "agent", "text", "auto_score",
          "repetition", "judge_score", "human_score", "human_score_2", "notes"]


def utterance_rows(manifest: list[dict], emergence_config: dict | None = None) -> list[dict]:
    rows = []
    for item in manifest:
        path = Path(item["jsonl_path"])
        if not path.exists():
            continue
        start, turns = load_turns(path)
        refs = build_reference_ngrams(instruction_corpus_from_config(start.get("config_snapshot") or {}))
        evaluator = EmergenceSignalEvaluator(emergence_config or
                                             (start.get("config_snapshot") or {}).get("emergence"))
        previous = ""
        for idx, turn in enumerate(turns):
            text = turn.get("raw_content") or turn.get("content", "")
            signals, _ = evaluator.evaluate_clean(text, refs)
            rep = repetition_score(text, [previous] if previous else [])
            rows.append({
                "session_id": item.get("session_id", path.stem),
                "mode": item.get("mode", ""),
                "turn": idx,
                "agent": turn.get("agent_name", ""),
                "text": text,
                "auto_score": overall_score(sum(signals.as_dict().values()), rep),
                "repetition": rep,
                "judge_score": "",
            })
            previous = text
    return rows


def export(manifest_path: Path, out: Path, n: int, seed: int) -> Path:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = utterance_rows(manifest)
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    out.parent.mkdir(parents=True, exist_ok=True)
    key = {}
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for i, row in enumerate(sample):
            sample_id = f"s{i:03d}"
            key[sample_id] = {"mode": row["mode"], "session_id": row["session_id"]}
            writer.writerow({**{k: row.get(k, "") for k in FIELDS}, "sample_id": sample_id,
                             "human_score": "", "human_score_2": "", "notes": ""})
    out.with_suffix(".key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return out


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return None if va == 0 or vb == 0 else round(cov / (va * vb), 3)


def spearman(a: list[float], b: list[float]) -> float | None:
    return pearson(_ranks(a), _ranks(b))


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def report(labels_path: Path) -> dict:
    with open(labels_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    labelled = [r for r in rows if _float(r.get("human_score")) is not None]
    result: dict = {"labelled": len(labelled), "total": len(rows)}
    for column in ("auto_score", "judge_score"):
        pairs = [(_float(r["human_score"]), _float(r.get(column))) for r in labelled]
        pairs = [(h, a) for h, a in pairs if a is not None]
        result[f"spearman_human_vs_{column}"] = spearman([h for h, _ in pairs], [a for _, a in pairs])
        result[f"n_{column}"] = len(pairs)
    second = [(_float(r["human_score"]), _float(r.get("human_score_2"))) for r in labelled]
    second = [(a, b) for a, b in second if b is not None]
    if second:
        result["inter_rater_n"] = len(second)
        result["inter_rater_spearman"] = spearman([a for a, _ in second], [b for _, b in second])
        result["inter_rater_within_1"] = round(
            sum(1 for a, b in second if abs(a - b) <= 1) / len(second), 3)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    exp = sub.add_parser("export", help="export a blind sample of utterances to label")
    exp.add_argument("manifest", type=Path)
    exp.add_argument("--out", type=Path, default=Path("logs/calibration/labels.csv"))
    exp.add_argument("--n", type=int, default=30)
    exp.add_argument("--seed", type=int, default=0)
    rep = sub.add_parser("report", help="report agreement between human and automatic scores")
    rep.add_argument("labels", type=Path)
    args = parser.parse_args(argv)
    if args.cmd == "export":
        path = export(args.manifest, args.out, args.n, args.seed)
        print(f"Wrote {path} (fill in human_score 1-5) and {path.with_suffix('.key.json')}")
        return 0
    result = report(args.labels)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["labelled"] < 20:
        print("note: label at least ~20-30 utterances before trusting these correlations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
