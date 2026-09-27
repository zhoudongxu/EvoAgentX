"""Freeze small benchmark cohorts from the reference protocol's pinned Parquet files."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path

from evoagentx.compactflow.benchmarks import import_dataset, write_tasks
from evoagentx.compactflow.replay import digest


def mbpp_interface(code: str, tests: list[str]) -> list[str]:
    """Expose required function headers only, never bodies or expected answers."""
    called = {
        node.func.id
        for test in tests
        for node in ast.walk(ast.parse(test))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    return [
        f"def {node.name}({ast.unparse(node.args)}):"
        for node in ast.parse(code).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in called
    ]


def select_tasks(tasks, count: int, seed: int, *, stratified: bool):
    if count < 1 or len(tasks) < count:
        raise ValueError("requested count exceeds available tasks")
    key = lambda t: digest([seed, t.benchmark, t.task_id])
    if not stratified:
        return sorted(tasks, key=key)[:count]
    subjects = collections.defaultdict(list)
    for task in tasks:
        subjects[task.metadata["subject"]].append(task)
    names = sorted(subjects, key=lambda name: digest([seed, name]))
    for name in names:
        subjects[name].sort(key=key)
    result = []
    while len(result) < count:
        for name in names:
            if subjects[name] and len(result) < count:
                result.append(subjects[name].pop(0))
    return result


def main():
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("manifest output already exists; preserve the frozen cohort")
    args.output.mkdir(parents=True)
    config = json.loads(args.config.read_text())
    sources = json.loads((args.source_dir / "sources.json").read_text())
    all_tasks, manifests = [], []
    for dataset in config["datasets"]:
        name = dataset["name"]
        entries = [row for row in sources if row["benchmark"] == name]
        if not entries:
            continue
        pool = []
        for entry in entries:
            relative = entry["repository_relative_path"]
            path = args.source_dir / name / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("source dataset checksum changed")
            if (
                f"/{dataset['repository']}/resolve/{dataset['revision']}/"
                not in entry["url"]
            ):
                raise ValueError("source dataset revision differs from protocol")
            raw = pq.read_table(path).to_pylist()
            tasks = import_dataset(
                name,
                raw,
                revision=dataset["revision"],
                original_split=dataset["original_splits"][0],
            )
            for task, original in zip(tasks, raw, strict=True):
                task.split = "target"
                task.metadata.update(
                    subject=original.get("type", ""), level=original.get("level", "")
                )
                if name == "MBPP":
                    interface = mbpp_interface(original["code"], original["test_list"])
                    if not interface:
                        raise ValueError(
                            f"cannot determine MBPP interface for {task.task_id}"
                        )
                    task.metadata["original_question_sha256"] = digest(task.question)
                    task.metadata["interface_protocol"] = (
                        "reference_function_headers_only_v1"
                    )
                    task.question += "\nRequired function interface(s):\n" + "\n".join(
                        interface
                    )
            pool.extend(tasks)
        selected = select_tasks(pool, args.count, args.seed, stratified=name == "MATH")
        all_tasks.extend(selected)
        manifests.append(
            {
                "benchmark": name,
                "pool_count": len(pool),
                "selected_count": len(selected),
                "task_ids": [t.task_id for t in selected],
                "revision": dataset["revision"],
                "selection": "subject round-robin, hash-ranked task IDs"
                if name == "MATH"
                else "hash-ranked task IDs",
                "source_files": [
                    {
                        k: entry[k]
                        for k in ("repository_relative_path", "url", "sha256", "bytes")
                    }
                    for entry in entries
                ],
            }
        )
    write_tasks(args.output / "tasks.private.jsonl", all_tasks)
    manifest = {
        "sample_seed": args.seed,
        "generation_seed": 42,
        "count_per_benchmark": args.count,
        "scope": "development trial; no policy discovery or target-time learning; no paper replication claim",
        "mbpp_input_protocol": "prompt plus required reference function headers; no solution bodies, tests or expected outputs sent to the model",
        "datasets": manifests,
        "deferred": {"GAIA": "user deferred after official data returned HTTP 401"},
        "tasks_sha256": hashlib.sha256(
            (args.output / "tasks.private.jsonl").read_bytes()
        ).hexdigest(),
    }
    (args.output / "sample_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "total_tasks": len(all_tasks),
                "datasets": {m["benchmark"]: m["selected_count"] for m in manifests},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
