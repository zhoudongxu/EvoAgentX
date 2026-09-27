"""Run frozen small cohorts through live planning/execution and paired call replay."""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

from evoagentx.compactflow.benchmarks import evaluate, read_tasks
from evoagentx.compactflow.llm import ModelClient
from evoagentx.compactflow.paper_workflow import compile_spec, plan_workflow
from evoagentx.compactflow.replay import (
    ReplayBundle,
    digest,
    graph_from_replay,
    graph_to_dict,
)
from evoagentx.compactflow.reproduction_config import load_reproduction_config
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.safety import CATEGORIES, audit_execution


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def execution_record(graph, result, reference_arguments=None):
    origin = result.metrics.started_at
    starts = collections.Counter(
        e.call_id for e in result.trace if e.kind.value == "start"
    )
    return {
        "outputs": result.outputs,
        "arguments": result.arguments,
        "errors": result.errors,
        "dispatch_audit": {
            "actual_start_events": sum(starts.values()),
            "duplicate_logical_calls": sum(n > 1 for n in starts.values()),
            "extra_start_events": sum(max(0, n - 1) for n in starts.values()),
            "refused_dispatch_attempts": result.metrics.violations.duplicate_dispatch,
        },
        "states": {key: state.value for key, state in result.states.items()},
        "latency_seconds": result.metrics.latency,
        "ttfo_seconds": result.metrics.ttfo,
        "drain_seconds": result.metrics.ended_at - origin,
        "contract_safety": audit_execution(
            graph, result, reference_arguments=reference_arguments
        ),
        "trace": [
            {
                "sequence": e.sequence,
                "time": e.timestamp - origin,
                "kind": e.kind.value,
                "call_id": e.call_id,
                "detail": dict(e.detail),
            }
            for e in result.trace
        ],
    }


async def run_task(task, config, protocol, directory):
    directory.mkdir()
    client = ModelClient(
        config["model"], token_budget=config["evaluation"]["task_token_budget"]
    )
    report = {
        "benchmark": task.benchmark,
        "task_id": task.task_id,
        "family_id": task.family_id,
        "seed": protocol["generation_seed"],
        "status": "started",
        "stage": "planner",
        "quality": 0.0,
        "answer": "",
        "replays": {},
    }
    start = time.perf_counter()
    bundle = ReplayBundle(
        f"{task.benchmark}:{task.task_id}:{protocol['generation_seed']}"
    )
    try:
        planning_start = time.perf_counter()
        spec, planning = await plan_workflow(
            client,
            task,
            [],
            config["construction"]["planner"],
            seed=protocol["generation_seed"],
        )
        report.update(
            planning_seconds=time.perf_counter() - planning_start,
            planning=planning,
            nodes=len(spec["nodes"]),
        )
        save(directory / "workflow.json", spec)
        graph = compile_spec(
            spec,
            task.public_input(),
            client,
            seed=protocol["generation_seed"],
            capacity=config["execution"]["external_call_capacity"],
            recorder=bundle,
        )
        report["edges"] = sum(d.producer is not None for d in graph.data_dependencies)
        options = {
            "sink_ids": spec["sinks"],
            "call_timeout": config["execution"]["call_timeout_seconds"],
            "workflow_timeout": config["execution"]["workflow_timeout_seconds"],
        }
        report["stage"] = "live_execution"
        live = await CompactFlowRuntime(graph, mode="complete", **options).execute(
            task.public_input()
        )
        live_record = execution_record(graph, live)
        save(directory / "live.json", live_record)
        bundle.save(directory / "replay.json")
        save(directory / "graph.json", graph_to_dict(graph, spec["sinks"]))
        report.update(
            live_latency_seconds=live.metrics.latency,
            live_ttfo_seconds=live.metrics.ttfo,
            live_errors=live.errors,
            live_dispatch_audit=live_record["dispatch_audit"],
            answer=live.outputs.get(spec["sinks"][0], {}).get("answer", ""),
        )
        if live.errors:
            report["status"] = "execution_failed"
            return report
        report["stage"] = "evaluation"
        score = await asyncio.to_thread(
            evaluate, task, report["answer"], sandbox=config["tools"]["mbpp_sandbox"]
        )
        report.update(quality=score["quality"], evaluation=score)
        report["stage"] = "replay"
        modes = list(protocol["replay_methods"])
        random.Random(
            int(
                digest([protocol["runtime_order_seed"], task.benchmark, task.task_id]),
                16,
            )
        ).shuffle(modes)
        report["replay_order"] = modes
        for mode in modes:
            replay_graph = graph_from_replay(
                graph_to_dict(graph),
                ReplayBundle.load(directory / "replay.json"),
                time_scale=protocol["replay_time_scale"],
                batch_size=config["execution"]["materialization_batch_size"],
            )
            result = await CompactFlowRuntime(
                replay_graph, mode=mode, **options
            ).execute(task.public_input())
            record = execution_record(replay_graph, result, live.arguments)
            save(directory / f"replay-{mode}.json", record)
            report["replays"][mode] = {
                key: record[key]
                for key in (
                    "errors",
                    "latency_seconds",
                    "ttfo_seconds",
                    "contract_safety",
                    "dispatch_audit",
                )
            }
            report["replays"][mode]["outputs_equal_live"] = (
                result.outputs == live.outputs
            )
        report["status"] = (
            "completed"
            if all(
                not r["errors"] and r["outputs_equal_live"]
                for r in report["replays"].values()
            )
            else "replay_failed"
        )
        report["stage"] = "finished"
    except Exception as error:  # noqa: BLE001 - retain every attempted task and its failure
        report["status"] = (
            "infrastructure_failed"
            if report["stage"] == "evaluation"
            else "pipeline_failed"
        )
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        bundle.save(directory / "replay.json")
        report["wall_seconds"] = time.perf_counter() - start
        if report["stage"] == "planner" and "planning_seconds" not in report:
            report["planning_seconds"] = time.perf_counter() - planning_start
        report["tokens"] = client.accounting()
        save(directory / "model_requests.json", client.records)
        save(directory / "result.json", report)
    return report


def summarize(records, protocol):
    summary = {
        "scope": protocol["scope"],
        "generation_seed": protocol["generation_seed"],
        "replay_repetitions": 1,
        "warmups": 0,
        "latency_claim": "single-repetition development diagnostic only",
        "datasets": {},
        "GAIA": {
            "status": "deferred_by_user",
            "reason": "official data access requires authorization",
        },
    }
    for benchmark in sorted({r["benchmark"] for r in records}):
        items = [r for r in records if r["benchmark"] == benchmark]
        pairs = [
            r
            for r in items
            if set(r["replays"]) == set(protocol["replay_methods"])
            and all(
                not v["errors"] and v["outputs_equal_live"]
                for v in r["replays"].values()
            )
        ]
        safety = [
            r["replays"]["guarded"]["contract_safety"]
            for r in items
            if "guarded" in r["replays"]
        ]
        counts = {key: sum(r["counts"][key] for r in safety) for key in CATEGORIES}
        early = sum(r["denominator"] for r in safety)
        unassessed = sum(r["unassessed_early_calls"] for r in safety)
        complete_mean = (
            statistics.mean(r["replays"]["complete"]["latency_seconds"] for r in pairs)
            if pairs
            else None
        )
        guarded_mean = (
            statistics.mean(r["replays"]["guarded"]["latency_seconds"] for r in pairs)
            if pairs
            else None
        )
        summary["datasets"][benchmark] = {
            "attempted": len(items),
            "requested": protocol["count_per_benchmark"],
            "status_counts": dict(collections.Counter(r["status"] for r in items)),
            "quality_mean_all_attempts": statistics.mean(r["quality"] for r in items),
            "quality_metric": {
                "MBPP": "pass@1",
                "HotpotQA": "answer_f1",
                "MATH": "normalized_exact_match",
            }[benchmark],
            "exact_correct": sum(
                r.get("evaluation", {}).get("em", r["quality"]) == 1 for r in items
            ),
            "tokens_observed": sum(r["tokens"]["total_tokens"] for r in items),
            "token_usage_complete": all(r["tokens"]["usage_complete"] for r in items),
            "mean_planning_seconds": statistics.mean(
                r["planning_seconds"] for r in items if "planning_seconds" in r
            )
            if any("planning_seconds" in r for r in items)
            else None,
            "planning_timing_samples": sum("planning_seconds" in r for r in items),
            "mean_live_latency_seconds": statistics.mean(
                r["live_latency_seconds"] for r in items if "live_latency_seconds" in r
            )
            if any("live_latency_seconds" in r for r in items)
            else None,
            "mean_nodes": statistics.mean(r["nodes"] for r in items if "nodes" in r)
            if any("nodes" in r for r in items)
            else None,
            "workflow_size_samples": sum("nodes" in r for r in items),
            "valid_replay_pairs": len(pairs),
            "mean_complete_replay_seconds": complete_mean,
            "mean_guarded_replay_seconds": guarded_mean,
            "ratio_of_paired_means": complete_mean / guarded_mean
            if pairs and guarded_mean
            else None,
            "guarded_dispatch_audit": {
                key: sum(
                    r["replays"]["guarded"]["dispatch_audit"][key]
                    for r in items
                    if "guarded" in r["replays"]
                )
                for key in (
                    "actual_start_events",
                    "duplicate_logical_calls",
                    "extra_start_events",
                    "refused_dispatch_attempts",
                )
            },
            "safety": {
                "counts": counts,
                "early_calls": early,
                "unassessed_early_calls": unassessed,
                "assessed_tasks": len(safety),
                "aggregate_rate": sum(counts.values()) / early
                if early and not unassessed
                else None,
            },
        }
    return summary


async def main_async(args):
    root = Path(__file__).resolve().parents[2]
    config = load_reproduction_config(args.config, root=root)
    tasks = read_tasks(args.cohort / "tasks.private.jsonl")
    manifest = json.loads((args.cohort / "sample_manifest.json").read_text())
    if (
        hashlib.sha256((args.cohort / "tasks.private.jsonl").read_bytes()).hexdigest()
        != manifest["tasks_sha256"]
    ):
        raise ValueError("frozen task manifest checksum mismatch")
    if args.output.exists():
        raise ValueError("output already exists; do not overwrite measured runs")
    args.output.mkdir(parents=True)
    protocol = {
        "scope": "base planner with no policies; live complete execution; paired complete/guarded recorded replay; policy evolution disabled",
        "generation_seed": 42,
        "runtime_order_seed": 31415,
        "count_per_benchmark": manifest["count_per_benchmark"],
        "replay_methods": ["complete", "guarded"],
        "replay_time_scale": 1.0,
        "repetitions": 1,
        "warmups": 0,
        "code_commit": (
            await asyncio.to_thread(
                subprocess.check_output,
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
            )
        ).strip(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "tasks_sha256": manifest["tasks_sha256"],
    }
    save(args.output / "config.lock.json", config)
    save(args.output / "trial_protocol.json", protocol)
    save(args.output / "sample_manifest.json", manifest)
    # Fail sandbox infrastructure before consuming model tokens.
    await asyncio.to_thread(
        subprocess.run,
        ["docker", "image", "inspect", config["tools"]["mbpp_sandbox"]["image"]],
        stdout=subprocess.DEVNULL,
        check=True,
    )
    records = []
    for index, task in enumerate(tasks):
        directory = args.output / f"{index:02d}-{task.benchmark}-{task.task_id}"
        result = await run_task(task, config, protocol, directory)
        records.append(result)
        with (args.output / "results.jsonl").open("a") as file:
            file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        save(args.output / "summary.json", summarize(records, protocol))
        print(
            json.dumps(
                {
                    "finished": len(records),
                    "total": len(tasks),
                    "benchmark": task.benchmark,
                    "task_id": task.task_id,
                    "status": result["status"],
                    "quality": result["quality"],
                    "tokens": result["tokens"]["total_tokens"],
                }
            ),
            flush=True,
        )
    print(json.dumps(summarize(records, protocol), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
