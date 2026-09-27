"""One live model/planner/replay integration check; never a benchmark result."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from evoagentx.compactflow.benchmarks import BenchmarkTask
from evoagentx.compactflow.llm import ModelClient
from evoagentx.compactflow.paper_workflow import compile_spec, plan_workflow
from evoagentx.compactflow.replay import ReplayBundle, graph_from_replay, graph_to_dict
from evoagentx.compactflow.reproduction_config import load_reproduction_config
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.safety import audit_execution


async def run(config_path: Path, output: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    config = load_reproduction_config(config_path, root=root)
    if output.exists():
        raise ValueError("choose a new output directory to preserve previous evidence")
    output.mkdir(parents=True)
    task = BenchmarkTask("integration_fixture", "arithmetic-1", "Calculate 17 + 25. Return only the integer as the final answer.", "42", "arithmetic")
    client = ModelClient(config["model"], token_budget=config["evaluation"]["task_token_budget"])
    seed = config["evaluation"]["generation_seeds"][0]
    spec, planning = await plan_workflow(client, task, [], config["construction"]["planner"], seed=seed)
    bundle = ReplayBundle(task.task_id)
    graph = compile_spec(spec, task.public_input(), client, seed=seed, capacity=config["execution"]["external_call_capacity"], recorder=bundle)
    options = {"sink_ids": spec["sinks"], "call_timeout": config["execution"]["call_timeout_seconds"],
               "workflow_timeout": config["execution"]["workflow_timeout_seconds"]}
    live = await CompactFlowRuntime(graph, mode="complete", **options).execute(task.public_input())
    bundle.save(output / "replay.json")
    replay_graph = graph_from_replay(graph_to_dict(graph, spec["sinks"]), ReplayBundle.load(output / "replay.json"))
    guarded = await CompactFlowRuntime(replay_graph, mode="guarded", **options).execute(task.public_input())
    answer = live.outputs[spec["sinks"][0]].get("answer", "")
    report = {"scope": "live Qwen planner/executor and recorded replay integration; one synthetic task, not a benchmark",
              "model": config["model"]["name"], "model_revision": config["model"]["revision"],
              "answer": answer, "correct": answer.strip() == task.answer, "planning": planning,
              "live_errors": live.errors, "replay_errors": guarded.errors,
              "replay_outputs_equal": live.outputs == guarded.outputs, "tokens": client.accounting(),
              "contract_safety": audit_execution(replay_graph, guarded, reference_arguments=live.arguments),
              "live_latency_seconds": live.metrics.latency, "guarded_replay_latency_seconds": guarded.metrics.latency}
    report["status"] = "ok" if report["correct"] and not live.errors and not guarded.errors and report["replay_outputs_equal"] and report["tokens"]["usage_complete"] else "failed"
    for name, value in (("config.lock.json", config), ("workflow.json", spec), ("summary.json", report), ("model_requests.json", client.records)):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/qwen3_coder_a100.pilot.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run(args.config, args.output))
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
