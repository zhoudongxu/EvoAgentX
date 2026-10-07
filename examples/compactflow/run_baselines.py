"""Run matched construction baselines against an existing frozen evolution manifest."""

from __future__ import annotations
import argparse, asyncio, json, urllib.request
from pathlib import Path
from evoagentx.compactflow.baselines import (
    AFlowAdapter,
    A2FlowAdapter,
    EvoAgentXAdapter,
    ConstructionBaselineRunner,
    REQUIRED_METHODS,
    BaselineUnavailable,
    verify_manifest,
)
from evoagentx.compactflow.baseline_controls import PolicyBaselineAdapter
from evoagentx.compactflow.evolution import atomic_json
from evoagentx.compactflow.reproduction_config import load_reproduction_config
from examples.compactflow.run_evolution import load_raw_bundle

ROOT = Path(__file__).resolve().parents[2]


def build_adapters(names, evolution_dir):
    adapters = []
    for name in names:
        if name == "a2flow":
            adapters.append(A2FlowAdapter())
        elif name == "aflow":
            adapters.append(AFlowAdapter())
        elif name == "evoagentx":
            adapters.append(EvoAgentXAdapter())
        elif name in REQUIRED_METHODS:
            adapters.append(PolicyBaselineAdapter(name, evolution_dir=evolution_dir))
        else:
            raise ValueError(f"unsupported baseline: {name}")
    return adapters


def check_evolution_protocol(config, evolution_dir):
    path = Path(evolution_dir) / "config.lock.json"
    locked = json.loads(path.read_text())["protocol"]
    for field in (
        "model",
        "encoder",
        "datasets",
        "partition",
        "evaluation",
        "construction",
        "execution",
        "tools",
    ):
        if locked[field] != config[field]:
            raise ValueError(f"frozen evolution protocol differs: {field}")
    if not (Path(evolution_dir) / "policy_snapshots/final.json").exists():
        raise ValueError("evolution has not frozen its final policy library")


async def run(args):
    config = load_reproduction_config(args.config, root=ROOT)
    check_evolution_protocol(config, args.evolution_dir)
    if args.benchmarks:
        wanted = args.benchmarks.split(",")
        if set(wanted) - {"MBPP", "HotpotQA", "MATH", "GAIA"}:
            raise ValueError("unknown benchmark")
        config["baseline_runner"]["benchmarks"] = wanted
    manifest = args.evolution_dir / "sample_manifest.jsonl"
    bundle = load_raw_bundle(args.data_dir, config, frozen_manifest=manifest)
    verify_manifest(bundle.tasks, manifest)
    adapters = build_adapters(args.methods.split(","), args.evolution_dir)
    if args.command == "preflight":
        result = {
            "status": "ready",
            "manifest_tasks": len(bundle.tasks),
            "datasets": bundle.coverage,
            "methods": {},
        }
        for adapter in adapters:
            try:
                adapter.preflight(config)
                result["methods"][adapter.method] = {"status": "ready"}
            except (BaselineUnavailable, ImportError) as e:
                result["methods"][adapter.method] = {
                    "status": "incomplete",
                    "reason": str(e),
                }
        try:
            with urllib.request.urlopen(
                config["model"]["base_url"] + "/models", timeout=5
            ) as r:
                models = json.load(r)
            result["model"] = {
                "status": "ready"
                if config["model"]["name"] in {m["id"] for m in models["data"]}
                else "incomplete"
            }
        except Exception as e:
            result["model"] = {"status": "incomplete", "reason": str(e)}
        if any(
            v["status"] not in {"ready", "complete"}
            for v in [
                *result["datasets"].values(),
                *result["methods"].values(),
                result["model"],
            ]
        ):
            result["status"] = "incomplete"
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output / "preflight.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "ready" else 2
    runner = ConstructionBaselineRunner(
        bundle.tasks,
        output=args.output,
        adapters=adapters,
        config=config,
        manifest=manifest,
        evolution_dir=args.evolution_dir,
        dataset_coverage=bundle.coverage,
    )
    summary = await runner.run(resume=args.resume, stage=args.stage)
    print(
        json.dumps(
            {
                "publication_status": summary.get("publication_status", "incomplete"),
                "records": summary.get("records", 0),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary.get("publication_status") == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "preflight"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--evolution-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(REQUIRED_METHODS))
    parser.add_argument(
        "--benchmarks", help="Explicit pilot subset; missing benchmarks stay incomplete"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=["all", "freeze", "target"], default="all")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
