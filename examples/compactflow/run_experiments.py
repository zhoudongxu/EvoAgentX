"""CompactFlow experiment entry point.

The default smoke experiment is offline and deterministic apart from normal
operating-system timer jitter.  It validates construction-policy retrieval,
complete-dependency execution, exact guarded execution, paired metrics, and
anonymous artifact generation without calling an LLM or downloading data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from evoagentx.compactflow.compiler import GFRGCompiler
from evoagentx.compactflow.construction import (
    ConstructionConfig,
    ConstructionPlane,
)
from evoagentx.compactflow.experiments import (
    ArtifactWriter,
    ConfigValidationError,
    build_anonymous_manifest,
    load_experiment_config,
    write_experiment_artifacts,
)
from evoagentx.compactflow.metrics import (
    RunRecord,
    RuntimeRecord,
    StructureRecord,
    ViolationCounts,
    aggregate_runtime_diagnostics,
    compare_paired_methods,
)
from evoagentx.compactflow.policy import (
    PolicyLibrary,
    RetrievalConfig,
    SelectionConfig,
)
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.schema import (
    GFRG,
    CallSpec,
    Complete,
    DataDependency,
    ExecutionMode,
    Partial,
    ResourceVector,
    StreamContract,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_SMOKE_CONFIG = ROOT / "configs" / "smoke.json"
DEFAULT_POLICY_LIBRARY = ROOT / "policies" / "seed_policies.json"

STRING = {"type": "string"}
INTEGER = {"type": "integer"}


def object_schema(
    properties: dict[str, dict[str, Any]],
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def streaming_fanout_graph(capacity: int) -> GFRG:
    async def retrieve():
        yield Partial({"paper_1": "A"}, stable_fields=("paper_1",))
        await asyncio.sleep(0.04)
        yield Partial({"paper_2": "B"}, stable_fields=("paper_2",))
        await asyncio.sleep(0.04)
        yield Complete({"paper_1": "A", "paper_2": "B"})

    async def summarize_first(paper: str):
        await asyncio.sleep(0.05)
        return {"summary_1": f"summary:{paper}"}

    async def summarize_second(paper: str):
        await asyncio.sleep(0.05)
        return {"summary_2": f"summary:{paper}"}

    async def aggregate(summary_1: str, summary_2: str):
        await asyncio.sleep(0.01)
        return {"report": f"{summary_1}|{summary_2}"}

    external = ResourceVector(external_call=1)
    calls = [
        CallSpec(
            "retrieve",
            retrieve,
            output_schema=object_schema(
                {"paper_1": STRING, "paper_2": STRING},
                ("paper_1", "paper_2"),
            ),
            resources=external,
            stream_contract=StreamContract(
                stable_fields=("paper_1", "paper_2")
            ),
        ),
        CallSpec(
            "summarize_1",
            summarize_first,
            input_schema=object_schema({"paper": STRING}, ("paper",)),
            output_schema=object_schema(
                {"summary_1": STRING}, ("summary_1",)
            ),
            resources=external,
            early_safe=True,
        ),
        CallSpec(
            "summarize_2",
            summarize_second,
            input_schema=object_schema({"paper": STRING}, ("paper",)),
            output_schema=object_schema(
                {"summary_2": STRING}, ("summary_2",)
            ),
            resources=external,
            early_safe=True,
        ),
        CallSpec(
            "aggregate",
            aggregate,
            input_schema=object_schema(
                {"summary_1": STRING, "summary_2": STRING},
                ("summary_1", "summary_2"),
            ),
            output_schema=object_schema({"report": STRING}, ("report",)),
            resources=external,
            early_safe=True,
        ),
    ]
    dependencies = [
        DataDependency(
            "retrieve", "summarize_1", "paper_1", "paper"
        ),
        DataDependency(
            "retrieve", "summarize_2", "paper_2", "paper"
        ),
        DataDependency(
            "summarize_1", "aggregate", "summary_1", "summary_1"
        ),
        DataDependency(
            "summarize_2", "aggregate", "summary_2", "summary_2"
        ),
    ]
    return GFRGCompiler(
        ResourceVector(external_call=float(capacity))
    ).compile(calls, dependencies)


def full_result_fallback_graph(capacity: int) -> GFRG:
    async def draft():
        yield Partial({"draft": 1})
        await asyncio.sleep(0.04)
        yield Complete({"draft": 2})

    async def consume(draft: int):
        await asyncio.sleep(0.01)
        return {"seen": draft}

    external = ResourceVector(external_call=1)
    calls = [
        CallSpec(
            "draft",
            draft,
            output_schema=object_schema({"draft": INTEGER}, ("draft",)),
            resources=external,
            stream_contract=StreamContract(mutable_fields=("draft",)),
        ),
        CallSpec(
            "consume",
            consume,
            input_schema=object_schema({"draft": INTEGER}, ("draft",)),
            output_schema=object_schema({"seen": INTEGER}, ("seen",)),
            resources=external,
            early_safe=True,
        ),
    ]
    return GFRGCompiler(
        ResourceVector(external_call=float(capacity))
    ).compile(
        calls,
        [DataDependency("draft", "consume", "draft", "draft")],
    )


def _structure(graph: GFRG) -> StructureRecord:
    edges = {
        (dependency.producer, dependency.consumer)
        for dependency in graph.data_dependencies
        if dependency.producer is not None
    }
    edges.update(
        (dependency.producer, dependency.consumer)
        for dependency in graph.effect_dependencies
    )
    return StructureRecord.from_edges(
        [call.id for call in graph.calls], edges
    )


def _runtime_record(graph: GFRG, execution) -> RuntimeRecord:
    readiness_gaps: list[float] = []
    guard_ready_times: list[float] = []
    early_calls: set[str] = set()
    for guard in graph.guards:
        producer_trace = execution.call_traces[guard.producer]
        consumer_trace = execution.call_traces[guard.consumer]
        ready_at = next(
            (
                event.timestamp
                for event in execution.trace
                if event.call_id == guard.producer
                and event.kind.value == "partial"
                and guard.source_path
                in event.detail.get("stable_fields", ())
            ),
            None,
        )
        if ready_at is not None and producer_trace.ended_at is not None:
            guard_ready_times.append(ready_at)
            readiness_gaps.append(
                max(0.0, producer_trace.ended_at - ready_at)
            )
        if (
            consumer_trace.started_at is not None
            and producer_trace.ended_at is not None
            and consumer_trace.started_at < producer_trace.ended_at
        ):
            early_calls.add(guard.consumer)
    violations = execution.metrics.violations
    return RuntimeRecord(
        latency_seconds=float(execution.metrics.latency or 0.0),
        ttfo_seconds=execution.metrics.ttfo,
        analyzed_data_edges=sum(
            dependency.producer is not None
            for dependency in graph.data_dependencies
        ),
        partial_edges=len(graph.guards),
        task_has_opportunity=any(gap > 0 for gap in readiness_gaps),
        readiness_gaps=tuple(readiness_gaps),
        first_guard_ready_item_seconds=(
            None
            if not guard_ready_times
            else min(guard_ready_times) - execution.metrics.started_at
        ),
        dispatched_calls=sum(
            trace.started_at is not None
            for trace in execution.call_traces.values()
        ),
        early_dispatched_calls=len(early_calls),
        violations=ViolationCounts(
            argument_mismatch=violations.argument_mismatch,
            duplicate_dispatch=violations.duplicate_dispatch,
            effect_order=violations.effect_order,
            capacity_overload=violations.resource_capacity,
        ),
    )


def _quality(task_id: str, execution) -> float:
    if task_id == "streaming-fanout":
        return float(
            execution.outputs["aggregate"].get("report")
            == "summary:A|summary:B"
        )
    return float(execution.outputs["consume"].get("seen") == 2)


def _normalized_trace(execution) -> list[dict[str, Any]]:
    origin = execution.metrics.started_at
    return [
        {
            "sequence": event.sequence,
            "time_seconds": event.timestamp - origin,
            "kind": event.kind.value,
            "call_id": event.call_id,
            "detail": dict(event.detail),
        }
        for event in execution.trace
    ]


async def run_execution_smoke(config: dict[str, Any]):
    capacity = int(config["execution"]["external_call_capacity"])
    seed = config["seed"]
    records: list[RunRecord] = []
    traces: dict[str, Any] = {}
    graph_factories = {
        "streaming-fanout": streaming_fanout_graph,
        "full-result-fallback": full_result_fallback_graph,
    }
    mode_by_method = {
        "complete_dependency": ExecutionMode.COMPLETE,
        "guarded": ExecutionMode.GUARDED,
    }
    for task_id, factory in graph_factories.items():
        for method in config["methods"]["execution"]:
            if method not in mode_by_method:
                raise ValueError(
                    f"offline smoke does not implement execution method "
                    f"{method!r}"
                )
            graph = factory(capacity)
            execution = await CompactFlowRuntime(
                graph, mode=mode_by_method[method]
            ).execute()
            records.append(
                RunRecord(
                    benchmark="synthetic-compactflow",
                    task_id=task_id,
                    seed=seed,
                    method=method,
                    quality=_quality(task_id, execution),
                    valid=not execution.errors,
                    structure=_structure(graph),
                    runtime=_runtime_record(graph, execution),
                )
            )
            traces[f"{task_id}:{method}"] = _normalized_trace(execution)
    return records, traces


def run_construction_smoke(config: dict[str, Any]) -> dict[str, Any]:
    library = PolicyLibrary.load(DEFAULT_POLICY_LIBRARY)
    construction = config["construction"]
    weights = construction["retrieval_weights"]
    confidence_weight = max(
        0.0,
        1.0
        - float(weights["semantic"])
        - float(weights["structural"])
        - float(weights["historical"]),
    )
    plane = ConstructionPlane(
        library,
        config=ConstructionConfig(
            retrieval=RetrievalConfig(
                semantic_top_k0=int(construction["semantic_top_k0"]),
                top_k=int(construction["top_k"]),
                semantic_weight=float(weights["semantic"]),
                applicability_weight=float(weights["structural"]),
                utility_weight=float(weights["historical"]),
                confidence_weight=confidence_weight,
                include_candidates=bool(
                    construction["include_candidate_policies"]
                ),
            ),
            selection=SelectionConfig(
                max_policies=int(construction["max_policies"]),
                minimum_score=-1.0,
                minimum_applicability=0.0,
            ),
        ),
    )
    guidance = plane.prepare(
        "compact streaming workflow with repeated roles and unused outputs",
        {
            "capabilities": {
                "typed_workflow": True,
                "explicit_streaming": True,
            }
        },
    )
    return {
        "library_size": len(library),
        "candidate_ids": [
            match.policy.id for match in guidance.candidates
        ],
        "selected_ids": [
            policy.id for policy in guidance.selected_policies
        ],
        "skipped": [
            {
                "policy_id": item.policy_id,
                "reason": item.reason,
                "conflicts_with": item.conflicts_with,
            }
            for item in guidance.selection.skipped
        ],
        "scope": (
            "retrieval and compatibility smoke only; no LLM planning or "
            "benchmark quality claim"
        ),
    }


async def smoke(config_path: Path, output: Path | None) -> int:
    config = load_experiment_config(config_path, for_run=True)
    records, traces = await run_execution_smoke(config)
    paired = compare_paired_methods(
        records,
        baseline_method="complete_dependency",
        candidate_method="guarded",
    )
    guarded_records = [
        record for record in records if record.method == "guarded"
    ]
    summary = {
        "construction": run_construction_smoke(config),
        "execution": paired.to_dict(),
        "guarded_diagnostics": aggregate_runtime_diagnostics(
            guarded_records
        ).to_dict(),
        "interpretation": (
            "Offline implementation smoke only. Timings are local synthetic "
            "measurements and are not paper reproduction results."
        ),
    }
    dataset_ids = {
        dataset["name"]: dataset.get("task_ids", ())
        for dataset in config["datasets"]
    }
    manifest = build_anonymous_manifest(
        config,
        dataset_ids=dataset_ids,
        extra={
            "experiment_kind": "offline_implementation_smoke",
            "paper_placeholder_results_used": False,
        },
    )
    destination = output or Path(config["artifacts"]["directory"])
    writer = ArtifactWriter(destination)
    paths = write_experiment_artifacts(
        writer,
        records=records,
        summary=summary,
        manifest=manifest,
    )
    paths["traces"] = writer.write_json("traces.json", traces)
    print(
        json.dumps(
            {
                "status": "ok",
                "artifacts": {
                    name: str(path) for name, path in paths.items()
                },
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def validate_config(config_path: Path, for_run: bool) -> int:
    config = load_experiment_config(config_path, for_run=for_run)
    print(
        json.dumps(
            {
                "status": "valid",
                "name": config["name"],
                "runner": config["runner"],
                "template": config["template"],
                "runnable": config["runnable"],
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or validate CompactFlow experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser(
        "smoke", help="run the offline construction/execution smoke"
    )
    smoke_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SMOKE_CONFIG,
    )
    smoke_parser.add_argument("--output", type=Path)
    validate_parser = subparsers.add_parser(
        "validate-config", help="validate a smoke or paper configuration"
    )
    validate_parser.add_argument("--config", type=Path, required=True)
    validate_parser.add_argument(
        "--for-run",
        action="store_true",
        help="also require that every runnable field is resolved",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "smoke":
            return asyncio.run(smoke(args.config, args.output))
        return validate_config(args.config, args.for_run)
    except ConfigValidationError as error:
        print(
            json.dumps(
                {"status": "error", "error": str(error)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
