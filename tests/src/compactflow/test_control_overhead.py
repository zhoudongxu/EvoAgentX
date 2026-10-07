"""Figure 6 control measurements: disjoint scopes and durable raw evidence."""
from __future__ import annotations

import asyncio
import csv
from types import SimpleNamespace

import pytest

from evoagentx.compactflow.compiler import GFRGCompiler
from evoagentx.compactflow.paper_reporting import _control_overhead_summary, report_paper
from evoagentx.compactflow.paper_studies import replay_one
from evoagentx.compactflow.replay import ReplayBundle, graph_from_replay, graph_to_dict
from evoagentx.compactflow.runtime import (
    CONTROL_PROFILE_VERSION, CompactFlowRuntime, _ControlProfiler,
)
from evoagentx.compactflow.schema import (
    CallSpec, Complete, DataDependency, Partial, StreamContract,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_nested_categories_are_exclusive_including_same_category():
    clock = Clock()
    totals = {}
    profiler = _ControlProfiler(totals, clock=clock)
    resource = profiler.wrap(lambda: clock.advance(1), "guard")

    def check():
        clock.advance(2)
        resource()
        clock.advance(2)

    guard = profiler.wrap(check, "guard")

    def launch():
        clock.advance(3)
        guard()
        clock.advance(4)

    dispatch = profiler.wrap(launch, "dispatch")

    def queue():
        clock.advance(1)
        dispatch()
        clock.advance(2)

    def materialize():
        clock.advance(4)
        guard()
        clock.advance(1)

    profiler.wrap(queue, "dispatch")()
    profiler.wrap(materialize, "materialization")()
    assert totals == {"guard": 10, "dispatch": 10, "materialization": 5}
    assert sum(totals.values()) == clock.now == 25
    assert profiler.stack == []


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
def test_failed_nested_scope_is_accounted_and_stack_is_reset(error):
    clock = Clock()
    totals = {}
    profiler = _ControlProfiler(totals, clock=clock)

    def failure():
        clock.advance(2)
        raise error("test")

    child = profiler.wrap(failure, "guard")
    with pytest.raises(error):
        profiler.wrap(child, "dispatch")()
    profiler.wrap(lambda: clock.advance(4), "dispatch")()
    assert totals == {"guard": 2, "dispatch": 4}
    assert profiler.stack == []


def schema(**properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def test_profile_preserves_guarded_streaming_and_excludes_model_wait(monkeypatch):
    import evoagentx.compactflow.runtime as runtime_module
    clock = Clock()
    monkeypatch.setattr(runtime_module, "_ControlProfiler",
                        lambda totals: _ControlProfiler(totals, clock=clock))

    async def run(profile):
        consumer_started = asyncio.Event()

        async def producer():
            clock.advance(1000)  # Model-side time must never enter control scopes.
            yield Partial({"answer": "7"}, stable_fields=("answer",))
            await asyncio.wait_for(consumer_started.wait(), 1)
            clock.advance(1000)
            yield Complete({"answer": "7", "tail": "done"})

        async def consumer(answer):
            consumer_started.set()
            return {"answer": answer}

        graph = GFRGCompiler().compile([
            CallSpec("producer", producer, output_schema=schema(
                answer={"type": "string"}, tail={"type": "string"}),
                stream_contract=StreamContract(stable_fields=("answer",))),
            CallSpec("consumer", consumer, input_schema=schema(answer={"type": "string"}),
                     output_schema=schema(answer={"type": "string"}), early_safe=True),
        ], [DataDependency("producer", "consumer", "answer", "answer")])
        result = await CompactFlowRuntime(graph, mode="guarded",
            sink_ids=("consumer",), profile_controls=profile).execute()
        assert not result.errors
        assert result.call_traces["consumer"].started_at < result.call_traces["producer"].ended_at
        return result

    unprofiled = asyncio.run(run(False))
    measured = asyncio.run(run(True))
    assert unprofiled.outputs == measured.outputs
    assert unprofiled.metrics.control_profile_version is None
    assert unprofiled.metrics.control_seconds == {}
    assert measured.metrics.control_profile_version == CONTROL_PROFILE_VERSION
    assert measured.metrics.control_seconds == {"guard": 0, "dispatch": 0, "materialization": 0}
    assert clock.now == 4000


def test_static_analysis_excludes_replay_binding(monkeypatch):
    import evoagentx.compactflow.replay as replay_module
    clock = Clock()
    graph = GFRGCompiler().compile([CallSpec("a", lambda: {}, output_schema=schema())], [])

    class Bundle:
        def bind(self, *args, **kwargs):
            clock.advance(100)
            return lambda: {}

    class TimedCompiler(GFRGCompiler):
        def compile(self, *args, **kwargs):
            clock.advance(5)
            return super().compile(*args, **kwargs)

    monkeypatch.setattr(replay_module, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(replay_module, "GFRGCompiler", TimedCompiler)
    timings = {}
    rebuilt = graph_from_replay(graph_to_dict(graph, ["a"]), Bundle(), compilation_metrics=timings)
    assert rebuilt.topological_order == graph.topological_order
    assert timings == {"static_analysis": 5}
    assert clock.now == 105


@pytest.mark.parametrize("native", [False, True])
def test_replay_retains_four_components_without_live_calls(native, monkeypatch):
    import socket
    from test_evolution import pilot
    live_calls = []

    async def run():
        async def answer():
            live_calls.append(True)
            return {"answer": "7"}

        bundle = ReplayBundle("control-overhead-test")
        graph = GFRGCompiler().compile([bundle.wrap(CallSpec(
            "a", answer, output_schema=schema(answer={"type": "string"})))], [])
        captured = await CompactFlowRuntime(graph, sink_ids=("a",)).execute({})
        record = {"graph": graph_to_dict(graph, ["a"]),
                  "replay": {"workflow_id": bundle.workflow_id, "records": bundle.records},
                  "arguments": captured.arguments, "answer": "7", "evaluation": {"quality": 1}}
        monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("unexpected network"))
        case = {"name": "guarded", "mode": "guarded"}
        if native:
            case = {"name": "llmcompiler", "mode": "complete", "backend": "llmcompiler"}
        result = await replay_one(record, {}, pilot(), case)
        assert result["status"] == "complete" and result["outputs_equal_capture"]
        assert live_calls == [True]
        assert result["compilation_seconds"] == pytest.approx(
            result["static_analysis_seconds"] + result["graph_setup_seconds"])
        if native:
            assert result["control_overhead"] is None
            assert result["control_profile_version"] is None
        else:
            assert result["control_profile_version"] == CONTROL_PROFILE_VERSION
            assert set(result["control_overhead"]) == {"static_analysis", "materialization", "guard", "dispatch"}
            assert all(v >= 0 for v in result["control_overhead"].values())
            assert result["control_overhead"]["static_analysis"] == result["static_analysis_seconds"]
    asyncio.run(run())


def profile_row(scale=1):
    return {"status": "complete", "control_profile_version": CONTROL_PROFILE_VERSION,
            "control_overhead": {k: v * scale for k, v in {
                "static_analysis": .001, "materialization": .002,
                "guard": .003, "dispatch": .004}.items()}}


def test_report_exports_components_and_consistent_units():
    result = _control_overhead_summary([profile_row(), profile_row(3)])
    assert result["control_overhead_status"] == "complete"
    assert result["control_profile_records"] == 2
    assert result["static_analysis_ms"] == pytest.approx(2)
    assert result["materialization_ms"] == pytest.approx(4)
    assert result["guard_ms"] == pytest.approx(6)
    assert result["queue_dispatch_ms"] == pytest.approx(8)
    assert result["runtime_control_seconds"] == pytest.approx(.018)
    assert result["control_seconds"] == pytest.approx(.020)
    assert result["control_ms"] == pytest.approx(20)


@pytest.mark.parametrize("mutation", ["legacy", "missing", "negative", "nonfinite", "failed"])
def test_report_does_not_mix_missing_or_legacy_profiles(mutation):
    row = profile_row()
    if mutation == "legacy":
        row.pop("control_profile_version")
    elif mutation == "missing":
        row["control_overhead"] = None
    elif mutation == "failed":
        row["status"] = "incomplete"
    else:
        row["control_overhead"]["guard"] = -1 if mutation == "negative" else float("nan")
    summary = _control_overhead_summary([profile_row(), row])
    assert summary["control_overhead_status"] == "incomplete"
    assert summary["control_profile_records"] == 1
    assert summary["control_seconds"] is None
    assert summary["static_analysis_ms"] is None


def test_full_report_persists_breakdown_and_rejects_legacy_coverage(tmp_path, monkeypatch):
    import json
    import evoagentx.compactflow.paper_reporting as reporting
    from test_evolution import pilot
    config = pilot()
    plan = {"studies": ["execution_main"], "benchmarks": ["GAIA"], "variants": {},
            "execution": {"tasks": 1, "seeds": [42], "warmups": 0, "repetitions": 1},
            "external_missing": ["a2flow", "llmorch"]}
    (tmp_path / "config.lock.json").write_text(json.dumps(config))
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(plan))
    (tmp_path / "sample_manifest.jsonl").write_text("")
    records = tmp_path / "studies/execution_main/records"
    records.mkdir(parents=True)
    row = {**profile_row(), "benchmark": "GAIA", "task_id": "a", "seed": 42,
           "case": {"name": "guarded"}, "repetition": 0, "warmup": False}
    path = records / "guarded.json"
    path.write_text(json.dumps({"record": row}))
    monkeypatch.setattr(reporting, "_plots", lambda *args: [])
    summary = report_paper(tmp_path)
    assert summary["checks"]["control_overhead"]["status"] == "complete"
    with (tmp_path / "tables/execution.csv").open() as stream:
        result = next(csv.DictReader(stream))
    assert float(result["control_ms"]) == pytest.approx(10)
    assert float(result["static_analysis_ms"]) == pytest.approx(1)
    row.pop("control_profile_version")
    path.write_text(json.dumps({"record": row}))
    summary = report_paper(tmp_path)
    assert summary["checks"]["control_overhead"]["status"] == "incomplete"
