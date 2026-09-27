"""Protocol regressions: mismatched settings and safety accounting must fail visibly."""
import asyncio
import json
from pathlib import Path

import pytest

from evoagentx.compactflow.compiler import GFRGCompiler
from evoagentx.compactflow.paper_workflow import compile_spec, validate_spec
from evoagentx.compactflow.replay import (
    ReplayBundle,
    ReplayMiss,
    graph_from_replay,
    graph_to_dict,
)
from evoagentx.compactflow.reproduction_config import (
    ReproductionConfigError,
    validate_model_export,
    validate_reproduction_config,
)
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.safety import CallAudit, audit_execution, summarize_safety
from evoagentx.compactflow.schema import (
    CallSpec,
    CallState,
    Complete,
    DataDependency,
    Failure,
    Partial,
    StreamContract,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "examples/compactflow/configs"
VALUE_SCHEMA = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}


def reference():
    return json.loads((CONFIG / "qwen3_coder_a100.reference.json").read_text())


def test_locked_profiles_and_model_agree():
    for name in ("reference", "pilot"):
        validate_reproduction_config(json.loads((CONFIG / f"qwen3_coder_a100.{name}.json").read_text()), root=ROOT)
    export = json.loads((CONFIG / "qwen3_coder_a100.model.json").read_text())
    validate_model_export(export, reference())
    export["model"]["components"]["executor"]["temperature"] = 0.9
    with pytest.raises(ReproductionConfigError, match="differs"):
        validate_model_export(export, reference())


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c["serving"].update(max_model_len=8192), "context mismatch"),
    (lambda c: c["construction"].update(max_policies=21), "k <= K"),
    (lambda c: c["model"].update(revision="main"), "immutable"),
    (lambda c: c["metrics"]["contract_safety"]["categories"].remove("duplicate_dispatch"), "four"),
    (lambda c: c["construction"].update(quality_tolerance=None), "unresolved"),
    (lambda c: c["artifacts"]["source_hashes"].update({"examples/compactflow/prompts/plan.txt": "0" * 64}), "hash mismatch"),
])
def test_protocol_rejects_inconsistent_or_unresolved_settings(mutation, match):
    config = reference()
    mutation(config)
    with pytest.raises((ReproductionConfigError, ValueError), match=match):
        validate_reproduction_config(config, root=ROOT)


def test_safety_uses_disjoint_early_calls_and_exposes_duplicate_category():
    report = summarize_safety([
        CallAudit("a", True, ("argument_mismatch", "duplicate_dispatch")),
        CallAudit("b", True, ("duplicate_dispatch",)),
        CallAudit("c", True),
        CallAudit("d", False, ("effect_order",)),
    ])
    assert report["denominator"] == 3
    assert report["counts"] == {"argument_mismatch": 1, "duplicate_dispatch": 1, "effect_order": 0, "capacity_overload": 0, "aggregate": 2}
    assert report["rates"]["aggregate"] == pytest.approx(2 / 3)
    assert report["all_call_failure_incidents"] == 4
    assert report["primary_failures"] == {"a": "argument_mismatch", "b": "duplicate_dispatch"}


def test_no_early_calls_and_incomplete_audits_cannot_claim_zero_percent():
    assert summarize_safety([CallAudit("a", False)])["rates"]["aggregate"] is None
    result = summarize_safety([CallAudit("a", True, assessed=False)])
    assert result["rates"]["aggregate"] is None
    assert result["unassessed_early_calls"] == 1


def test_planner_schema_error_has_a_repairable_field_location():
    spec = {"nodes": [{"id": "answer", "tool": "llm", "instruction": "answer", "inputs": {"q": "$input.question"}, "outputs": ["answer"]}],
            "sinks": [{"node": "answer"}], "applied_policies": [], "unapplied_policies": []}
    with pytest.raises(ValueError, match="sinks"):
        validate_spec(spec, {"question": "test"}, set())
    spec["sinks"] = ["answer"]
    validate_spec(spec, {"question": "test"}, set())


@pytest.mark.asyncio
async def test_replay_is_persisted_before_terminal_consumer_exit_and_fails_on_miss(tmp_path):
    async def produce():
        yield Partial({"value": "ready"}, ("value",))
        yield Complete({"value": "ready"})
    bundle = ReplayBundle("workflow")
    call = CallSpec("p", produce, output_schema=VALUE_SCHEMA, stream_contract=StreamContract(("value",)))
    graph = GFRGCompiler().compile([bundle.wrap(call)])
    live = await CompactFlowRuntime(graph).execute()
    path = tmp_path / "replay.json"
    bundle.save(path)
    loaded = ReplayBundle.load(path)
    replayed = await CompactFlowRuntime(graph_from_replay(graph_to_dict(graph), loaded, time_scale=0)).execute()
    assert replayed.outputs == live.outputs
    with pytest.raises(ReplayMiss):
        await anext(loaded.bind("p")(unexpected="argument"))


@pytest.mark.asyncio
async def test_sink_ttfo_and_completed_descendant_invalidation():
    consumer_finished = asyncio.Event()
    async def producer():
        yield Partial({"value": "ready"}, ("value",))
        await asyncio.wait_for(consumer_finished.wait(), timeout=1)
        await asyncio.sleep(0.01)
        yield Failure("source invalidated")
    async def consumer(value):
        consumer_finished.set()
        return {"answer": value}
    graph = GFRGCompiler().compile([
        CallSpec("p", producer, output_schema=VALUE_SCHEMA, stream_contract=StreamContract(("value",))),
        CallSpec("sink", consumer, input_schema=VALUE_SCHEMA, early_safe=True),
    ], [DataDependency("p", "sink", "value", "value")])
    result = await CompactFlowRuntime(graph, sink_ids=["sink"]).execute()
    assert result.metrics.first_output_at == result.call_traces["sink"].first_output_at
    assert result.metrics.first_output_at > result.metrics.first_internal_output_at
    assert result.states["sink"] is CallState.SKIPPED
    assert result.outputs["sink"] == result.call_traces["sink"].output == {}
    assert result.metrics.output_ready_at is None
    audit = audit_execution(graph, result)
    assert audit["denominator"] == 1
    assert audit["unassessed_early_calls"] == 1


@pytest.mark.asyncio
async def test_timeout_reports_timeout_without_fake_terminal_event():
    async def slow():
        await asyncio.sleep(10)
    graph = GFRGCompiler().compile([CallSpec("slow", slow)])
    result = await CompactFlowRuntime(graph, call_timeout=0.01).execute()
    assert result.states["slow"] is CallState.FAILED
    assert "call timeout" in result.errors["slow"]


@pytest.mark.asyncio
async def test_explicit_empty_footprint_does_not_inherit_all_workflow_inputs():
    class FixtureClient:
        async def stream(self, system, prompt, **kwargs):
            assert json.loads(prompt)["inputs"] == {}
            yield '{"field":"answer","value":"42"}\n'
    spec = {"nodes": [{"id": "answer", "tool": "llm", "instruction": "compute 17+25", "inputs": {}, "outputs": ["answer"]}],
            "sinks": ["answer"], "applied_policies": [], "unapplied_policies": []}
    public_input = {"question": "17+25", "context": []}
    graph = compile_spec(spec, public_input, FixtureClient(), seed=42)
    result = await CompactFlowRuntime(graph).execute(public_input)
    assert not result.errors
    assert result.arguments["answer"] == {}
    assert result.outputs["answer"] == {"answer": "42"}
