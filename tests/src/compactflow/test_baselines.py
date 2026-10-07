import asyncio, copy, json
from dataclasses import asdict
from pathlib import Path
import pytest
from evoagentx.compactflow.baselines import (
    AFlowAdapter,
    EvoAgentXAdapter,
    BaselineUnavailable,
    ConstructionBaselineRunner,
    WorkflowCandidate,
    FrozenBenchmarkView,
    verify_manifest,
)
from evoagentx.compactflow.baseline_native import ModelSession, SearchBudget
from evoagentx.compactflow.baseline_workflows import (
    export_aflow,
    UnsupportedWorkflow,
    lower_aflow,
    expression_value,
    graph_topology,
)
from evoagentx.compactflow.benchmarks import BenchmarkTask
from evoagentx.compactflow.replay import digest
from evoagentx.compactflow.runtime import CompactFlowRuntime

ROOT = Path(__file__).resolve().parents[3]
GRAPH = """class Workflow:
    def __init__(self, name, llm_config, benchmark):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
    async def __call__(self, problem):
        result = await self.custom(input=problem, instruction=prompt_custom.SOLVE)
        return result['response']
"""


def config():
    c = json.loads(
        (ROOT / "examples/compactflow/configs/qwen3_coder_a100.pilot.json").read_text()
    )
    c["evaluation"]["generation_seeds"] = [42, 43]
    c["construction"]["heldout_seeds"] = [101]
    c["baselines"]["aflow"].update(max_rounds=1, validation_rounds=1)
    c["baselines"]["evoagentx"].update(max_iterations=1, validation_rounds=1)
    return c


def tasks():
    return [
        BenchmarkTask(
            "MATH",
            s,
            q,
            "SECRET_GOLD",
            s,
            split=s,
            metadata={"validation_fold": 1} if s == "validation" else {},
        )
        for s, q in [
            ("source", "How many apples?"),
            ("validation", "Count the bananas."),
            ("target", "Count cherries."),
        ]
    ]


def manifest(tmp_path, rows=None):
    rows = tasks() if rows is None else rows
    p = tmp_path / "sample_manifest.jsonl"
    p.write_text(
        "".join(
            json.dumps(
                {
                    "benchmark": t.benchmark,
                    "task_id": t.task_id,
                    "family_id": t.family_id,
                    "split": t.split,
                    "validation_fold": t.metadata.get("validation_fold"),
                    "task_digest": digest(asdict(t)),
                }
            )
            + "\n"
            for t in rows
        )
    )
    return p


def runner(tmp_path, adapters, **kwargs):
    c = kwargs.pop("config", config())
    m = manifest(tmp_path)
    return ConstructionBaselineRunner(
        tasks(),
        output=tmp_path / "out",
        adapters=adapters,
        config=c,
        manifest=m,
        evaluate_fn=lambda t, a: {"quality": float(a == "1")},
        **kwargs,
    )


class FakeAdapter:
    method = "aflow"

    def __init__(self, missing=False):
        self.calls = []
        self.missing = missing
        self.searches = []

    def preflight(self, config):
        pass

    async def search(self, source_tasks, *, seed, config, workspace, score):
        assert [t["task_id"] for t in source_tasks] == ["source"]
        assert all("answer" not in t and "tests" not in t for t in source_tasks)
        self.searches.append(seed)
        c = WorkflowCandidate.create(
            self.method, "MATH", export_aflow(GRAPH, "SOLVE = 'Solve '")
        )
        assert await score(c, 101) == 1
        return [c]

    async def execute(self, candidate, public_task, *, seed, workspace, budget=None):
        self.calls.append((public_task["task_id"], seed))
        assert "SECRET_GOLD" not in json.dumps(public_task)
        return {
            "runtime": "complete_dependency",
            "answer": "1",
            "valid": True,
            "tokens": {"usage_complete": not self.missing, "total_tokens": 5},
            "latency": 0.01,
            "topology": graph_topology(candidate.artifact),
        }


def test_frozen_manifest_and_resume(tmp_path):
    a = FakeAdapter()
    r = runner(tmp_path, [a])
    summary = asyncio.run(r.run())
    assert summary["publication_status"] == "incomplete"
    assert summary["methods"]["aflow"]["MATH"]["status"] == "complete"
    assert [t for t, s in a.calls][-2:] == ["target", "target"]
    records = [
        json.loads(l)
        for l in (tmp_path / "out/baseline_records.jsonl").read_text().splitlines()
    ]
    assert {r["split"] for r in records} == {"source", "validation", "target"}
    assert {
        (r["task_id"], r["seed"], r["benchmark"])
        for r in records
        if r["split"] == "target"
    } == {("target", 42, "MATH"), ("target", 43, "MATH")}
    b = FakeAdapter()
    second = runner(tmp_path, [b])
    asyncio.run(second.run(resume=True))
    assert b.calls == [] and b.searches == []
    second.config["model"]["name"] = "changed"
    with pytest.raises(ValueError, match="lock mismatch"):
        asyncio.run(second.run(resume=True))


def test_manifest_tamper_and_frozen_view(tmp_path):
    p = manifest(tmp_path)
    t = tasks()
    t[0].answer = "changed"
    with pytest.raises(ValueError, match="content mismatch"):
        verify_manifest(t, p)
    view = FrozenBenchmarkView("MATH", [tasks()[0].public_input()])
    with pytest.raises(BaselineUnavailable):
        view.get_test_data()
    with pytest.raises(ValueError):
        view.get_dev_data(sample_k=50)


def test_missing_dependencies_only_coverage(tmp_path):
    class Missing(FakeAdapter):
        def preflight(self, c):
            raise BaselineUnavailable("missing test dependency")

    result = asyncio.run(runner(tmp_path, [Missing()]).run())
    assert result["records"] == 0
    assert (tmp_path / "out/baseline_records.jsonl").read_text() == ""
    assert "missing test dependency" in result["methods"]["aflow"]["MATH"]["reason"]


def test_missing_usage_never_complete(tmp_path):
    result = asyncio.run(runner(tmp_path, [FakeAdapter(missing=True)]).run())
    assert result["methods"]["aflow"]["MATH"]["status"] == "incomplete"
    records = [
        json.loads(l)
        for l in (tmp_path / "out/baseline_records.jsonl").read_text().splitlines()
    ]
    assert records and all(
        r["quality"] is None and r["token_cost"] is None for r in records
    )
    assert all(r["split"] != "target" for r in records)


def test_candidate_key_must_match(tmp_path):
    r = runner(tmp_path, [FakeAdapter()])
    c = WorkflowCandidate.create("aflow", "MBPP", {})
    with pytest.raises(ValueError, match="pair key"):
        asyncio.run(r._evaluate(FakeAdapter(), c, tasks()[0], 42, 42, tmp_path))


@pytest.mark.parametrize(
    "bad",
    [
        GRAPH.replace(
            "return result['response']",
            "if problem:\n            return result['response']",
        ),
        GRAPH.replace(
            "self.custom(input=problem, instruction=prompt_custom.SOLVE)",
            "eval(problem)",
        ),
        GRAPH.replace("operator.Custom", "operator.Programmer"),
        "open('/tmp/forbidden','w')\n" + GRAPH,
    ],
)
def test_dynamic_aflow_fail_closed(bad):
    with pytest.raises(UnsupportedWorkflow):
        export_aflow(bad, "SOLVE = 'Solve '")


def test_canonical_aflow_executes_real_gfrg():
    artifact = export_aflow(GRAPH, "SOLVE = 'Solve '")

    def factory(op, scope):
        async def execute(**kwargs):
            return {"response": kwargs["instruction"] + kwargs["input"]}

        return execute

    g = lower_aflow(artifact, {"problem": "apple"}, factory, 4)
    result = asyncio.run(CompactFlowRuntime(g, mode="complete").execute({}))
    assert not result.errors
    assert (
        expression_value(
            artifact["result"],
            {"problem": "apple"},
            {k: v["result"] for k, v in result.outputs.items()},
        )
        == "Solve apple"
    )


class FakeModelClient:
    requests = []

    def __init__(self, *args, **kwargs):
        self.records = []

    async def text(self, system, prompt, *, component, seed):
        assert "SECRET_GOLD" not in system + prompt
        self.requests.append((component, seed, system, prompt))
        self.records.append(
            {
                "component": component,
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                "usage_missing": False,
            }
        )
        if component == "executor":
            return "<answer>1</answer>" if "<answer>" in system + prompt else "1"
        if "<operator_description>" in prompt:
            return (
                "<modification>Improve instruction</modification><graph>"
                + GRAPH
                + '</graph><prompt>SOLVE = "Solve carefully "</prompt>'
            )
        if "refine the instruction" in prompt:
            return "Solve precisely.\n{problem}"
        if "Workflow Steps:" in prompt:
            fence = chr(96) * 3
            return (
                fence
                + "yaml\n- name: solve\n  args: [problem]\n  outputs: [answer]\n"
                + fence
            )
        return "Improve the supplied workflow."

    def accounting(self):
        return {"usage_complete": True, "total_tokens": 10 * len(self.records)}


@pytest.mark.parametrize("cls", [AFlowAdapter, EvoAgentXAdapter])
def test_real_native_optimizer_fake_model_end_to_end(tmp_path, cls):
    # No fake optimizer/factory: exercise the repository's real optimize().
    FakeModelClient.requests = []
    a = cls(client_factory=FakeModelClient)
    r = runner(tmp_path, [a])
    result = asyncio.run(r.run())
    assert result["methods"][a.method]["MATH"]["status"] == "complete", result[
        "methods"
    ][a.method]["MATH"]
    records = [
        json.loads(l)
        for l in (tmp_path / "out/baseline_records.jsonl").read_text().splitlines()
    ]
    assert all(
        r["status"] == "complete" and r["quality"] == 1 and r["token_cost"] > 0
        for r in records
    )
    assert {r["split"] for r in records} == {"source", "validation", "target"}
    assert any(c == "planner" for c, _, _, _ in FakeModelClient.requests)
    assert {s for c, s, _, _ in FakeModelClient.requests if c == "executor"} >= {
        42,
        43,
        101,
    }
    assert len({r["workflow_id"] for r in records}) >= 2
    before = len(FakeModelClient.requests)
    asyncio.run(
        runner(tmp_path, [cls(client_factory=FakeModelClient)]).run(resume=True)
    )
    assert len(FakeModelClient.requests) == before


def test_budget_and_model_journal(tmp_path):
    FakeModelClient.requests = []
    c = config()
    ledger = SearchBudget(tmp_path / "ledger.json", 100000)

    async def run():
        for _ in range(2):
            s = ModelSession(
                c, 42, tmp_path / "calls", budget=ledger, client_factory=FakeModelClient
            )
            assert (
                await s.text([{"role": "user", "content": "hello"}], "executor", "one")
                == "1"
            )
            assert s.accounting()["total_tokens"] == 10

    asyncio.run(run())
    assert len(FakeModelClient.requests) == 1 and ledger.used == 10
    from evoagentx.compactflow.llm import TokenBudgetExceeded
    with pytest.raises(TokenBudgetExceeded, match="budget"):
        ledger.reserve("x", 100000)


def test_target_cannot_change_policy_library(tmp_path):
    evo = tmp_path / "evo"
    (evo / "policy_snapshots").mkdir(parents=True)
    snap = evo / "policy_snapshots/final.json"
    snap.write_text("{}")

    class Mutating(FakeAdapter):
        async def execute(self, c, t, **kw):
            result = await super().execute(c, t, **kw)
            if t["task_id"] == "target":
                snap.write_text('{"mutated":true}')
            return result

    with pytest.raises(ValueError, match="policy snapshot changed"):
        asyncio.run(runner(tmp_path, [Mutating()], evolution_dir=evo).run())


class FakePolicyClient(FakeModelClient):
    async def stream(self, system, prompt, *, component, seed):
        yield await self.text(system, prompt, component=component, seed=seed)

    async def text(self, system, prompt, *, component, seed):
        await super().text(system, prompt, component=component, seed=seed)
        if component == "executor":
            return json.dumps({"field": "answer", "value": "1"}) + "\n"
        data = json.loads(prompt)
        if component == "query":
            return json.dumps({"query": "fuse compatible reasoning roles"})
        if component == "selector":
            return json.dumps(
                {"selected": [p["id"] for p in data["candidate_policies"][:1]]}
            )
        return json.dumps(
            {
                "nodes": [
                    {
                        "id": "solve",
                        "tool": "llm",
                        "instruction": "Solve",
                        "inputs": {"problem": "$input.question"},
                        "outputs": ["answer"],
                    }
                ],
                "sinks": ["solve"],
                "applied_policies": [],
                "unapplied_policies": [
                    {"id": p["id"], "reason": "not needed for one node"}
                    for p in data.get("selected_policies", [])
                ],
            }
        )


@pytest.mark.parametrize(
    "method", ["base_planner", "expert_policies", "static_library", "compactflow"]
)
def test_frozen_policy_controls_and_shared_runtime(tmp_path, method):
    from evoagentx.compactflow.baseline_controls import PolicyBaselineAdapter
    from evoagentx.compactflow.llm import HashEmbedder

    evo = tmp_path / "evolution"
    (evo / "policy_snapshots").mkdir(parents=True)
    value = json.loads(
        (ROOT / "examples/compactflow/policies/seed_policies.json").read_text()
    )
    for name in ("initial", "bootstrap", "final"):
        snapshot = copy.deepcopy(value)
        if name == "final":
            for p in snapshot["policies"]:
                p["status"] = "verified"
        (evo / "policy_snapshots" / (name + ".json")).write_text(json.dumps(snapshot))
    (evo / "checkpoint.json").write_text(json.dumps({"stage": "complete"}))
    before = {p.name: p.read_bytes() for p in (evo / "policy_snapshots").glob("*")}
    FakeModelClient.requests = []
    adapter = PolicyBaselineAdapter(
        method,
        evolution_dir=evo,
        client_factory=FakePolicyClient,
        embedder=HashEmbedder(),
    )
    summary = asyncio.run(runner(tmp_path, [adapter], evolution_dir=evo).run())
    assert summary["methods"][method]["MATH"]["status"] == "complete"
    records = [
        json.loads(l)
        for l in (tmp_path / "out/baseline_records.jsonl").read_text().splitlines()
    ]
    assert records and all(
        r["quality"] == 1 and r["token_cost"] > 0 and r["node_count"] == 1
        for r in records
    ), records
    assert {
        p.name: p.read_bytes() for p in (evo / "policy_snapshots").glob("*")
    } == before
    if method in {"base_planner", "expert_policies"}:
        assert all(
            c not in {"query", "selector"} for c, _, _, _ in FakeModelClient.requests
        )


def test_failed_model_accounting_stays_incomplete_after_resume(tmp_path):
    class Broken(FakeModelClient):
        async def text(self, *args, **kwargs):
            raise ConnectionError("offline")

    for i in range(2):
        s = ModelSession(config(), 42, tmp_path / "calls", client_factory=Broken)
        with pytest.raises(BaselineUnavailable):
            asyncio.run(
                s.text([{"role": "user", "content": "test"}], "executor", "one")
            )
        assert not s.accounting()["usage_complete"]


def test_search_crash_reservation_is_not_silently_retried(tmp_path):
    p = tmp_path / "ledger.json"
    a = SearchBudget(p, 100)
    a.reserve("uncommitted", 50)
    b = SearchBudget(p, 100)
    assert b.used == 50
    with pytest.raises(BaselineUnavailable, match="durable response"):
        b.reserve("uncommitted", 50)


def test_ged_is_invariant_to_node_order():
    from evoagentx.compactflow.baseline_workflows import topology_statistics

    a = {"nodes": ["Generate", "Review"], "edges": [(0, 1)]}
    b = {"nodes": ["Review", "Generate"], "edges": [(1, 0)]}
    result = topology_statistics([a, b], timeout=3)
    assert result == {"value": 0.0, "pairs": 1, "timed_out": 0}


def test_target_selection_is_frozen_across_resume(tmp_path):
    asyncio.run(runner(tmp_path, [FakeAdapter()]).run())
    selected = next((tmp_path / "out/baselines").glob("*/*/*/selected.json"))
    payload = json.loads(selected.read_text())
    payload["selection"] = "modified"
    selected.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="target selection changed"):
        asyncio.run(runner(tmp_path, [FakeAdapter()]).run(resume=True))
