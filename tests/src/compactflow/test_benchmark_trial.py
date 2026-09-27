"""Trial regressions for private labels, failures, and auditable run artifacts."""

import json
from dataclasses import replace

import pytest

from evoagentx.compactflow.benchmarks import BenchmarkTask
from examples.compactflow import run_benchmark_trial as trial
from examples.compactflow.prepare_benchmark_trial import mbpp_interface, select_tasks


def test_mbpp_interface_exposes_no_solution_body_or_assertions():
    headers = mbpp_interface(
        "def solve(items):\n    return [93817]\ndef helper():\n    return 'secret'",
        ["assert solve([1]) == [93817]"],
    )
    assert headers == ["def solve(items):"]


def test_cohort_selection_is_independent_of_labels_and_input_order():
    tasks = [
        BenchmarkTask(
            "MATH", str(i), "question", "gold", str(i), metadata={"subject": str(i % 3)}
        )
        for i in range(30)
    ]
    changed = [replace(t, answer="different gold") for t in reversed(tasks)]
    selected = select_tasks(tasks, 10, 43, stratified=True)
    assert [t.task_id for t in selected] == [
        t.task_id for t in select_tasks(changed, 10, 43, stratified=True)
    ]
    assert {t.metadata["subject"] for t in selected} == {"0", "1", "2"}


@pytest.mark.asyncio
async def test_trial_persists_real_traces_and_keeps_planner_failures_in_denominator(
    tmp_path, monkeypatch
):
    class Client:
        def __init__(self, *args, **kwargs):
            self.records = []

        async def json(self, system, request, **kwargs):
            assert "PRIVATE-GOLD" not in request
            if json.loads(request)["task"]["task_id"] == "bad":
                raise ValueError("invalid workflow")
            return {
                "nodes": [
                    {
                        "id": "solve",
                        "tool": "llm",
                        "instruction": "return 42",
                        "inputs": {"question": "$input.question"},
                        "outputs": ["answer"],
                    }
                ],
                "sinks": ["solve"],
                "applied_policies": [],
                "unapplied_policies": [],
            }

        async def stream(self, system, request, **kwargs):
            assert "PRIVATE-GOLD" not in request
            yield '{"field":"answer","value":"42"}\n'

        def accounting(self):
            return {"total_tokens": 0, "usage_complete": True}

    monkeypatch.setattr(trial, "ModelClient", Client)
    config = {
        "model": {},
        "evaluation": {"task_token_budget": 65536},
        "construction": {
            "planner": {"max_nodes": 12, "max_repairs": 0, "fallback": "fail"}
        },
        "execution": {
            "external_call_capacity": 4,
            "call_timeout_seconds": 10,
            "workflow_timeout_seconds": 20,
            "materialization_batch_size": 1,
        },
        "tools": {"mbpp_sandbox": {}},
    }
    protocol = {
        "generation_seed": 42,
        "runtime_order_seed": 31415,
        "replay_methods": ["complete", "guarded"],
        "replay_time_scale": 0,
        "scope": "test",
        "count_per_benchmark": 2,
    }
    good = BenchmarkTask(
        "MATH",
        "good",
        "What is 6*7?",
        "42",
        "good",
        metadata={"private": "PRIVATE-GOLD"},
    )
    bad = replace(good, task_id="bad", answer="PRIVATE-GOLD")
    results = [
        await trial.run_task(task, config, protocol, tmp_path / task.task_id)
        for task in (good, bad)
    ]
    assert results[0]["status"] == "completed"
    assert results[1]["status"] == "pipeline_failed"
    assert (
        json.loads((tmp_path / "good/live.json").read_text())["dispatch_audit"][
            "actual_start_events"
        ]
        == 1
    )
    assert (tmp_path / "bad/result.json").exists()
    report = trial.summarize(results, protocol)["datasets"]["MATH"]
    assert report["attempted"] == 2
    assert report["quality_mean_all_attempts"] == 0.5
    assert report["valid_replay_pairs"] == 1
    assert (
        report["safety"]["aggregate_rate"] is None
    )  # No early calls cannot establish 0%.
