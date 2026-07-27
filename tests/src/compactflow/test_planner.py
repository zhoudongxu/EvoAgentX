"""Offline tests for the EvoAgentX policy-guided planner adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evoagentx.compactflow.construction import (
    ConstructionConfig,
    ConstructionPlane,
)
from evoagentx.compactflow.models import CompactnessPolicy, PolicyStatus
from evoagentx.compactflow.policy import (
    PolicyLibrary,
    RetrievalConfig,
    SelectionConfig,
)

try:
    from evoagentx.compactflow.planner import (
        PolicyGuidedWorkflowGenerator,
        render_policy_guidance,
    )
    from evoagentx.core.base_config import Parameter
    from evoagentx.workflow.workflow_graph import WorkFlowGraph, WorkFlowNode
except ModuleNotFoundError as error:
    if error.name and error.name.startswith("evoagentx"):
        raise
    pytest.skip(
        "native EvoAgentX integration requires optional project dependencies",
        allow_module_level=True,
    )


class FakeGenerator:
    def __init__(self, *, fail_conditioned: bool = False) -> None:
        self.tools = []
        self.fail_conditioned = fail_conditioned
        self.suggestions: list[str] = []
        self.fallback_calls = 0

    def generate_plan(self, goal, history="", suggestion=""):
        del history
        self.suggestions.append(suggestion)
        if self.fail_conditioned:
            raise ValueError("scripted invalid plan")
        return SimpleNamespace(
            sub_tasks=[
                WorkFlowNode(
                    name="solve",
                    description="solve once",
                    inputs=[
                        Parameter(
                            name="goal",
                            type="string",
                            description="task",
                        )
                    ],
                    outputs=[
                        Parameter(
                            name="answer",
                            type="string",
                            description="answer",
                        )
                    ],
                    agents=["solver"],
                )
            ]
        )

    def build_workflow_from_plan(self, goal, plan):
        return WorkFlowGraph(goal=goal, nodes=plan.sub_tasks)

    def generate_agents(self, goal, workflow, existing_agents=None):
        del goal, existing_agents
        return workflow

    def generate_workflow(self, goal, existing_agents=None, **kwargs):
        del existing_agents, kwargs
        self.fallback_calls += 1
        self.fail_conditioned = False
        plan = self.generate_plan(goal)
        return self.build_workflow_from_plan(goal, plan)


class RetryAwareGenerator(FakeGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.retry_budgets: list[int] = []

    def _execute_with_retry(
        self,
        operation_name,
        operation,
        retries_left=1,
        **kwargs,
    ):
        del operation_name
        self.retry_budgets.append(retries_left)
        return operation(**kwargs), 0


def make_plane() -> ConstructionPlane:
    policy = CompactnessPolicy(
        id="merge-chain",
        description="merge compatible linear chain agents",
        precondition={},
        operation={"type": "fusion"},
        expected_effect={"nodes": "decrease"},
        status=PolicyStatus.VERIFIED,
    )
    return ConstructionPlane(
        PolicyLibrary([policy]),
        config=ConstructionConfig(
            retrieval=RetrievalConfig(semantic_top_k0=1, top_k=1),
            selection=SelectionConfig(
                max_policies=1,
                minimum_score=-1.0,
                minimum_applicability=0.0,
            ),
        ),
    )


def test_selected_policy_is_injected_into_existing_planner() -> None:
    base = FakeGenerator()
    adapter = PolicyGuidedWorkflowGenerator(base, make_plane())

    graph = adapter.generate_workflow("Solve a sufficiently detailed task")

    assert graph.list_nodes() == ["solve"]
    assert "merge-chain" in base.suggestions[0]
    assert "Do not first generate a large workflow" in base.suggestions[0]
    assert adapter.last_trace is not None
    assert adapter.last_trace.selected_policy_ids == ("merge-chain",)
    assert not adapter.last_trace.fallback_used


def test_invalid_conditioned_plan_falls_back_to_base_generator() -> None:
    base = FakeGenerator(fail_conditioned=True)
    adapter = PolicyGuidedWorkflowGenerator(base, make_plane())

    graph = adapter.generate_workflow("Solve a sufficiently detailed task")

    assert graph.list_nodes() == ["solve"]
    assert base.fallback_calls == 1
    assert adapter.last_trace is not None
    assert adapter.last_trace.fallback_used
    assert "scripted invalid plan" in (adapter.last_trace.failure or "")


def test_guidance_renderer_contains_no_identity_metadata() -> None:
    text = render_policy_guidance(make_plane().library.all())

    assert "author" not in text.casefold()
    assert "email" not in text.casefold()
    assert "merge-chain" in text


def test_conditioned_path_uses_upstream_retry_budget() -> None:
    base = RetryAwareGenerator()
    adapter = PolicyGuidedWorkflowGenerator(base, make_plane())

    adapter.generate_workflow(
        "Solve a sufficiently detailed task",
        retry=3,
    )

    assert base.retry_budgets == [3, 3, 3]
