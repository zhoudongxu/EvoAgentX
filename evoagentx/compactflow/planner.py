"""EvoAgentX adapter for CompactFlow's policy-guided construction plane.

The adapter keeps the existing :class:`WorkFlowGenerator` public contract:
``generate_workflow`` still returns a native ``WorkFlowGraph``.  The selected
compactness policies are rendered into the task planner's existing
``suggestion`` input, and a failed policy-conditioned plan falls back to the
unmodified generator.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..agents import Agent
from ..workflow.workflow_generator import WorkFlowGenerator
from ..workflow.workflow_graph import WorkFlowGraph
from .construction import ConstructionPlane, PolicyGuidance
from .models import CompactnessPolicy

StructuralQueryBuilder = Callable[[str, Mapping[str, Any]], str]
StructuralContextBuilder = Callable[
    [str, WorkFlowGenerator], Mapping[str, Any]
]


@dataclass(frozen=True, slots=True)
class PlannerTrace:
    """Auditable record for one policy-conditioned planning request."""

    query: str
    context: Mapping[str, Any]
    candidate_policy_ids: tuple[str, ...]
    selected_policy_ids: tuple[str, ...]
    skipped_policies: Mapping[str, str]
    fallback_used: bool
    elapsed_seconds: float
    failure: str | None = None


def build_structural_context(
    goal: str, generator: WorkFlowGenerator
) -> dict[str, Any]:
    """Create deterministic task/tool context for policy applicability.

    This is deliberately structural rather than answer-oriented.  Tool schema
    extraction is defensive because third-party ``Toolkit`` implementations
    do not all expose the same metadata fields.
    """

    tool_records: list[dict[str, Any]] = []
    tool_names: list[str] = []
    for tool in generator.tools or ():
        name = str(getattr(tool, "name", type(tool).__name__))
        tool_names.append(name)
        try:
            schemas = tool.get_tool_schemas()
        except (AttributeError, TypeError):
            schemas = ()
        functions: list[dict[str, Any]] = []
        for schema in schemas or ():
            function = schema.get("function", schema) if isinstance(
                schema, Mapping
            ) else {}
            if isinstance(function, Mapping):
                functions.append(
                    {
                        "name": function.get("name"),
                        "description": function.get("description"),
                        "parameters": function.get("parameters"),
                    }
                )
        tool_records.append({"name": name, "functions": functions})
    return {
        "task": {
            "goal": goal,
            "length": len(goal),
        },
        "tools": tool_names,
        "tool_schemas": tool_records,
        "capabilities": {
            "typed_workflow": True,
            "explicit_streaming": False,
        },
    }


def default_structural_query(
    goal: str, context: Mapping[str, Any]
) -> str:
    """Return a stable retrieval query when no model-backed builder is set."""

    tools = context.get("tools", ())
    tool_text = ", ".join(str(name) for name in tools) or "no external tools"
    return (
        f"Compact workflow for task: {goal}. Available tools: {tool_text}. "
        "Consider repeated roles, linear chains, unused outputs, routing, "
        "stopping criteria, and stream-aware per-item processing."
    )


def render_policy_guidance(
    policies: Sequence[CompactnessPolicy],
) -> str:
    """Render verified policies as direct compact-planning constraints."""

    payload = [
        {
            "id": policy.id,
            "description": policy.description,
            "precondition": policy.precondition,
            "operation": policy.operation,
            "expected_effect": policy.expected_effect,
            "utility": policy.utility,
            "confidence": policy.confidence,
        }
        for policy in policies
    ]
    return (
        "Directly construct one compact workflow. Do not first generate a "
        "large workflow and prune it afterward. Apply a policy only when its "
        "precondition holds; preserve task-essential information, typed "
        "inputs/outputs, validation steps, and reachability. Prefer fewer "
        "nodes, fewer repeated messages, and a shorter critical path subject "
        "to correctness.\n\nSelected compactness policies, in priority order:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


class PolicyGuidedWorkflowGenerator:
    """Wrap an EvoAgentX generator with CompactFlow retrieval and fallback."""

    def __init__(
        self,
        generator: WorkFlowGenerator,
        construction_plane: ConstructionPlane,
        *,
        query_builder: StructuralQueryBuilder | None = None,
        context_builder: StructuralContextBuilder | None = None,
        fallback_on_error: bool = True,
    ) -> None:
        self.generator = generator
        self.construction_plane = construction_plane
        self.query_builder = query_builder or default_structural_query
        self.context_builder = context_builder or build_structural_context
        self.fallback_on_error = fallback_on_error
        self.last_trace: PlannerTrace | None = None

    @staticmethod
    def _validate_compact_graph(
        graph: WorkFlowGraph, *, require_executors: bool = True
    ) -> None:
        if require_executors:
            graph.validate_workflow_graph()
        else:
            graph._validate_workflow_structure()
            graph._check_workflow_inputs_outputs()
        if not nx.is_directed_acyclic_graph(graph.graph):
            raise ValueError("CompactFlow construction requires a DAG")
        if graph.nodes and not graph.find_end_nodes():
            raise ValueError("compact workflow has no reachable output node")

    def _generate_conditioned(
        self,
        goal: str,
        guidance: PolicyGuidance,
        *,
        existing_agents: Sequence[Agent] | None,
        retry: int,
    ) -> WorkFlowGraph:
        suggestion = render_policy_guidance(guidance.selected_policies)
        retry_operation = getattr(
            self.generator, "_execute_with_retry", None
        )
        retries_used = 0

        def execute(operation_name: str, operation, **operation_kwargs):
            nonlocal retries_used
            if retry_operation is None:
                return operation(**operation_kwargs)
            value, added_retries = retry_operation(
                operation_name=operation_name,
                operation=operation,
                retries_left=max(0, retry - retries_used),
                **operation_kwargs,
            )
            retries_used += added_retries
            return value

        plan = execute(
            "Generating a policy-conditioned workflow plan",
            self.generator.generate_plan,
            goal=goal,
            history="",
            suggestion=suggestion,
        )
        graph = execute(
            "Building a policy-conditioned workflow",
            self.generator.build_workflow_from_plan,
            goal=goal,
            plan=plan,
        )
        self._validate_compact_graph(graph, require_executors=False)
        graph = execute(
            "Generating agents for a policy-conditioned workflow",
            self.generator.generate_agents,
            goal=goal,
            workflow=graph,
            existing_agents=list(existing_agents or ()),
        )
        self._validate_compact_graph(graph)
        for node in graph.nodes:
            if node.action_graph is None and not node.agents:
                raise ValueError(
                    f"node {node.name!r} has neither agents nor an action graph"
                )
        return graph

    def generate_workflow(
        self,
        goal: str,
        existing_agents: Sequence[Agent] | None = None,
        retry: int = 1,
        **kwargs: Any,
    ) -> WorkFlowGraph:
        """Generate a native graph using compatible retrieved policies.

        The conditioned path consumes the same bounded retry budget as the
        upstream generator. ``kwargs`` are forwarded only to the fallback
        because the three upstream planning primitives expose fixed
        signatures. The latest decision record is available as
        :attr:`last_trace`.
        """

        if isinstance(retry, bool) or not isinstance(retry, int) or retry < 0:
            raise ValueError("retry must be a non-negative integer")
        started = time.perf_counter()
        context = dict(self.context_builder(goal, self.generator))
        query = self.query_builder(goal, context)
        guidance = self.construction_plane.prepare(query, context)
        selected_ids = tuple(
            policy.id for policy in guidance.selected_policies
        )
        candidate_ids = tuple(
            match.policy.id for match in guidance.candidates
        )
        skipped = {
            item.policy_id: item.reason for item in guidance.selection.skipped
        }
        fallback_used = False
        failure: str | None = None
        try:
            graph = self._generate_conditioned(
                goal,
                guidance,
                existing_agents=existing_agents,
                retry=retry,
            )
        except Exception as error:
            if not self.fallback_on_error:
                raise
            fallback_used = True
            failure = f"{type(error).__name__}: {error}"
            graph = self.generator.generate_workflow(
                goal=goal,
                existing_agents=list(existing_agents or ()),
                retry=retry,
                **kwargs,
            )
            self._validate_compact_graph(graph)
        self.last_trace = PlannerTrace(
            query=query,
            context=context,
            candidate_policy_ids=candidate_ids,
            selected_policy_ids=selected_ids,
            skipped_policies=skipped,
            fallback_used=fallback_used,
            elapsed_seconds=time.perf_counter() - started,
            failure=failure,
        )
        return graph
