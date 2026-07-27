"""Bridge native EvoAgentX workflow graphs to CompactFlow's execution IR."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from ..agents.agent_manager import AgentManager
from ..core.base_config import Parameter
from ..models.base_model import BaseLLM
from ..utils.async_utils import is_method_overridden, run_coroutine_sync
from ..workflow.action_graph import ActionGraph
from ..workflow.workflow import WorkFlow
from ..workflow.workflow_graph import WorkFlowGraph, WorkFlowNode
from .compiler import GFRGCompiler
from .runtime import CompactFlowRuntime
from .schema import (
    GFRG,
    CallSpec,
    CallState,
    DataDependency,
    EffectDependency,
    ExecutionMode,
    ExecutionResult,
    ResourceVector,
    StreamContract,
)

OperationMap = Mapping[str, Any]
_PARAMETER_JSON_TYPES = {
    "string": "string",
    "str": "string",
    "integer": "integer",
    "int": "integer",
    "number": "number",
    "float": "number",
    "boolean": "boolean",
    "bool": "boolean",
    "object": "object",
    "dict": "object",
    "array": "array",
    "list": "array",
}


@dataclass(frozen=True, slots=True)
class NodeExecutionContract:
    """Sidecar execution properties absent from ``WorkFlowNode``.

    ``early_safe`` is an explicit assertion that the consumer is pure,
    idempotent, or compensable for the guarded inputs it may observe.
    ``stream_contract`` is meaningful only when the node operation emits
    CompactFlow ``Partial`` events.
    """

    early_safe: bool = False
    resources: Mapping[str, float] = field(default_factory=dict)
    effects: tuple[str, ...] = ()
    stream_contract: StreamContract | None = None


@dataclass(slots=True)
class CompactFlowWorkflowResult:
    """Native workflow outputs together with the full runtime audit record."""

    status: Literal["success", "failed"]
    result: dict[str, Any] | None
    execution: ExecutionResult
    error_msg: str | None = None


def _parameter_schema(parameter: Parameter) -> dict[str, Any]:
    if parameter.json_schema is not None:
        return deepcopy(parameter.json_schema)
    schema_type = _PARAMETER_JSON_TYPES.get(parameter.type)
    if schema_type is None:
        raise ValueError(
            f"parameter {parameter.name!r} has unsupported type "
            f"{parameter.type!r}"
        )
    return {"type": schema_type}


def _node_schema(
    parameters: Sequence[Parameter],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            parameter.name: _parameter_schema(parameter)
            for parameter in parameters
        },
        "required": [
            parameter.name for parameter in parameters if parameter.required
        ],
        "additionalProperties": False,
    }


def _action_graph_target(node: WorkFlowNode):
    action_graph = node.action_graph
    if action_graph is None:
        return None
    if is_method_overridden(action_graph, ActionGraph, "async_execute"):
        return action_graph.async_execute
    if is_method_overridden(action_graph, ActionGraph, "execute"):
        return action_graph.execute
    raise NotImplementedError(
        f"action graph for node {node.name!r} has no execution method"
    )


def _agent_target(
    node: WorkFlowNode,
    *,
    goal: str,
    llm: BaseLLM,
    agent_manager: AgentManager,
    max_execution_steps: int,
):
    async def execute_agent_node(**inputs: Any) -> dict[str, Any]:
        node_copy = deepcopy(node)
        single_node_graph = WorkFlowGraph(
            goal=goal,
            nodes=[node_copy],
            workflow_inputs=deepcopy(node.inputs),
            workflow_outputs=deepcopy(node.outputs),
        )
        workflow = WorkFlow(
            graph=single_node_graph,
            llm=llm,
            agent_manager=agent_manager,
            max_execution_steps=max_execution_steps,
        )
        result = await workflow.async_execute(
            inputs=inputs,
            extract_output=False,
        )
        if result.status != "success" or not isinstance(result.result, dict):
            detail = result.error_msg or result.displayable_error or "unknown error"
            raise RuntimeError(
                f"agent node {node.name!r} failed: {detail}"
            )
        return result.result

    return execute_agent_node


def lower_workflow_graph(
    graph: WorkFlowGraph,
    *,
    operations: OperationMap | None = None,
    contracts: Mapping[str, NodeExecutionContract] | None = None,
    extra_data_dependencies: Sequence[DataDependency] = (),
    resource_capacity: ResourceVector | Mapping[str, float] | None = None,
    llm: BaseLLM | None = None,
    agent_manager: AgentManager | None = None,
    max_execution_steps: int = 5,
) -> GFRG:
    """Lower an EvoAgentX graph into a validated field-readiness graph.

    Native inputs/outputs become exact top-level field dependencies.  More
    precise nested paths can be supplied through ``extra_data_dependencies``;
    an explicit dependency replaces the automatically inferred writer for the
    same consumer target path.

    ActionGraph nodes are executable directly.  Agent nodes are isolated as
    one-node native workflows and require ``llm`` plus ``agent_manager``.
    ``operations`` can override either form and is the recommended hook for
    explicitly instrumented streaming tools.
    """

    if not isinstance(graph, WorkFlowGraph):
        raise TypeError("graph must be a WorkFlowGraph")
    graph._validate_workflow_structure()
    graph._check_workflow_inputs_outputs()
    operations = dict(operations or {})
    contracts = dict(contracts or {})
    unknown_operations = set(operations).difference(graph.list_nodes())
    unknown_contracts = set(contracts).difference(graph.list_nodes())
    if unknown_operations:
        raise ValueError(
            f"operations reference unknown nodes: {sorted(unknown_operations)}"
        )
    if unknown_contracts:
        raise ValueError(
            f"contracts reference unknown nodes: {sorted(unknown_contracts)}"
        )

    output_producers: dict[str, str] = {}
    for node in graph.nodes:
        for output in node.outputs:
            if output.name in output_producers:
                raise ValueError(
                    f"output {output.name!r} has multiple producers"
                )
            output_producers[output.name] = node.name

    data_dependencies = list(extra_data_dependencies)
    explicit_targets = {
        (dependency.consumer, dependency.target_path)
        for dependency in extra_data_dependencies
    }
    for node in graph.nodes:
        for input_parameter in node.inputs:
            target = (node.name, input_parameter.name)
            if target in explicit_targets:
                continue
            producer = output_producers.get(input_parameter.name)
            data_dependencies.append(
                DataDependency(
                    producer=producer,
                    consumer=node.name,
                    source_path=input_parameter.name,
                    target_path=input_parameter.name,
                    required=bool(input_parameter.required),
                    allow_early=True,
                )
            )

    data_pairs = {
        (dependency.producer, dependency.consumer)
        for dependency in data_dependencies
        if dependency.producer is not None
    }
    control_effects: dict[str, list[str]] = defaultdict(list)
    effect_dependencies: list[EffectDependency] = []
    for edge in graph.edges:
        pair = (edge.source, edge.target)
        if pair in data_pairs:
            continue
        effect = f"control:{edge.source}->{edge.target}"
        control_effects[edge.source].append(effect)
        effect_dependencies.append(
            EffectDependency(
                producer=edge.source,
                consumer=edge.target,
                effect=effect,
                allow_partial=False,
            )
        )

    auto_capacity: dict[str, float] = {}
    calls: list[CallSpec] = []
    for node in graph.nodes:
        contract = contracts.get(node.name, NodeExecutionContract())
        target = operations.get(node.name)
        resources = dict(contract.resources)
        if target is None:
            target = _action_graph_target(node)
        if target is None:
            if llm is None or agent_manager is None:
                raise ValueError(
                    f"agent node {node.name!r} requires llm and agent_manager "
                    "or an explicit operation override"
                )
            target = _agent_target(
                node,
                goal=graph.goal,
                llm=llm,
                agent_manager=agent_manager,
                max_execution_steps=max_execution_steps,
            )
            # Reserving every candidate agent is conservative but prevents two
            # concurrently lowered nodes from sharing mutable agent state.
            for agent_name in node.get_agents():
                resource_name = f"agent:{agent_name}"
                resources.setdefault(resource_name, 1.0)
                auto_capacity.setdefault(resource_name, 1.0)
        effects = tuple(
            dict.fromkeys((*contract.effects, *control_effects[node.name]))
        )
        calls.append(
            CallSpec(
                id=node.name,
                target=target,
                input_schema=_node_schema(node.inputs),
                output_schema=_node_schema(node.outputs),
                early_safe=contract.early_safe,
                resources=ResourceVector(resources),
                effects=effects,
                stream_contract=contract.stream_contract,
                metadata={"evoagentx_node": node.name},
            )
        )

    if resource_capacity is None:
        capacity = ResourceVector(auto_capacity)
    elif isinstance(resource_capacity, ResourceVector):
        merged_capacity = {
            **auto_capacity,
            **resource_capacity.as_dict(),
        }
        capacity = ResourceVector(merged_capacity)
    else:
        capacity = ResourceVector({**auto_capacity, **resource_capacity})

    return GFRGCompiler(capacity).compile(
        calls,
        data_dependencies,
        effect_dependencies,
    )


class CompactFlowWorkFlow:
    """Execute a native graph with complete or guarded DAG scheduling."""

    def __init__(
        self,
        graph: WorkFlowGraph,
        *,
        mode: ExecutionMode | str = ExecutionMode.GUARDED,
        operations: OperationMap | None = None,
        contracts: Mapping[str, NodeExecutionContract] | None = None,
        extra_data_dependencies: Sequence[DataDependency] = (),
        resource_capacity: ResourceVector | Mapping[str, float] | None = None,
        llm: BaseLLM | None = None,
        agent_manager: AgentManager | None = None,
        max_execution_steps: int = 5,
    ) -> None:
        self.graph = graph
        self.mode = ExecutionMode(mode)
        self.gfrg = lower_workflow_graph(
            graph,
            operations=operations,
            contracts=contracts,
            extra_data_dependencies=extra_data_dependencies,
            resource_capacity=resource_capacity,
            llm=llm,
            agent_manager=agent_manager,
            max_execution_steps=max_execution_steps,
        )

    def _project_outputs(
        self, execution: ExecutionResult
    ) -> tuple[dict[str, Any], list[str]]:
        projected: dict[str, Any] = {}
        missing: list[str] = []
        producers: dict[str, str] = {}
        for node in self.graph.nodes:
            for output in node.outputs:
                producers[output.name] = node.name
        for output in self.graph.workflow_outputs:
            producer = producers.get(output.name)
            value = (
                execution.outputs.get(producer, {}).get(output.name)
                if producer is not None
                else None
            )
            if producer is not None and output.name in execution.outputs.get(
                producer, {}
            ):
                projected[output.name] = value
            elif output.required:
                missing.append(output.name)
        return projected, missing

    async def async_execute(
        self,
        inputs: Mapping[str, Any] | None = None,
        *,
        expected_arguments: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> CompactFlowWorkflowResult:
        execution = await CompactFlowRuntime(
            self.gfrg, mode=self.mode
        ).execute(inputs, expected_arguments=expected_arguments)
        projected, missing = self._project_outputs(execution)
        failed = [
            call_id
            for call_id, state in execution.states.items()
            if state in {CallState.FAILED, CallState.SKIPPED}
        ]
        if missing or failed:
            details = []
            if missing:
                details.append(f"missing required outputs: {sorted(missing)}")
            if failed:
                details.append(f"non-completed calls: {sorted(failed)}")
            if execution.errors:
                details.append(f"errors: {execution.errors}")
            return CompactFlowWorkflowResult(
                status="failed",
                result=None,
                execution=execution,
                error_msg="; ".join(details),
            )
        return CompactFlowWorkflowResult(
            status="success",
            result=projected,
            execution=execution,
        )

    def execute(
        self,
        inputs: Mapping[str, Any] | None = None,
        *,
        expected_arguments: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> CompactFlowWorkflowResult:
        return run_coroutine_sync(
            self.async_execute(
                inputs,
                expected_arguments=expected_arguments,
            )
        )


__all__ = [
    "CompactFlowWorkFlow",
    "CompactFlowWorkflowResult",
    "NodeExecutionContract",
    "lower_workflow_graph",
]
