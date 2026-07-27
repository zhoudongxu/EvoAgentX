"""Offline integration tests for native EvoAgentX workflow lowering."""

from __future__ import annotations

import asyncio
import unittest

import pytest

try:
    from evoagentx.compactflow.adapter import (
        CompactFlowWorkFlow,
        NodeExecutionContract,
        lower_workflow_graph,
    )
except ModuleNotFoundError as error:
    if error.name and error.name.startswith("evoagentx"):
        raise
    pytest.skip(
        "native EvoAgentX integration requires optional project dependencies",
        allow_module_level=True,
    )
from evoagentx.compactflow.schema import (
    Complete,
    ExecutionMode,
    Partial,
    StreamContract,
)
from evoagentx.core.base_config import Parameter
from evoagentx.workflow.workflow_graph import WorkFlowGraph, WorkFlowNode


def parameter(name: str, *, required: bool = True) -> Parameter:
    return Parameter(
        name=name,
        type="integer",
        required=required,
        description=name,
    )


class TestEvoAgentXCompactFlowAdapter(unittest.IsolatedAsyncioTestCase):
    async def test_python_style_parameter_types_become_json_schema_types(self):
        node = WorkFlowNode(
            name="convert",
            description="use canonical JSON schema types",
            inputs=[
                Parameter(
                    name="value",
                    type="int",
                    description="integer value",
                )
            ],
            outputs=[
                Parameter(
                    name="payload",
                    type="dict",
                    description="object payload",
                )
            ],
        )
        graph = WorkFlowGraph(
            goal="convert an integer into an object payload",
            nodes=[node],
            workflow_inputs=node.inputs,
            workflow_outputs=node.outputs,
        )

        lowered = lower_workflow_graph(
            graph,
            operations={
                "convert": lambda value: {"payload": {"value": value}}
            },
        )

        call = lowered.call_map["convert"]
        self.assertEqual(
            call.input_schema["properties"]["value"]["type"], "integer"
        )
        self.assertEqual(
            call.output_schema["properties"]["payload"]["type"], "object"
        )

    async def test_native_fields_lower_and_project_workflow_output(self):
        left = WorkFlowNode(
            name="left",
            description="left branch",
            inputs=[parameter("value")],
            outputs=[parameter("left_value")],
        )
        right = WorkFlowNode(
            name="right",
            description="right branch",
            inputs=[parameter("value")],
            outputs=[parameter("right_value")],
        )
        join = WorkFlowNode(
            name="join",
            description="join branches",
            inputs=[parameter("left_value"), parameter("right_value")],
            outputs=[parameter("answer")],
        )
        graph = WorkFlowGraph(
            goal="add two derived values",
            nodes=[left, right, join],
            workflow_inputs=[parameter("value")],
            workflow_outputs=[parameter("answer")],
        )
        both_started = asyncio.Event()
        starts = 0

        async def left_op(value):
            nonlocal starts
            starts += 1
            if starts == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=0.5)
            return {"left_value": value + 1}

        async def right_op(value):
            nonlocal starts
            starts += 1
            if starts == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=0.5)
            return {"right_value": value + 2}

        workflow = CompactFlowWorkFlow(
            graph,
            mode=ExecutionMode.COMPLETE,
            operations={
                "left": left_op,
                "right": right_op,
                "join": lambda left_value, right_value: {
                    "answer": left_value + right_value
                },
            },
        )
        result = await workflow.async_execute({"value": 3})

        self.assertEqual(result.status, "success")
        self.assertEqual(result.result, {"answer": 9})
        self.assertEqual(starts, 2)
        self.assertEqual(result.execution.metrics.violations.total, 0)

    async def test_native_stream_contract_enables_guarded_consumer(self):
        producer = WorkFlowNode(
            name="producer",
            description="stream a stable answer",
            inputs=[],
            outputs=[parameter("answer"), parameter("metadata")],
        )
        consumer = WorkFlowNode(
            name="consumer",
            description="consume only the stable answer",
            inputs=[parameter("answer")],
            outputs=[parameter("seen")],
        )
        graph = WorkFlowGraph(
            goal="stream one stable field",
            nodes=[producer, consumer],
            workflow_inputs=[],
            workflow_outputs=[parameter("seen")],
        )
        consumer_started = asyncio.Event()

        async def produce():
            yield Partial({"answer": 7}, stable_fields=("answer",))
            await asyncio.wait_for(consumer_started.wait(), timeout=0.5)
            yield Complete({"answer": 7, "metadata": 1})

        async def consume(answer):
            consumer_started.set()
            return {"seen": answer}

        workflow = CompactFlowWorkFlow(
            graph,
            operations={"producer": produce, "consumer": consume},
            contracts={
                "producer": NodeExecutionContract(
                    stream_contract=StreamContract(
                        stable_fields=("answer",)
                    )
                ),
                "consumer": NodeExecutionContract(early_safe=True),
            },
        )
        result = await workflow.async_execute()

        self.assertEqual(result.status, "success")
        self.assertEqual(result.result, {"seen": 7})
        self.assertLess(
            result.execution.call_traces["consumer"].started_at,
            result.execution.call_traces["producer"].ended_at,
        )

    async def test_without_contract_native_stream_keeps_completion_barrier(self):
        order: list[str] = []
        producer = WorkFlowNode(
            name="producer",
            description="mutable draft",
            inputs=[],
            outputs=[parameter("draft")],
        )
        consumer = WorkFlowNode(
            name="consumer",
            description="read final draft",
            inputs=[parameter("draft")],
            outputs=[parameter("seen")],
        )
        graph = WorkFlowGraph(
            goal="read the final draft",
            nodes=[producer, consumer],
            workflow_inputs=[],
            workflow_outputs=[parameter("seen")],
        )

        async def produce():
            order.append("partial")
            yield Partial({"draft": 1})
            await asyncio.sleep(0)
            order.append("complete")
            yield Complete({"draft": 2})

        async def consume(draft):
            order.append("consumer")
            return {"seen": draft}

        workflow = CompactFlowWorkFlow(
            graph,
            operations={"producer": produce, "consumer": consume},
            contracts={
                "consumer": NodeExecutionContract(early_safe=True),
            },
        )
        result = await workflow.async_execute()

        self.assertEqual(result.status, "success")
        self.assertEqual(result.result, {"seen": 2})
        self.assertLess(order.index("complete"), order.index("consumer"))
