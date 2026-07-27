"""Offline correctness tests for CompactFlow's execution plane."""

from __future__ import annotations

import asyncio
import unittest

from evoagentx.compactflow.compiler import GFRGCompiler
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.schema import (
    CallSpec,
    CallState,
    Complete,
    DataDependency,
    EffectDependency,
    ExecutionMode,
    Failure,
    Partial,
    ResourceVector,
    SchemaContractError,
    StreamContract,
)


def object_schema(
    properties: dict[str, dict],
    required: tuple[str, ...] = (),
) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


INTEGER = {"type": "integer"}
STRING = {"type": "string"}


class TestCompactFlowExecution(unittest.IsolatedAsyncioTestCase):
    async def test_guarded_consumer_starts_on_exact_stable_field(self):
        consumer_started = asyncio.Event()

        async def producer():
            yield Partial({"answer": 7}, stable_fields=("answer",))
            # This makes the test causal instead of depending on sleep timing:
            # the producer cannot complete until the consumer was dispatched.
            await asyncio.wait_for(consumer_started.wait(), timeout=0.5)
            yield Complete({"answer": 7, "metadata": "done"})

        async def consumer(answer: int):
            consumer_started.set()
            return {"seen": answer}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema(
                        {"answer": INTEGER, "metadata": STRING},
                        ("answer", "metadata"),
                    ),
                    stream_contract=StreamContract(
                        stable_fields=("answer",)
                    ),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema(
                        {"answer": INTEGER}, ("answer",)
                    ),
                    output_schema=object_schema(
                        {"seen": INTEGER}, ("seen",)
                    ),
                    early_safe=True,
                ),
            ],
            [
                DataDependency(
                    "producer", "consumer", "answer", "answer"
                )
            ],
        )

        result = await CompactFlowRuntime(
            graph, ExecutionMode.GUARDED
        ).execute()

        self.assertEqual(result.states["producer"], CallState.COMPLETED)
        self.assertEqual(result.states["consumer"], CallState.COMPLETED)
        self.assertEqual(result.outputs["consumer"], {"seen": 7})
        self.assertLess(
            result.call_traces["consumer"].started_at,
            result.call_traces["producer"].ended_at,
        )
        self.assertEqual(result.metrics.violations.total, 0)

    async def test_mutable_field_falls_back_to_completion(self):
        order: list[str] = []

        async def producer():
            order.append("partial")
            yield Partial({"draft": 1})
            await asyncio.sleep(0)
            order.append("producer_complete")
            yield Complete({"draft": 2})

        async def consumer(draft: int):
            order.append("consumer_start")
            return {"seen": draft}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema(
                        {"draft": INTEGER}, ("draft",)
                    ),
                    stream_contract=StreamContract(
                        mutable_fields=("draft",)
                    ),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema(
                        {"draft": INTEGER}, ("draft",)
                    ),
                    output_schema=object_schema({"seen": INTEGER}),
                    early_safe=True,
                ),
            ],
            [
                DataDependency(
                    "producer", "consumer", "draft", "draft"
                )
            ],
        )

        self.assertEqual(graph.guards, ())
        result = await CompactFlowRuntime(graph, "guarded").execute()
        self.assertEqual(result.outputs["consumer"], {"seen": 2})
        self.assertLess(
            order.index("producer_complete"), order.index("consumer_start")
        )

    async def test_full_footprint_waits_for_every_stable_field(self):
        consumer_started = asyncio.Event()

        async def producer():
            yield Partial({"left": 1}, stable_fields=("left",))
            await asyncio.sleep(0.02)
            self.assertFalse(consumer_started.is_set())
            yield Partial({"right": 2}, stable_fields=("right",))
            await asyncio.wait_for(consumer_started.wait(), timeout=0.5)
            yield Complete({"left": 1, "right": 2})

        async def consumer(left: int, right: int):
            consumer_started.set()
            return {"sum": left + right}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema(
                        {"left": INTEGER, "right": INTEGER},
                        ("left", "right"),
                    ),
                    stream_contract=StreamContract(
                        stable_fields=("left", "right")
                    ),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema(
                        {"left": INTEGER, "right": INTEGER},
                        ("left", "right"),
                    ),
                    output_schema=object_schema({"sum": INTEGER}),
                    early_safe=True,
                ),
            ],
            [
                DataDependency(
                    "producer", "consumer", "left", "left"
                ),
                DataDependency(
                    "producer", "consumer", "right", "right"
                ),
            ],
        )

        result = await CompactFlowRuntime(graph, "guarded").execute()
        self.assertEqual(result.outputs["consumer"], {"sum": 3})
        partial_times = [
            event.timestamp
            for event in result.trace
            if event.call_id == "producer" and event.kind.value == "partial"
        ]
        self.assertEqual(len(partial_times), 2)
        self.assertGreaterEqual(
            result.call_traces["consumer"].started_at, partial_times[-1]
        )

    async def test_unsafe_effect_keeps_completion_barrier(self):
        consumer_started = asyncio.Event()

        async def producer():
            yield Partial({"value": 3}, stable_fields=("value",))
            await asyncio.sleep(0.02)
            self.assertFalse(consumer_started.is_set())
            yield Complete({"value": 3})

        async def consumer(value: int):
            consumer_started.set()
            return {"seen": value}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema({"value": INTEGER}),
                    effects=("database_commit",),
                    stream_contract=StreamContract(
                        stable_fields=("value",)
                    ),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema({"value": INTEGER}),
                    output_schema=object_schema({"seen": INTEGER}),
                    early_safe=True,
                ),
            ],
            [
                DataDependency(
                    "producer", "consumer", "value", "value"
                )
            ],
            [
                EffectDependency(
                    "producer",
                    "consumer",
                    "database_commit",
                    allow_partial=False,
                )
            ],
        )

        self.assertEqual(graph.guards, ())
        result = await CompactFlowRuntime(graph, "guarded").execute()
        self.assertEqual(result.states["consumer"], CallState.COMPLETED)
        self.assertLessEqual(
            result.call_traces["producer"].ended_at,
            result.call_traces["consumer"].started_at,
        )
        self.assertEqual(result.metrics.violations.effect_order, 0)

    async def test_resource_capacity_is_never_exceeded(self):
        running = 0
        peak_running = 0

        async def operation():
            nonlocal running, peak_running
            running += 1
            peak_running = max(peak_running, running)
            await asyncio.sleep(0.01)
            running -= 1
            return {"ok": 1}

        calls = [
            CallSpec(
                call_id,
                operation,
                output_schema=object_schema({"ok": INTEGER}),
                resources=ResourceVector(gpu=1),
            )
            for call_id in ("first", "second")
        ]
        graph = GFRGCompiler(ResourceVector(gpu=1)).compile(calls)

        result = await CompactFlowRuntime(graph, "complete").execute()

        self.assertEqual(peak_running, 1)
        self.assertEqual(result.metrics.peak_resources["gpu"], 1)
        self.assertEqual(result.metrics.violations.resource_capacity, 0)
        self.assertTrue(
            all(state is CallState.COMPLETED for state in result.states.values())
        )

    async def test_effect_dependency_enforces_observed_order(self):
        order: list[str] = []

        async def producer():
            await asyncio.sleep(0)
            order.append("effect")
            return {"ok": 1}

        def consumer():
            order.append("consumer")
            return {"ok": 1}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema({"ok": INTEGER}),
                    effects=("publish",),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    output_schema=object_schema({"ok": INTEGER}),
                ),
            ],
            effect_dependencies=[
                EffectDependency("producer", "consumer", "publish")
            ],
        )
        result = await CompactFlowRuntime(graph, "complete").execute()

        self.assertEqual(order, ["effect", "consumer"])
        self.assertLessEqual(
            result.call_traces["producer"].ended_at,
            result.call_traces["consumer"].started_at,
        )
        self.assertEqual(result.metrics.violations.effect_order, 0)

    async def test_repeated_readiness_dispatches_consumer_at_most_once(self):
        calls = 0
        consumer_started = asyncio.Event()

        async def producer():
            yield Partial({"value": 5}, stable_fields=("value",))
            await consumer_started.wait()
            yield Partial({"value": 5}, stable_fields=("value",))
            yield Complete({"value": 5})

        async def consumer(value: int):
            nonlocal calls
            calls += 1
            consumer_started.set()
            await asyncio.sleep(0)
            return {"seen": value}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema({"value": INTEGER}),
                    stream_contract=StreamContract(
                        stable_fields=("value",)
                    ),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema({"value": INTEGER}),
                    output_schema=object_schema({"seen": INTEGER}),
                    early_safe=True,
                ),
            ],
            [
                DataDependency(
                    "producer", "consumer", "value", "value"
                )
            ],
        )
        result = await CompactFlowRuntime(graph, "guarded").execute()

        self.assertEqual(calls, 1)
        self.assertEqual(result.metrics.violations.duplicate_dispatch, 0)
        starts = [
            event
            for event in result.trace
            if event.call_id == "consumer" and event.kind.value == "start"
        ]
        self.assertEqual(len(starts), 1)

    async def test_failure_cancels_running_and_skips_descendants(self):
        consumer_started = asyncio.Event()
        consumer_cancelled = asyncio.Event()

        async def producer():
            yield Partial({"value": 5}, stable_fields=("value",))
            await asyncio.wait_for(consumer_started.wait(), timeout=0.5)
            yield Failure("producer failed")

        async def consumer(value: int):
            consumer_started.set()
            try:
                await asyncio.sleep(10)
                return {"middle": value}
            finally:
                consumer_cancelled.set()

        async def descendant(middle: int):
            return {"final": middle}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema({"value": INTEGER}),
                    stream_contract=StreamContract(
                        stable_fields=("value",)
                    ),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema({"value": INTEGER}),
                    output_schema=object_schema({"middle": INTEGER}),
                    early_safe=True,
                ),
                CallSpec(
                    "descendant",
                    descendant,
                    input_schema=object_schema({"middle": INTEGER}),
                    output_schema=object_schema({"final": INTEGER}),
                ),
            ],
            [
                DataDependency(
                    "producer", "consumer", "value", "value"
                ),
                DataDependency(
                    "consumer", "descendant", "middle", "middle"
                ),
            ],
        )

        result = await asyncio.wait_for(
            CompactFlowRuntime(graph, "guarded").execute(), timeout=1
        )
        self.assertEqual(result.states["producer"], CallState.FAILED)
        self.assertEqual(result.states["consumer"], CallState.SKIPPED)
        self.assertEqual(result.states["descendant"], CallState.SKIPPED)
        self.assertTrue(consumer_cancelled.is_set())

    async def test_guarded_and_complete_capture_identical_arguments(self):
        async def producer():
            yield Partial(
                {"record": {"identifier": 11}},
                stable_fields=("record.identifier",),
            )
            await asyncio.sleep(0)
            yield Complete(
                {"record": {"identifier": 11, "label": "stable"}}
            )

        def consumer(identifier: int):
            return {"seen": identifier}

        record_schema = {
            "type": "object",
            "properties": {
                "identifier": INTEGER,
                "label": STRING,
            },
            "required": ["identifier", "label"],
            "additionalProperties": False,
        }
        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema(
                        {"record": record_schema}, ("record",)
                    ),
                    stream_contract=StreamContract(
                        stable_fields=("record.identifier",)
                    ),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema(
                        {"identifier": INTEGER}, ("identifier",)
                    ),
                    output_schema=object_schema({"seen": INTEGER}),
                    early_safe=True,
                ),
            ],
            [
                DataDependency(
                    "producer",
                    "consumer",
                    "record.identifier",
                    "identifier",
                )
            ],
        )

        complete = await CompactFlowRuntime(graph, "complete").execute()
        guarded = await CompactFlowRuntime(graph, "guarded").execute(
            expected_arguments=complete.arguments
        )
        self.assertEqual(guarded.arguments, complete.arguments)
        self.assertEqual(
            guarded.outputs["consumer"], complete.outputs["consumer"]
        )
        self.assertEqual(
            guarded.metrics.violations.argument_mismatch, 0
        )

    async def test_valid_nested_partial_relaxes_required_recursively(self):
        consumer_started = asyncio.Event()

        async def producer():
            yield Partial(
                {"record": {"identifier": 17}},
                stable_fields=("record.identifier",),
            )
            await asyncio.wait_for(consumer_started.wait(), timeout=0.5)
            yield Complete(
                {"record": {"identifier": 17, "label": "complete"}}
            )

        async def consumer(identifier: int):
            consumer_started.set()
            return {"seen": identifier}

        record_schema = object_schema(
            {"identifier": INTEGER, "label": STRING},
            ("identifier", "label"),
        )
        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema(
                        {"record": record_schema}, ("record",)
                    ),
                    stream_contract=StreamContract(
                        stable_fields=("record.identifier",)
                    ),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema(
                        {"identifier": INTEGER}, ("identifier",)
                    ),
                    output_schema=object_schema(
                        {"seen": INTEGER}, ("seen",)
                    ),
                    early_safe=True,
                ),
            ],
            [
                DataDependency(
                    "producer",
                    "consumer",
                    "record.identifier",
                    "identifier",
                )
            ],
        )

        result = await CompactFlowRuntime(graph, "guarded").execute()

        self.assertEqual(result.states["producer"], CallState.COMPLETED)
        self.assertEqual(result.states["consumer"], CallState.COMPLETED)
        self.assertLess(
            result.call_traces["consumer"].started_at,
            result.call_traces["producer"].ended_at,
        )

    async def test_wrong_output_type_fails_schema_contract(self):
        async def operation():
            return {"count": "not-an-integer"}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "operation",
                    operation,
                    output_schema=object_schema(
                        {"count": INTEGER}, ("count",)
                    ),
                )
            ]
        )

        result = await CompactFlowRuntime(graph).execute()

        self.assertEqual(result.states["operation"], CallState.FAILED)
        self.assertIn("complete output", result.errors["operation"])
        self.assertIn("integer", result.errors["operation"])

    async def test_partial_output_validates_present_nested_type(self):
        async def producer():
            yield Partial({"record": {"identifier": "invalid"}})

        record_schema = object_schema(
            {"identifier": INTEGER, "label": STRING},
            ("identifier", "label"),
        )
        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema(
                        {"record": record_schema}, ("record",)
                    ),
                )
            ]
        )

        result = await CompactFlowRuntime(graph).execute()

        self.assertEqual(result.states["producer"], CallState.FAILED)
        self.assertIn("partial output", result.errors["producer"])
        self.assertIn("integer", result.errors["producer"])

    async def test_partial_output_rejects_nested_additional_property(self):
        async def producer():
            yield Partial({"record": {"unexpected": True}})

        record_schema = object_schema(
            {"identifier": INTEGER, "label": STRING},
            ("identifier", "label"),
        )
        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema(
                        {"record": record_schema}, ("record",)
                    ),
                )
            ]
        )

        result = await CompactFlowRuntime(graph).execute()

        self.assertEqual(result.states["producer"], CallState.FAILED)
        self.assertIn(
            "Additional properties are not allowed",
            result.errors["producer"],
        )

    async def test_complete_output_requires_all_nested_fields(self):
        async def producer():
            yield Partial({"record": {"identifier": 2}})
            yield Complete({})

        record_schema = object_schema(
            {"identifier": INTEGER, "label": STRING},
            ("identifier", "label"),
        )
        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema(
                        {"record": record_schema}, ("record",)
                    ),
                )
            ]
        )

        result = await CompactFlowRuntime(graph).execute()

        self.assertEqual(result.states["producer"], CallState.FAILED)
        self.assertIn("'label' is a required property", result.errors["producer"])

    async def test_invalid_call_input_fails_before_target_runs(self):
        target_ran = False

        async def operation(count: int):
            nonlocal target_ran
            target_ran = True
            return {"count": count}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "operation",
                    operation,
                    input_schema=object_schema(
                        {"count": INTEGER}, ("count",)
                    ),
                    output_schema=object_schema(
                        {"count": INTEGER}, ("count",)
                    ),
                )
            ]
        )

        result = await CompactFlowRuntime(graph).execute(
            {"count": "invalid"}
        )

        self.assertEqual(result.states["operation"], CallState.FAILED)
        self.assertFalse(target_ran)
        self.assertIn("input for call", result.errors["operation"])

    async def test_failed_optional_data_dependency_is_omitted(self):
        async def producer():
            return Failure("optional source unavailable")

        async def consumer(value: int | None = None):
            return {"seen": value is not None}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema({"value": INTEGER}),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema({"value": INTEGER}),
                    output_schema=object_schema(
                        {"seen": {"type": "boolean"}}, ("seen",)
                    ),
                ),
            ],
            [
                DataDependency(
                    "producer",
                    "consumer",
                    "value",
                    "value",
                    required=False,
                )
            ],
        )

        result = await CompactFlowRuntime(graph).execute()

        self.assertEqual(result.states["producer"], CallState.FAILED)
        self.assertEqual(result.states["consumer"], CallState.COMPLETED)
        self.assertEqual(result.arguments["consumer"], {})
        self.assertEqual(result.outputs["consumer"], {"seen": False})

    async def test_effect_dependency_still_propagates_failure(self):
        async def producer():
            return Failure("publish failed")

        async def consumer(value: int | None = None):
            return {"seen": value is not None}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema({"value": INTEGER}),
                    effects=("publish",),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema({"value": INTEGER}),
                    output_schema=object_schema(
                        {"seen": {"type": "boolean"}}, ("seen",)
                    ),
                ),
            ],
            [
                DataDependency(
                    "producer",
                    "consumer",
                    "value",
                    "value",
                    required=False,
                )
            ],
            [EffectDependency("producer", "consumer", "publish")],
        )

        result = await CompactFlowRuntime(graph).execute()

        self.assertEqual(result.states["producer"], CallState.FAILED)
        self.assertEqual(result.states["consumer"], CallState.SKIPPED)

    async def test_sequential_mode_serializes_independent_calls(self):
        running = 0
        peak = 0

        async def operation():
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0)
            running -= 1
            return {"ok": 1}

        graph = GFRGCompiler().compile(
            [
                CallSpec("a", operation),
                CallSpec("b", operation),
            ]
        )
        result = await CompactFlowRuntime(graph, "sequential").execute()
        self.assertEqual(peak, 1)
        self.assertTrue(
            all(state is CallState.COMPLETED for state in result.states.values())
        )


class TestGFRGCompiler(unittest.TestCase):
    def test_rejects_invalid_json_schema(self):
        with self.assertRaisesRegex(
            SchemaContractError, "not a valid JSON Schema"
        ):
            GFRGCompiler().compile(
                [
                    CallSpec(
                        "invalid",
                        dict,
                        output_schema={"type": "not-a-json-type"},
                    )
                ]
            )

    def test_sync_consumer_cannot_receive_partial_guard(self):
        async def producer():
            yield Partial({"value": 1}, stable_fields=("value",))
            yield Complete({"value": 1})

        def consumer(value: int):
            return {"seen": value}

        graph = GFRGCompiler().compile(
            [
                CallSpec(
                    "producer",
                    producer,
                    output_schema=object_schema(
                        {"value": INTEGER}, ("value",)
                    ),
                    stream_contract=StreamContract(
                        stable_fields=("value",)
                    ),
                ),
                CallSpec(
                    "consumer",
                    consumer,
                    input_schema=object_schema(
                        {"value": INTEGER}, ("value",)
                    ),
                    output_schema=object_schema(
                        {"seen": INTEGER}, ("seen",)
                    ),
                    early_safe=True,
                ),
            ],
            [
                DataDependency(
                    "producer", "consumer", "value", "value"
                )
            ],
        )

        self.assertEqual(graph.guards, ())

    def test_rejects_invalid_schema_path(self):
        graph_calls = [
            CallSpec(
                "producer",
                lambda: {"actual": 1},
                output_schema=object_schema({"actual": INTEGER}),
            ),
            CallSpec(
                "consumer",
                lambda wanted: {"wanted": wanted},
                input_schema=object_schema({"wanted": INTEGER}),
            ),
        ]
        with self.assertRaisesRegex(ValueError, "source path"):
            GFRGCompiler().compile(
                graph_calls,
                [
                    DataDependency(
                        "producer", "consumer", "missing", "wanted"
                    )
                ],
            )

    def test_rejects_combined_data_effect_cycle(self):
        calls = [
            CallSpec("a", dict, effects=("a_done",)),
            CallSpec("b", lambda value=None: {}),
        ]
        with self.assertRaisesRegex(ValueError, "acyclic"):
            GFRGCompiler().compile(
                calls,
                [DataDependency("b", "a", "", "", required=False)],
                [EffectDependency("a", "b", "a_done")],
            )

    def test_rejects_demand_larger_than_capacity(self):
        with self.assertRaisesRegex(ValueError, "exceeding capacity"):
            GFRGCompiler(ResourceVector(gpu=1)).compile(
                [
                    CallSpec(
                        "large",
                        dict,
                        resources=ResourceVector(gpu=2),
                    )
                ]
            )

    def test_partial_effect_rejects_non_cancellable_sync_consumer(self):
        async def producer():
            yield Partial(effects=("published",))
            yield Complete(effects=("published",))

        with self.assertRaisesRegex(
            ValueError, "partial effect dependencies require"
        ):
            GFRGCompiler().compile(
                [
                    CallSpec(
                        "producer",
                        producer,
                        effects=("published",),
                    ),
                    CallSpec("consumer", dict),
                ],
                effect_dependencies=[
                    EffectDependency(
                        "producer",
                        "consumer",
                        "published",
                        allow_partial=True,
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
