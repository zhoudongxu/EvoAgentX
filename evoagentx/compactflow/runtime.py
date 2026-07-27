"""Async event-driven runtime for a compiled CompactFlow GFRG.

The runtime has one state-mutating event loop.  Call workers only publish
typed events through an ``asyncio.Queue``; readiness checks, resource
reservation, state transitions, and descendant cancellation happen under one
async lock.  This gives the in-memory scheduler an auditable at-most-once
dispatch guarantee for each logical call.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import math
from collections.abc import Mapping
from typing import Any

from jsonschema.validators import extend, validator_for

from .compiler import parse_field_path
from .schema import (
    GFRG,
    CallSpec,
    CallState,
    CallTrace,
    Complete,
    DataDependency,
    EffectDependency,
    ExecutionMetrics,
    ExecutionMode,
    ExecutionResult,
    Failure,
    Partial,
    ResourceVector,
    RuntimeEvent,
    SchemaContractError,
    TraceEvent,
    TraceKind,
)

_MISSING = object()


class StreamContractError(RuntimeError):
    """Raised when a producer violates its declared monotone stream."""


class UnsatisfiedWorkflow(RuntimeError):
    """Diagnostic used when a compiled run reaches quiescence."""


def _ignore_required(validator, required, instance, schema):
    """JSON Schema ``required`` validator used for accumulated partials."""

    del validator, required, instance, schema
    yield from ()


_PARTIAL_VALIDATORS: dict[type[Any], type[Any]] = {}


def _validate_json_instance(
    schema: Mapping[str, Any],
    instance: Any,
    *,
    label: str,
    partial: bool = False,
) -> None:
    """Validate one call-boundary value and raise a stable contract error.

    Partial output is validated with only the ``required`` keyword disabled.
    The override is inherited by every nested subschema, so absent fields are
    allowed at any depth while fields that are present must still satisfy
    their types, object properties, ``additionalProperties``, and all other
    declared constraints.
    """

    base_validator = validator_for(schema)
    validator_class = base_validator
    if partial:
        validator_class = _PARTIAL_VALIDATORS.get(base_validator)
        if validator_class is None:
            validator_class = extend(
                base_validator, {"required": _ignore_required}
            )
            _PARTIAL_VALIDATORS[base_validator] = validator_class
    errors = sorted(
        validator_class(schema).iter_errors(instance),
        key=lambda error: tuple(str(token) for token in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = "$" + "".join(
        f"[{token}]" if isinstance(token, int) else f".{token}"
        for token in error.absolute_path
    )
    raise SchemaContractError(
        f"{label} violates its JSON Schema at {path}: {error.message}"
    )


def _deepcopy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - integrations can define arbitrary protocols
        # Argument snapshots must never retain a mutable reference merely
        # because an integration object has an unusual deepcopy protocol.
        return repr(value)


def _path_get(value: Any, path: str, default: Any = _MISSING) -> Any:
    current = value
    for token in parse_field_path(path):
        if (
            isinstance(current, (list, tuple))
            and isinstance(token, str)
            and token.isdigit()
        ):
            token = int(token)
        if isinstance(token, int):
            if (
                not isinstance(current, (list, tuple))
                or token < 0
                or token >= len(current)
            ):
                return default
            current = current[token]
        else:
            if not isinstance(current, Mapping) or token not in current:
                return default
            current = current[token]
    return current


def _path_set(container: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    tokens = parse_field_path(path)
    if not tokens:
        if not isinstance(value, Mapping):
            raise ValueError("a root target path requires a mapping value")
        container.clear()
        container.update(_deepcopy(dict(value)))
        return container

    current: Any = container
    for position, token in enumerate(tokens):
        if (
            isinstance(current, list)
            and isinstance(token, str)
            and token.isdigit()
        ):
            token = int(token)
        last = position == len(tokens) - 1
        next_token = None if last else tokens[position + 1]
        if isinstance(token, str):
            if not isinstance(current, dict):
                raise TypeError(
                    f"cannot assign object key {token!r} in {path!r}"
                )
            if last:
                current[token] = _deepcopy(value)
                continue
            expected = (
                list
                if isinstance(next_token, int)
                or (
                    isinstance(next_token, str)
                    and next_token.isdigit()
                    and path.startswith("/")
                )
                else dict
            )
            child = current.get(token, _MISSING)
            if child is _MISSING:
                child = expected()
                current[token] = child
            if not isinstance(child, expected):
                raise TypeError(
                    f"target path {path!r} conflicts at component {token!r}"
                )
            current = child
        else:
            if not isinstance(current, list):
                raise TypeError(
                    f"cannot assign array index {token} in {path!r}"
                )
            while len(current) <= token:
                current.append(None)
            if last:
                current[token] = _deepcopy(value)
                continue
            expected = list if isinstance(next_token, int) else dict
            child = current[token]
            if child is None:
                child = expected()
                current[token] = child
            if not isinstance(child, expected):
                raise TypeError(
                    f"target path {path!r} conflicts at index {token}"
                )
            current = child
    return container


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = _deepcopy(dict(base))
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = _deepcopy(value)
    return merged


def _normalize_output(call: CallSpec, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    properties = call.output_schema.get("properties", {})
    if isinstance(properties, Mapping) and len(properties) == 1:
        return {next(iter(properties)): value}
    return {"result": value}


class CompactFlowRuntime:
    """Execute a :class:`~evoagentx.compactflow.schema.GFRG`.

    Args:
        graph: A graph produced by :class:`GFRGCompiler`.
        mode: ``sequential`` (one call at a time), ``complete`` (parallel
            calls, but all producer completion barriers retained), or
            ``guarded`` (compiler-proven exact stable fields may satisfy data
            dependencies).
        resource_capacity: Optional per-run capacity override.

    The runtime's at-most-once property covers scheduler dispatch in this
    process.  External side effects still require idempotency keys in their
    underlying tools if exactly-once semantics are needed.
    """

    def __init__(
        self,
        graph: GFRG,
        mode: ExecutionMode | str = ExecutionMode.GUARDED,
        resource_capacity: ResourceVector | Mapping[str, float] | None = None,
    ) -> None:
        if not isinstance(graph, GFRG):
            raise TypeError("graph must be a compiled GFRG")
        self.graph = graph
        self.mode = ExecutionMode(mode)
        if resource_capacity is None:
            self.resource_capacity = graph.resource_capacity
        elif isinstance(resource_capacity, ResourceVector):
            self.resource_capacity = resource_capacity
        else:
            self.resource_capacity = ResourceVector(resource_capacity)
        for call in graph.calls:
            for resource, demand in call.resources.items():
                if (
                    resource in self.resource_capacity.amounts
                    and demand > self.resource_capacity.get(resource)
                ):
                    raise ValueError(
                        f"call {call.id!r} demand for {resource!r} exceeds "
                        "runtime capacity"
                    )

    async def execute(
        self,
        inputs: Mapping[str, Any] | None = None,
        *,
        expected_arguments: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> ExecutionResult:
        """Execute the graph and return outputs, traces, and metrics."""

        execution = _Execution(
            graph=self.graph,
            mode=self.mode,
            inputs=inputs or {},
            resource_capacity=self.resource_capacity,
            expected_arguments=expected_arguments or {},
        )
        return await execution.run()

    async def run(
        self,
        inputs: Mapping[str, Any] | None = None,
        *,
        expected_arguments: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> ExecutionResult:
        """Alias for :meth:`execute`."""

        return await self.execute(
            inputs, expected_arguments=expected_arguments
        )


GuardedRuntime = CompactFlowRuntime


class _Execution:
    def __init__(
        self,
        *,
        graph: GFRG,
        mode: ExecutionMode,
        inputs: Mapping[str, Any],
        resource_capacity: ResourceVector,
        expected_arguments: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.graph = graph
        self.mode = mode
        self.call_map = graph.call_map
        self.order = graph.topological_order or tuple(self.call_map)
        self.inputs = _deepcopy(dict(inputs))
        self.capacity = resource_capacity
        self.expected_arguments = expected_arguments

        self.states = {
            call_id: CallState.WAITING for call_id in self.call_map
        }
        self.outputs: dict[str, dict[str, Any]] = {
            call_id: {} for call_id in self.call_map
        }
        self.stable_fields: dict[str, set[str]] = {
            call_id: set() for call_id in self.call_map
        }
        self.observed_effects: set[tuple[str, str]] = set()
        self.effect_times: dict[tuple[str, str], float] = {}
        self.resource_usage: dict[str, float] = {}
        self.reserved_calls: set[str] = set()
        self.dispatch_ledger: set[str] = set()
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.queue: asyncio.Queue[RuntimeEvent] = asyncio.Queue()
        self.lock = asyncio.Lock()

        loop = asyncio.get_running_loop()
        self.metrics = ExecutionMetrics(started_at=loop.time())
        self.traces = {
            call_id: CallTrace(call_id=call_id)
            for call_id in self.call_map
        }
        self.events: list[TraceEvent] = []
        self.last_timestamp = self.metrics.started_at
        self.sequence = 0

        failure_adjacency: dict[str, set[str]] = {
            call_id: set() for call_id in self.call_map
        }
        for dependency in graph.data_dependencies:
            if dependency.producer is not None and dependency.required:
                failure_adjacency[dependency.producer].add(
                    dependency.consumer
                )
        for dependency in graph.effect_dependencies:
            failure_adjacency[dependency.producer].add(dependency.consumer)
        self.failure_adjacency = failure_adjacency

    def _now(self) -> float:
        timestamp = asyncio.get_running_loop().time()
        timestamp = max(timestamp, self.last_timestamp)
        self.last_timestamp = timestamp
        return timestamp

    def _record(
        self,
        kind: TraceKind,
        call_id: str,
        detail: Mapping[str, Any] | None = None,
        *,
        timestamp: float | None = None,
    ) -> float:
        observed_at = self._now() if timestamp is None else timestamp
        observed_at = max(observed_at, self.last_timestamp)
        self.last_timestamp = observed_at
        self.sequence += 1
        self.events.append(
            TraceEvent(
                sequence=self.sequence,
                timestamp=observed_at,
                kind=kind,
                call_id=call_id,
                detail=dict(detail or {}),
            )
        )
        return observed_at

    async def run(self) -> ExecutionResult:
        try:
            while not all(state.terminal for state in self.states.values()):
                await self._refresh_and_dispatch()
                if all(state.terminal for state in self.states.values()):
                    break
                if not any(
                    state is CallState.RUNNING
                    for state in self.states.values()
                ):
                    async with self.lock:
                        self._terminalize_quiescent_locked()
                    break
                event = await self.queue.get()
                await self._handle_event(event)
        except asyncio.CancelledError:
            await self._cancel_all()
            raise
        finally:
            unfinished = [
                task for task in self.tasks.values() if not task.done()
            ]
            if unfinished:
                for task in unfinished:
                    task.cancel()
                await asyncio.gather(*unfinished, return_exceptions=True)
            self.metrics.ended_at = self._now()

        return ExecutionResult(
            outputs={
                call_id: _deepcopy(output)
                for call_id, output in self.outputs.items()
            },
            states=dict(self.states),
            trace=tuple(self.events),
            call_traces=self.traces,
            metrics=self.metrics,
        )

    async def _refresh_and_dispatch(self) -> None:
        async with self.lock:
            self._propagate_failed_ancestors_locked()
            for call_id in self.order:
                if (
                    self.states[call_id] is CallState.WAITING
                    and self._semantically_ready(call_id)
                ):
                    ready_at = self._record(TraceKind.READY, call_id)
                    self.states[call_id] = CallState.READY
                    self.traces[call_id].state = CallState.READY
                    self.traces[call_id].ready_at = ready_at

            if self.mode is ExecutionMode.SEQUENTIAL and any(
                state is CallState.RUNNING for state in self.states.values()
            ):
                return

            for call_id in self.order:
                if self.states[call_id] is not CallState.READY:
                    continue
                if not self._resources_available(call_id):
                    continue
                self._dispatch_locked(call_id)
                if self.mode is ExecutionMode.SEQUENTIAL:
                    break

    def _semantically_ready(self, call_id: str) -> bool:
        for dependency in self.graph.incoming_data(call_id):
            if not self._data_ready(dependency):
                return False
        for dependency in self.graph.incoming_effects(call_id):
            if not self._effect_ready(dependency):
                return False
        try:
            self._build_arguments(call_id)
        except (KeyError, TypeError, ValueError):
            return False
        return True

    def _data_ready(self, dependency: DataDependency) -> bool:
        if dependency.producer is None:
            value = _path_get(
                self.inputs, dependency.source_path, _MISSING
            )
            return value is not _MISSING or not dependency.required

        producer_state = self.states[dependency.producer]
        if producer_state in {CallState.FAILED, CallState.SKIPPED}:
            return not dependency.required
        if producer_state is CallState.COMPLETED:
            value = _path_get(
                self.outputs[dependency.producer],
                dependency.source_path,
                _MISSING,
            )
            return value is not _MISSING or not dependency.required
        if (
            self.mode is ExecutionMode.GUARDED
            and self.graph.has_guard(dependency)
            and dependency.source_path
            in self.stable_fields[dependency.producer]
        ):
            return (
                _path_get(
                    self.outputs[dependency.producer],
                    dependency.source_path,
                    _MISSING,
                )
                is not _MISSING
            )
        return False

    def _effect_ready(self, dependency: EffectDependency) -> bool:
        observed = (
            dependency.producer,
            dependency.effect,
        ) in self.observed_effects
        if not observed:
            return False
        if dependency.allow_partial:
            return True
        return self.states[dependency.producer] is CallState.COMPLETED

    def _build_arguments(self, call_id: str) -> dict[str, Any]:
        call = self.call_map[call_id]
        properties = call.input_schema.get("properties", {})
        arguments: dict[str, Any] = {}
        if isinstance(properties, Mapping) and properties:
            for name in properties:
                if name in self.inputs:
                    arguments[name] = _deepcopy(self.inputs[name])
        elif not self.graph.incoming_data(call_id):
            arguments.update(_deepcopy(self.inputs))

        for dependency in self.graph.incoming_data(call_id):
            if dependency.producer is None:
                source = self.inputs
            else:
                if self.states[dependency.producer] in {
                    CallState.FAILED,
                    CallState.SKIPPED,
                }:
                    if dependency.required:
                        raise KeyError(
                            f"required producer {dependency.producer!r} "
                            f"for {call_id!r} did not complete"
                        )
                    # Never consume a possibly mutable partial value from a
                    # failed optional producer.
                    continue
                source = self.outputs[dependency.producer]
            value = _path_get(source, dependency.source_path, _MISSING)
            if value is _MISSING:
                if dependency.required:
                    raise KeyError(
                        f"required source field {dependency.source_path!r} "
                        f"for {call_id!r} is unavailable"
                    )
                continue
            _path_set(arguments, dependency.target_path, value)
        return _deepcopy(arguments)

    def _resources_available(self, call_id: str) -> bool:
        for resource, demand in self.call_map[call_id].resources.items():
            capacity = self.capacity.get(resource, math.inf)
            if self.resource_usage.get(resource, 0.0) + demand > capacity + 1e-12:
                return False
        return True

    def _reserve_resources_locked(self, call_id: str) -> None:
        call = self.call_map[call_id]
        for resource, demand in call.resources.items():
            used = self.resource_usage.get(resource, 0.0) + demand
            capacity = self.capacity.get(resource, math.inf)
            if used > capacity + 1e-12:
                self.metrics.violations.resource_capacity += 1
                raise RuntimeError(
                    f"resource invariant violated for {resource!r}"
                )
            self.resource_usage[resource] = used
            self.metrics.peak_resources[resource] = max(
                self.metrics.peak_resources.get(resource, 0.0), used
            )
        self.reserved_calls.add(call_id)

    def _release_resources_locked(self, call_id: str) -> None:
        if call_id not in self.reserved_calls:
            return
        self.reserved_calls.remove(call_id)
        for resource, demand in self.call_map[call_id].resources.items():
            remaining = self.resource_usage.get(resource, 0.0) - demand
            if remaining <= 1e-12:
                self.resource_usage.pop(resource, None)
            else:
                self.resource_usage[resource] = remaining

    def _dispatch_locked(self, call_id: str) -> None:
        if self.states[call_id] is not CallState.READY:
            self.metrics.violations.duplicate_dispatch += 1
            return
        if call_id in self.dispatch_ledger:
            self.metrics.violations.duplicate_dispatch += 1
            return
        # Recheck predicates while holding the same lock used for reservation
        # and the READY -> RUNNING transition.
        if not self._semantically_ready(call_id):
            self.states[call_id] = CallState.WAITING
            self.traces[call_id].state = CallState.WAITING
            return
        for dependency in self.graph.incoming_effects(call_id):
            if not self._effect_ready(dependency):
                self.metrics.violations.effect_order += 1
                return
        if not self._resources_available(call_id):
            return

        arguments = self._build_arguments(call_id)
        try:
            _validate_json_instance(
                self.call_map[call_id].input_schema,
                arguments,
                label=f"input for call {call_id!r}",
            )
        except SchemaContractError as error:
            self._fail_locked(call_id, error)
            return
        self._reserve_resources_locked(call_id)
        self.dispatch_ledger.add(call_id)
        self.states[call_id] = CallState.RUNNING
        call_trace = self.traces[call_id]
        call_trace.state = CallState.RUNNING
        call_trace.arguments = _deepcopy(arguments)
        started_at = self._record(TraceKind.START, call_id)
        call_trace.started_at = started_at
        expected = self.expected_arguments.get(call_id, _MISSING)
        if expected is not _MISSING and arguments != expected:
            self.metrics.violations.argument_mismatch += 1

        task = asyncio.create_task(
            self._invoke(self.call_map[call_id], arguments),
            name=f"compactflow:{call_id}",
        )
        self.tasks[call_id] = task

    async def _invoke(
        self, call: CallSpec, arguments: Mapping[str, Any]
    ) -> None:
        terminal_sent = False
        local_output: dict[str, Any] = {}
        try:
            if inspect.isasyncgenfunction(call.target):
                result: Any = call.target(**arguments)
            elif inspect.iscoroutinefunction(call.target):
                result = await call.target(**arguments)
            else:
                result = await asyncio.to_thread(call.target, **arguments)

            if inspect.isawaitable(result):
                result = await result
            if hasattr(result, "__aiter__"):
                async for raw_event in result:
                    event = self._normalize_event(call, raw_event)
                    if isinstance(event, (Partial, Complete)):
                        local_output = _deep_merge(local_output, event.data)
                    await self.queue.put(event)
                    if isinstance(event, (Complete, Failure)):
                        terminal_sent = True
                        break
                if not terminal_sent:
                    await self.queue.put(
                        Complete(data=local_output, call_id=call.id)
                    )
                    terminal_sent = True
            elif isinstance(result, (Partial, Complete, Failure)):
                event = self._normalize_event(call, result)
                await self.queue.put(event)
                terminal_sent = isinstance(event, (Complete, Failure))
                if isinstance(event, Partial):
                    await self.queue.put(
                        Complete(data=event.data, call_id=call.id)
                    )
                    terminal_sent = True
            else:
                await self.queue.put(
                    Complete(
                        data=_normalize_output(call, result),
                        call_id=call.id,
                    )
                )
                terminal_sent = True
        except asyncio.CancelledError:
            # Internal descendant cancellation has already terminalized state
            # and released resources in the sole state-mutating loop.
            raise
        except BaseException as error:  # noqa: BLE001 - arbitrary call boundary
            await self.queue.put(Failure(error=error, call_id=call.id))
            terminal_sent = True
        finally:
            if not terminal_sent and not asyncio.current_task().cancelled():
                await self.queue.put(
                    Failure(
                        error=RuntimeError(
                            f"call {call.id!r} exited without a terminal event"
                        ),
                        call_id=call.id,
                    )
                )

    @staticmethod
    def _normalize_event(call: CallSpec, event: Any) -> RuntimeEvent:
        if not isinstance(event, (Partial, Complete, Failure)):
            raise TypeError(
                f"async stream for {call.id!r} yielded unsupported "
                f"event {type(event).__name__}"
            )
        if event.call_id not in {None, call.id}:
            raise ValueError(
                f"stream for {call.id!r} emitted event for {event.call_id!r}"
            )
        if isinstance(event, Partial):
            return Partial(
                data=event.data,
                stable_fields=event.stable_fields,
                effects=event.effects,
                call_id=call.id,
            )
        if isinstance(event, Complete):
            return Complete(
                data=event.data,
                effects=event.effects,
                call_id=call.id,
            )
        return Failure(error=event.error, call_id=call.id)

    async def _handle_event(self, event: RuntimeEvent) -> None:
        call_id = event.call_id
        if call_id is None or call_id not in self.call_map:
            return
        async with self.lock:
            if self.states[call_id] is not CallState.RUNNING:
                # A late event from a cancelled descendant or an already
                # terminal stream cannot mutate the current generation.
                return
            if isinstance(event, Partial):
                self._handle_partial_locked(call_id, event)
            elif isinstance(event, Complete):
                self._handle_complete_locked(call_id, event)
            else:
                self._fail_locked(call_id, event.error)

    def _mark_first_output_locked(self, call_id: str, timestamp: float) -> None:
        call_trace = self.traces[call_id]
        if call_trace.first_output_at is None:
            call_trace.first_output_at = timestamp
        if self.metrics.first_output_at is None:
            self.metrics.first_output_at = timestamp

    def _handle_partial_locked(self, call_id: str, event: Partial) -> None:
        call = self.call_map[call_id]
        contract = call.stream_contract
        try:
            if contract is None or not contract.monotone:
                if event.stable_fields:
                    raise StreamContractError(
                        f"{call_id!r} declared stable fields without a "
                        "monotone stream contract"
                    )
            else:
                undeclared = set(event.stable_fields).difference(
                    contract.stable_fields
                )
                if undeclared:
                    raise StreamContractError(
                        f"{call_id!r} emitted undeclared stable fields "
                        f"{sorted(undeclared)}"
                    )
            unknown_effects = set(event.effects).difference(call.effects)
            if unknown_effects:
                raise StreamContractError(
                    f"{call_id!r} emitted undeclared effects "
                    f"{sorted(unknown_effects)}"
                )

            candidate = _deep_merge(self.outputs[call_id], event.data)
            for path in self.stable_fields[call_id]:
                old_value = _path_get(self.outputs[call_id], path, _MISSING)
                new_value = _path_get(candidate, path, _MISSING)
                if old_value is _MISSING or new_value is _MISSING:
                    raise StreamContractError(
                        f"stable field {path!r} disappeared in {call_id!r}"
                    )
                if old_value != new_value:
                    raise StreamContractError(
                        f"stable field {path!r} changed in {call_id!r}"
                    )
            for path in event.stable_fields:
                if _path_get(candidate, path, _MISSING) is _MISSING:
                    raise StreamContractError(
                        f"stable field {path!r} is absent from partial output "
                        f"of {call_id!r}"
                    )
            _validate_json_instance(
                call.output_schema,
                candidate,
                label=f"partial output for call {call_id!r}",
                partial=True,
            )

            self.outputs[call_id] = candidate
            self.stable_fields[call_id].update(event.stable_fields)
            observed_at = self._record(
                TraceKind.PARTIAL,
                call_id,
                {
                    "stable_fields": list(event.stable_fields),
                    "effects": list(event.effects),
                },
            )
            self._mark_first_output_locked(call_id, observed_at)
            for effect in event.effects:
                self.observed_effects.add((call_id, effect))
                self.effect_times[(call_id, effect)] = observed_at
        except Exception as error:  # noqa: BLE001 - contract failure is terminal
            self._fail_locked(call_id, error)

    def _handle_complete_locked(self, call_id: str, event: Complete) -> None:
        call = self.call_map[call_id]
        try:
            unknown_effects = set(event.effects).difference(call.effects)
            if unknown_effects:
                raise StreamContractError(
                    f"{call_id!r} completed with undeclared effects "
                    f"{sorted(unknown_effects)}"
                )
            candidate = _deep_merge(self.outputs[call_id], event.data)
            for path in self.stable_fields[call_id]:
                old_value = _path_get(self.outputs[call_id], path, _MISSING)
                new_value = _path_get(candidate, path, _MISSING)
                if (
                    old_value is _MISSING
                    or new_value is _MISSING
                    or old_value != new_value
                ):
                    raise StreamContractError(
                        f"final output changed stable field {path!r} "
                        f"of {call_id!r}"
                    )
            _validate_json_instance(
                call.output_schema,
                candidate,
                label=f"complete output for call {call_id!r}",
            )
            self.outputs[call_id] = candidate
            observed_at = self._record(
                TraceKind.COMPLETE,
                call_id,
                {"effects": list(dict.fromkeys(call.effects + event.effects))},
            )
            self._mark_first_output_locked(call_id, observed_at)
            for effect in dict.fromkeys(call.effects + event.effects):
                self.observed_effects.add((call_id, effect))
                self.effect_times[(call_id, effect)] = observed_at
            self.states[call_id] = CallState.COMPLETED
            call_trace = self.traces[call_id]
            call_trace.state = CallState.COMPLETED
            call_trace.output = _deepcopy(candidate)
            call_trace.ended_at = observed_at
            self._release_resources_locked(call_id)
        except Exception as error:  # noqa: BLE001 - contract failure is terminal
            self._fail_locked(call_id, error)

    def _fail_locked(self, call_id: str, error: BaseException | str) -> None:
        if self.states[call_id].terminal:
            return
        observed_at = self._record(
            TraceKind.FAILURE, call_id, {"error": str(error)}
        )
        self.states[call_id] = CallState.FAILED
        call_trace = self.traces[call_id]
        call_trace.state = CallState.FAILED
        call_trace.error = str(error)
        call_trace.ended_at = observed_at
        self._release_resources_locked(call_id)
        self._skip_descendants_locked(call_id, f"upstream {call_id!r} failed")

    def _skip_descendants_locked(self, call_id: str, reason: str) -> None:
        pending = list(self.failure_adjacency[call_id])
        visited: set[str] = set()
        while pending:
            descendant = pending.pop()
            if descendant in visited:
                continue
            visited.add(descendant)
            state = self.states[descendant]
            if state in {
                CallState.WAITING,
                CallState.READY,
                CallState.RUNNING,
            }:
                kind = (
                    TraceKind.CANCEL
                    if state is CallState.RUNNING
                    else TraceKind.SKIP
                )
                observed_at = self._record(
                    kind, descendant, {"reason": reason}
                )
                self.states[descendant] = CallState.SKIPPED
                call_trace = self.traces[descendant]
                call_trace.state = CallState.SKIPPED
                call_trace.error = reason
                call_trace.ended_at = observed_at
                self._release_resources_locked(descendant)
                task = self.tasks.get(descendant)
                if task is not None and not task.done():
                    task.cancel()
            pending.extend(self.failure_adjacency[descendant])

    def _propagate_failed_ancestors_locked(self) -> None:
        for call_id, state in tuple(self.states.items()):
            if state is CallState.FAILED:
                self._skip_descendants_locked(
                    call_id, f"upstream {call_id!r} failed"
                )

    def _terminalize_quiescent_locked(self) -> None:
        for call_id in self.order:
            if self.states[call_id] not in {
                CallState.WAITING,
                CallState.READY,
            }:
                continue
            reasons: list[str] = []
            for dependency in self.graph.incoming_data(call_id):
                if not self._data_ready(dependency):
                    source = dependency.producer or "workflow input"
                    reasons.append(
                        f"data {source}:{dependency.source_path}"
                    )
            for dependency in self.graph.incoming_effects(call_id):
                if not self._effect_ready(dependency):
                    reasons.append(
                        f"effect {dependency.producer}:{dependency.effect}"
                    )
            if not self._resources_available(call_id):
                reasons.append("resource capacity")
            if not reasons:
                reasons.append("required input/schema")
            reason = "unsatisfied predicates: " + ", ".join(reasons)
            observed_at = self._record(
                TraceKind.SKIP, call_id, {"reason": reason}
            )
            self.states[call_id] = CallState.SKIPPED
            call_trace = self.traces[call_id]
            call_trace.state = CallState.SKIPPED
            call_trace.error = reason
            call_trace.ended_at = observed_at

    async def _cancel_all(self) -> None:
        async with self.lock:
            for call_id, state in tuple(self.states.items()):
                if state.terminal:
                    continue
                observed_at = self._record(
                    TraceKind.CANCEL,
                    call_id,
                    {"reason": "workflow execution cancelled"},
                )
                self.states[call_id] = CallState.SKIPPED
                call_trace = self.traces[call_id]
                call_trace.state = CallState.SKIPPED
                call_trace.error = "workflow execution cancelled"
                call_trace.ended_at = observed_at
                self._release_resources_locked(call_id)
            for task in self.tasks.values():
                if not task.done():
                    task.cancel()


async def execute_gfrg(
    graph: GFRG,
    inputs: Mapping[str, Any] | None = None,
    *,
    mode: ExecutionMode | str = ExecutionMode.GUARDED,
    resource_capacity: ResourceVector | Mapping[str, float] | None = None,
    expected_arguments: Mapping[str, Mapping[str, Any]] | None = None,
) -> ExecutionResult:
    """One-shot convenience API for a compiled graph."""

    return await CompactFlowRuntime(
        graph, mode=mode, resource_capacity=resource_capacity
    ).execute(inputs, expected_arguments=expected_arguments)


__all__ = [
    "CompactFlowRuntime",
    "GuardedRuntime",
    "SchemaContractError",
    "StreamContractError",
    "UnsatisfiedWorkflow",
    "execute_gfrg",
]
