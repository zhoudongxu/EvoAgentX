"""Execution-plane schemas for CompactFlow.

The classes in this module form a small sidecar IR.  They intentionally do
not modify :mod:`evoagentx.workflow`: a normal ``WorkFlowGraph`` can be
lowered to these records, while the guarded runtime can also be tested with
plain Python callables.

Only explicitly declared, monotone stream fields may satisfy a guarded data
dependency.  Everything else retains the conservative completion barrier.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

JsonSchema: TypeAlias = Mapping[str, Any]
CallTarget: TypeAlias = Callable[..., Any]


class SchemaContractError(RuntimeError):
    """A call input or output violates its declared JSON Schema contract."""


class ExecutionMode(str, Enum):
    """Scheduling semantics supported by :class:`CompactFlowRuntime`."""

    SEQUENTIAL = "sequential"
    COMPLETE = "complete"
    GUARDED = "guarded"
    INDEPENDENT = "independent_only"
    PERCENTAGE = "percentage_threshold"


class CallState(str, Enum):
    """Atomic lifecycle state for one compiled call."""

    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        return self in {
            CallState.COMPLETED,
            CallState.FAILED,
            CallState.SKIPPED,
        }


class TraceKind(str, Enum):
    """Kinds emitted into the monotonic execution trace."""

    READY = "ready"
    START = "start"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILURE = "failure"
    SKIP = "skip"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True, init=False)
class ResourceVector:
    """A non-negative resource demand or capacity vector.

    Both ``ResourceVector({"gpu": 1})`` and ``ResourceVector(gpu=1)`` are
    accepted.  Unspecified capacities are treated as unbounded by the runtime;
    callers should therefore explicitly configure every resource that needs
    admission control (for example ``agent:planner`` or ``gpu``).
    """

    amounts: Mapping[str, float]

    def __init__(
        self,
        amounts: Mapping[str, float] | None = None,
        **resources: float,
    ) -> None:
        merged: dict[str, float] = dict(amounts or {})
        for name, amount in resources.items():
            if name in merged:
                raise ValueError(f"duplicate resource {name!r}")
            merged[name] = amount
        normalized: dict[str, float] = {}
        for name, amount in merged.items():
            if not isinstance(name, str) or not name:
                raise ValueError("resource names must be non-empty strings")
            numeric = float(amount)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(
                    f"resource {name!r} must have a finite, non-negative amount"
                )
            if numeric:
                normalized[name] = numeric
        object.__setattr__(self, "amounts", normalized)

    def __getitem__(self, name: str) -> float:
        return self.amounts[name]

    def get(self, name: str, default: float = 0.0) -> float:
        return self.amounts.get(name, default)

    def items(self):
        return self.amounts.items()

    def __bool__(self) -> bool:
        return bool(self.amounts)

    def as_dict(self) -> dict[str, float]:
        return dict(self.amounts)


@dataclass(frozen=True, slots=True)
class StreamContract:
    """Fields that an async producer may publish monotonically.

    ``stable_fields`` are exact field paths.  Declaring ``"document"`` does
    not implicitly declare ``"document.title"`` stable.  This exactness keeps
    field-footprint dispatch conservative and auditable.

    ``mutable_fields`` is documentary as well as defensive: a path cannot be
    both stable and mutable.  A non-monotone contract is accepted as metadata
    but never produces early-dispatch guards.
    """

    stable_fields: tuple[str, ...] = ()
    mutable_fields: tuple[str, ...] = ()
    monotone: bool = True

    def __post_init__(self) -> None:
        stable = tuple(dict.fromkeys(self.stable_fields))
        mutable = tuple(dict.fromkeys(self.mutable_fields))
        if any(not isinstance(path, str) for path in stable + mutable):
            raise TypeError("stream field paths must be strings")
        overlap = set(stable).intersection(mutable)
        if overlap:
            raise ValueError(
                "stream fields cannot be both stable and mutable: "
                + ", ".join(sorted(overlap))
            )
        object.__setattr__(self, "stable_fields", stable)
        object.__setattr__(self, "mutable_fields", mutable)

    def explicitly_stabilizes(self, path: str) -> bool:
        return self.monotone and path in self.stable_fields


@dataclass(frozen=True, slots=True)
class CallSpec:
    """One executable call in the guarded field-readiness graph.

    ``target`` is invoked with keyword arguments.  It may be a normal sync
    callable, a coroutine function, or an async generator/event-stream
    function.  A consumer must explicitly opt into ``early_safe`` before any
    partial producer output can make it ready.
    """

    id: str
    target: CallTarget
    input_schema: JsonSchema = field(default_factory=dict)
    output_schema: JsonSchema = field(default_factory=dict)
    early_safe: bool = False
    resources: ResourceVector = field(default_factory=ResourceVector)
    effects: tuple[str, ...] = ()
    stream_contract: StreamContract | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("call id must not be empty")
        if not callable(self.target):
            raise TypeError(f"target for call {self.id!r} must be callable")
        if not isinstance(self.resources, ResourceVector):
            object.__setattr__(self, "resources", ResourceVector(self.resources))
        effects = tuple(dict.fromkeys(self.effects))
        if any(not isinstance(effect, str) or not effect for effect in effects):
            raise ValueError("effect names must be non-empty strings")
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "input_schema", dict(self.input_schema))
        object.__setattr__(self, "output_schema", dict(self.output_schema))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def name(self) -> str:
        """Compatibility alias useful when lowering named workflow nodes."""

        return self.id


@dataclass(frozen=True, slots=True)
class FieldFootprint:
    """One exact producer-field to consumer-argument mapping."""

    source_path: str
    target_path: str
    required: bool = True
    allow_early: bool = True


@dataclass(frozen=True, slots=True)
class DataDependency:
    """A field-level data dependency.

    ``producer=None`` maps a workflow input into a call argument.  Otherwise
    the source is a field in the producer's output.  Completion is the default
    barrier; ``allow_early`` merely permits the compiler to consider a guard
    and is never sufficient without a monotone producer contract and an
    ``early_safe`` consumer.
    """

    producer: str | None
    consumer: str
    source_path: str
    target_path: str
    required: bool = True
    allow_early: bool = True

    @property
    def footprint(self) -> FieldFootprint:
        return FieldFootprint(
            source_path=self.source_path,
            target_path=self.target_path,
            required=self.required,
            allow_early=self.allow_early,
        )


@dataclass(frozen=True, slots=True)
class EffectDependency:
    """An ordering dependency on an externally visible producer effect.

    Effects are completion barriers by default.  ``allow_partial`` is an
    explicit opt-in for effects emitted by a producer event stream; data
    readiness remains independently guarded.
    """

    producer: str
    consumer: str
    effect: str
    allow_partial: bool = False


@dataclass(frozen=True, slots=True)
class GuardSpec:
    """A compiler-proven partial-field readiness guard."""

    producer: str
    consumer: str
    source_path: str
    target_path: str

    @classmethod
    def from_dependency(cls, dependency: DataDependency) -> GuardSpec:
        if dependency.producer is None:
            raise ValueError("workflow inputs do not need partial guards")
        return cls(
            producer=dependency.producer,
            consumer=dependency.consumer,
            source_path=dependency.source_path,
            target_path=dependency.target_path,
        )


@dataclass(frozen=True, slots=True, init=False)
class Partial:
    """A typed partial-output event from an explicit async stream."""

    data: Mapping[str, Any]
    stable_fields: tuple[str, ...]
    effects: tuple[str, ...]
    call_id: str | None

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        stable_fields: tuple[str, ...] = (),
        effects: tuple[str, ...] = (),
        call_id: str | None = None,
        *,
        outputs: Mapping[str, Any] | None = None,
    ) -> None:
        if data is not None and outputs is not None:
            raise ValueError("provide either data or outputs, not both")
        value = outputs if outputs is not None else data
        object.__setattr__(self, "data", dict(value or {}))
        object.__setattr__(
            self, "stable_fields", tuple(dict.fromkeys(stable_fields))
        )
        object.__setattr__(self, "effects", tuple(dict.fromkeys(effects)))
        object.__setattr__(self, "call_id", call_id)

    @property
    def outputs(self) -> Mapping[str, Any]:
        return self.data


@dataclass(frozen=True, slots=True, init=False)
class Complete:
    """A typed terminal success event."""

    data: Mapping[str, Any]
    effects: tuple[str, ...]
    call_id: str | None

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        effects: tuple[str, ...] = (),
        call_id: str | None = None,
        *,
        outputs: Mapping[str, Any] | None = None,
    ) -> None:
        if data is not None and outputs is not None:
            raise ValueError("provide either data or outputs, not both")
        value = outputs if outputs is not None else data
        object.__setattr__(self, "data", dict(value or {}))
        object.__setattr__(self, "effects", tuple(dict.fromkeys(effects)))
        object.__setattr__(self, "call_id", call_id)

    @property
    def outputs(self) -> Mapping[str, Any]:
        return self.data


@dataclass(frozen=True, slots=True)
class Failure:
    """A typed terminal failure event."""

    error: BaseException | str
    call_id: str | None = None


# More explicit aliases for callers that prefer the ``*Event`` spelling.
PartialEvent = Partial
CompleteEvent = Complete
FailureEvent = Failure
RuntimeEvent: TypeAlias = Partial | Complete | Failure
AsyncEventStream: TypeAlias = AsyncIterator[RuntimeEvent]


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One monotonic runtime observation."""

    sequence: int
    timestamp: float
    kind: TraceKind
    call_id: str
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CallTrace:
    """Timing, arguments, and terminal information for one call."""

    call_id: str
    state: CallState = CallState.WAITING
    ready_at: float | None = None
    started_at: float | None = None
    first_output_at: float | None = None
    ended_at: float | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ttfo(self) -> float | None:
        if self.started_at is None or self.first_output_at is None:
            return None
        return self.first_output_at - self.started_at

    @property
    def latency(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return self.ended_at - self.started_at


@dataclass(slots=True)
class ViolationStats:
    """Four correctness/capacity invariants tracked by every execution."""

    argument_mismatch: int = 0
    duplicate_dispatch: int = 0
    effect_order: int = 0
    resource_capacity: int = 0
    unsafe_dispatch: int = 0

    @property
    def total(self) -> int:
        return (
            self.argument_mismatch
            + self.duplicate_dispatch
            + self.effect_order
            + self.resource_capacity
            + self.unsafe_dispatch
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "argument_mismatch": self.argument_mismatch,
            "duplicate_dispatch": self.duplicate_dispatch,
            "effect_order": self.effect_order,
            "resource_capacity": self.resource_capacity,
            "unsafe_dispatch": self.unsafe_dispatch,
        }

    def __getitem__(self, name: str) -> int:
        return self.as_dict()[name]

    def get(self, name: str, default: int = 0) -> int:
        return self.as_dict().get(name, default)


@dataclass(slots=True)
class ExecutionMetrics:
    """End-to-end timings and invariant counters."""

    started_at: float
    ended_at: float | None = None
    first_output_at: float | None = None
    output_ready_at: float | None = None
    first_internal_output_at: float | None = None
    violations: ViolationStats = field(default_factory=ViolationStats)
    peak_resources: dict[str, float] = field(default_factory=dict)

    @property
    def latency(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.output_ready_at or self.ended_at) - self.started_at

    @property
    def ttfo(self) -> float | None:
        if self.first_output_at is None:
            return None
        return self.first_output_at - self.started_at


@dataclass(frozen=True, slots=True)
class GFRG:
    """Compiled guarded field-readiness graph."""

    calls: tuple[CallSpec, ...]
    data_dependencies: tuple[DataDependency, ...] = ()
    effect_dependencies: tuple[EffectDependency, ...] = ()
    guards: tuple[GuardSpec, ...] = ()
    resource_capacity: ResourceVector = field(default_factory=ResourceVector)
    topological_order: tuple[str, ...] = ()

    @property
    def call_map(self) -> dict[str, CallSpec]:
        return {call.id: call for call in self.calls}

    def incoming_data(self, call_id: str) -> tuple[DataDependency, ...]:
        return tuple(
            dependency
            for dependency in self.data_dependencies
            if dependency.consumer == call_id
        )

    def incoming_effects(self, call_id: str) -> tuple[EffectDependency, ...]:
        return tuple(
            dependency
            for dependency in self.effect_dependencies
            if dependency.consumer == call_id
        )

    def has_guard(self, dependency: DataDependency) -> bool:
        if dependency.producer is None:
            return False
        return GuardSpec.from_dependency(dependency) in self.guards


@dataclass(slots=True)
class ExecutionResult:
    """Completed execution, including outputs and audit data."""

    outputs: dict[str, Mapping[str, Any]]
    states: dict[str, CallState]
    trace: tuple[TraceEvent, ...]
    call_traces: dict[str, CallTrace]
    metrics: ExecutionMetrics

    @property
    def arguments(self) -> dict[str, Mapping[str, Any]]:
        return {
            call_id: call_trace.arguments
            for call_id, call_trace in self.call_traces.items()
        }

    @property
    def errors(self) -> dict[str, str]:
        return {
            call_id: call_trace.error
            for call_id, call_trace in self.call_traces.items()
            if call_trace.error is not None
        }


__all__ = [
    "GFRG",
    "AsyncEventStream",
    "CallSpec",
    "CallState",
    "CallTrace",
    "Complete",
    "CompleteEvent",
    "DataDependency",
    "EffectDependency",
    "ExecutionMetrics",
    "ExecutionMode",
    "ExecutionResult",
    "Failure",
    "FailureEvent",
    "FieldFootprint",
    "GuardSpec",
    "Partial",
    "PartialEvent",
    "ResourceVector",
    "RuntimeEvent",
    "SchemaContractError",
    "StreamContract",
    "TraceEvent",
    "TraceKind",
    "ViolationStats",
]
