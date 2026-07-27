"""Reproducible records and metrics for CompactFlow experiments.

This module is deliberately independent of model providers and workflow
runtimes.  A runner records one :class:`RunRecord` per benchmark task, seed,
and method; the helpers below then perform strict paired comparisons and
aggregate execution diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

JsonObject = dict[str, Any]
PairKey = tuple[str, str, int | str, str]
TaskSeedKey = tuple[str, str, int | str]


class PairingError(ValueError):
    """Raised when a paired experiment is incomplete or ambiguous."""


def _finite(name: str, value: float, *, non_negative: bool = False) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if non_negative and numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def percentile(values: Sequence[float], probability: float) -> float | None:
    """Return a deterministic linearly interpolated percentile.

    The interpolation is equivalent to NumPy's default ``linear`` method:
    rank is ``(n - 1) * probability``.  ``None`` is returned for an empty
    input so empty diagnostics remain JSON-compatible.

    Args:
        values: Finite numeric observations.
        probability: Quantile in the inclusive interval ``[0, 1]``.

    Returns:
        The requested percentile, or ``None`` when ``values`` is empty.
    """

    probability = _finite("probability", probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not values:
        return None
    ordered = sorted(_finite("percentile value", value) for value in values)
    rank = (len(ordered) - 1) * probability
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = rank - lower_index
    return (
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


@dataclass(frozen=True, slots=True)
class StructureRecord:
    """Workflow structure measurements for one task execution."""

    node_count: int = 0
    edge_count: int = 0
    critical_path_length: int = 0

    def __post_init__(self) -> None:
        for name in ("node_count", "edge_count", "critical_path_length"):
            _non_negative_int(name, getattr(self, name))
        if self.node_count == 0 and (
            self.edge_count != 0 or self.critical_path_length != 0
        ):
            raise ValueError(
                "an empty graph cannot contain edges or a critical path"
            )
        if self.node_count and self.critical_path_length > self.node_count:
            raise ValueError("critical_path_length cannot exceed node_count")

    @classmethod
    def from_edges(
        cls,
        nodes: Iterable[str],
        edges: Iterable[tuple[str, str]],
    ) -> StructureRecord:
        """Measure a DAG using node count, edge count, and longest node path.

        Duplicate nodes and edges are canonicalized.  The critical-path
        length counts nodes, so an isolated node has length one.

        Raises:
            ValueError: If an edge references an unknown node or the graph
                contains a directed cycle.
        """

        node_set = set(nodes)
        if any(not isinstance(node, str) or not node for node in node_set):
            raise ValueError("graph node identifiers must be non-empty strings")
        edge_set = set(edges)
        successors = {node: set() for node in node_set}
        indegree = {node: 0 for node in node_set}
        for source, target in edge_set:
            if source not in node_set or target not in node_set:
                raise ValueError(
                    f"edge {(source, target)!r} references an unknown node"
                )
            if target not in successors[source]:
                successors[source].add(target)
                indegree[target] += 1

        ready = sorted(node for node, degree in indegree.items() if degree == 0)
        longest = {node: 1 for node in ready}
        visited = 0
        while ready:
            node = ready.pop(0)
            visited += 1
            for successor in sorted(successors[node]):
                longest[successor] = max(
                    longest.get(successor, 1), longest[node] + 1
                )
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort()
        if visited != len(node_set):
            raise ValueError("critical path is undefined for a cyclic graph")

        return cls(
            node_count=len(node_set),
            edge_count=len(edge_set),
            critical_path_length=max(longest.values(), default=0),
        )

    def to_dict(self) -> JsonObject:
        """Return a JSON-serializable representation."""

        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "critical_path_length": self.critical_path_length,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StructureRecord:
        """Build a structure record from serialized data."""

        return cls(
            node_count=int(data.get("node_count", 0)),
            edge_count=int(data.get("edge_count", 0)),
            critical_path_length=int(data.get("critical_path_length", 0)),
        )


@dataclass(frozen=True, slots=True)
class ViolationCounts:
    """Disjoint early-dispatch contract violation counts."""

    argument_mismatch: int = 0
    duplicate_dispatch: int = 0
    effect_order: int = 0
    capacity_overload: int = 0

    def __post_init__(self) -> None:
        for name in (
            "argument_mismatch",
            "duplicate_dispatch",
            "effect_order",
            "capacity_overload",
        ):
            _non_negative_int(name, getattr(self, name))

    @property
    def total(self) -> int:
        """Return the sum of the four disjoint violation categories."""

        return (
            self.argument_mismatch
            + self.duplicate_dispatch
            + self.effect_order
            + self.capacity_overload
        )

    def __add__(self, other: ViolationCounts) -> ViolationCounts:
        if not isinstance(other, ViolationCounts):
            return NotImplemented
        return ViolationCounts(
            argument_mismatch=(
                self.argument_mismatch + other.argument_mismatch
            ),
            duplicate_dispatch=(
                self.duplicate_dispatch + other.duplicate_dispatch
            ),
            effect_order=self.effect_order + other.effect_order,
            capacity_overload=(
                self.capacity_overload + other.capacity_overload
            ),
        )

    def to_dict(self) -> JsonObject:
        """Return a JSON-serializable representation."""

        return {
            "argument_mismatch": self.argument_mismatch,
            "duplicate_dispatch": self.duplicate_dispatch,
            "effect_order": self.effect_order,
            "capacity_overload": self.capacity_overload,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ViolationCounts:
        """Build violation counts from serialized data."""

        return cls(
            argument_mismatch=int(data.get("argument_mismatch", 0)),
            duplicate_dispatch=int(data.get("duplicate_dispatch", 0)),
            effect_order=int(data.get("effect_order", 0)),
            capacity_overload=int(data.get("capacity_overload", 0)),
        )


@dataclass(frozen=True, slots=True)
class RuntimeRecord:
    """Token, timing, readiness, and safety data for one execution."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    ttfo_seconds: float | None = None
    analyzed_data_edges: int = 0
    partial_edges: int = 0
    task_has_opportunity: bool = False
    readiness_gaps: tuple[float, ...] = ()
    first_guard_ready_item_seconds: float | None = None
    dispatched_calls: int = 0
    early_dispatched_calls: int = 0
    violations: ViolationCounts = field(default_factory=ViolationCounts)

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "analyzed_data_edges",
            "partial_edges",
            "dispatched_calls",
            "early_dispatched_calls",
        ):
            _non_negative_int(name, getattr(self, name))
        if self.partial_edges > self.analyzed_data_edges:
            raise ValueError("partial_edges cannot exceed analyzed_data_edges")
        latency = _finite(
            "latency_seconds", self.latency_seconds, non_negative=True
        )
        object.__setattr__(self, "latency_seconds", latency)
        if self.ttfo_seconds is not None:
            ttfo = _finite(
                "ttfo_seconds", self.ttfo_seconds, non_negative=True
            )
            if ttfo > latency:
                raise ValueError("ttfo_seconds cannot exceed latency_seconds")
            object.__setattr__(self, "ttfo_seconds", ttfo)
        gaps = tuple(
            _finite("readiness gap", gap, non_negative=True)
            for gap in self.readiness_gaps
        )
        object.__setattr__(self, "readiness_gaps", gaps)
        if self.first_guard_ready_item_seconds is not None:
            first_item = _finite(
                "first_guard_ready_item_seconds",
                self.first_guard_ready_item_seconds,
                non_negative=True,
            )
            if first_item > latency:
                raise ValueError(
                    "first_guard_ready_item_seconds cannot exceed "
                    "latency_seconds"
                )
            object.__setattr__(
                self, "first_guard_ready_item_seconds", first_item
            )
        if not isinstance(self.violations, ViolationCounts):
            object.__setattr__(
                self,
                "violations",
                ViolationCounts.from_dict(self.violations),
            )

    @property
    def total_tokens(self) -> int:
        """Return input plus output tokens."""

        return self.input_tokens + self.output_tokens

    def to_dict(self) -> JsonObject:
        """Return a JSON-serializable representation."""

        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": self.latency_seconds,
            "ttfo_seconds": self.ttfo_seconds,
            "analyzed_data_edges": self.analyzed_data_edges,
            "partial_edges": self.partial_edges,
            "task_has_opportunity": self.task_has_opportunity,
            "readiness_gaps": list(self.readiness_gaps),
            "first_guard_ready_item_seconds": (
                self.first_guard_ready_item_seconds
            ),
            "dispatched_calls": self.dispatched_calls,
            "early_dispatched_calls": self.early_dispatched_calls,
            "violations": self.violations.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RuntimeRecord:
        """Build a runtime record from serialized data."""

        first_guard_ready_item = data.get(
            "first_guard_ready_item_seconds",
            data.get("first_useful_item_seconds"),
        )
        return cls(
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            latency_seconds=float(data.get("latency_seconds", 0.0)),
            ttfo_seconds=(
                None
                if data.get("ttfo_seconds") is None
                else float(data["ttfo_seconds"])
            ),
            analyzed_data_edges=int(data.get("analyzed_data_edges", 0)),
            partial_edges=int(data.get("partial_edges", 0)),
            task_has_opportunity=bool(
                data.get("task_has_opportunity", False)
            ),
            readiness_gaps=tuple(data.get("readiness_gaps", ())),
            first_guard_ready_item_seconds=(
                None
                if first_guard_ready_item is None
                else float(first_guard_ready_item)
            ),
            dispatched_calls=int(data.get("dispatched_calls", 0)),
            early_dispatched_calls=int(
                data.get("early_dispatched_calls", 0)
            ),
            violations=ViolationCounts.from_dict(
                data.get("violations", {})
            ),
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One benchmark task/seed/method result used by all experiment tables."""

    benchmark: str
    task_id: str
    seed: int | str
    method: str
    quality: float
    structure: StructureRecord = field(default_factory=StructureRecord)
    runtime: RuntimeRecord = field(default_factory=RuntimeRecord)
    valid: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("benchmark", "task_id", "method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, str)):
            raise TypeError("seed must be an integer or string")
        object.__setattr__(self, "quality", _finite("quality", self.quality))
        if not isinstance(self.structure, StructureRecord):
            object.__setattr__(
                self,
                "structure",
                StructureRecord.from_dict(self.structure),
            )
        if not isinstance(self.runtime, RuntimeRecord):
            object.__setattr__(
                self, "runtime", RuntimeRecord.from_dict(self.runtime)
            )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def pair_key(self) -> PairKey:
        """Return the required unique paired-record key."""

        return self.benchmark, self.task_id, self.seed, self.method

    @property
    def task_seed_key(self) -> TaskSeedKey:
        """Return the identity shared by competing methods."""

        return self.benchmark, self.task_id, self.seed

    def to_dict(self) -> JsonObject:
        """Return a JSON-serializable representation."""

        return {
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "seed": self.seed,
            "method": self.method,
            "quality": self.quality,
            "valid": self.valid,
            "structure": self.structure.to_dict(),
            "runtime": self.runtime.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunRecord:
        """Build a run record from serialized data."""

        return cls(
            benchmark=str(data["benchmark"]),
            task_id=str(data["task_id"]),
            seed=data["seed"],
            method=str(data["method"]),
            quality=float(data["quality"]),
            valid=bool(data.get("valid", True)),
            structure=StructureRecord.from_dict(data.get("structure", {})),
            runtime=RuntimeRecord.from_dict(data.get("runtime", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """Means and paired change for one scalar metric."""

    baseline_mean: float
    candidate_mean: float
    absolute_delta: float
    relative_reduction: float | None

    def to_dict(self) -> JsonObject:
        """Return a JSON-serializable representation."""

        return {
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "absolute_delta": self.absolute_delta,
            "relative_reduction": self.relative_reduction,
        }


@dataclass(frozen=True, slots=True)
class PairedSummary:
    """Aggregate comparison between two methods over exact task/seed pairs."""

    baseline_method: str
    candidate_method: str
    pair_count: int
    quality: MetricComparison
    total_tokens: MetricComparison
    nodes: MetricComparison
    edges: MetricComparison
    critical_path: MetricComparison
    latency: MetricComparison
    ttfo: MetricComparison | None
    speedup: float | None
    valid_pair_rate: float

    @property
    def delta_quality(self) -> float:
        """Return candidate minus baseline mean quality."""

        return self.quality.absolute_delta

    def to_dict(self) -> JsonObject:
        """Return a JSON-serializable representation."""

        return {
            "baseline_method": self.baseline_method,
            "candidate_method": self.candidate_method,
            "pair_count": self.pair_count,
            "quality": self.quality.to_dict(),
            "delta_quality": self.delta_quality,
            "total_tokens": self.total_tokens.to_dict(),
            "nodes": self.nodes.to_dict(),
            "edges": self.edges.to_dict(),
            "critical_path": self.critical_path.to_dict(),
            "latency": self.latency.to_dict(),
            "ttfo": None if self.ttfo is None else self.ttfo.to_dict(),
            "speedup": self.speedup,
            "valid_pair_rate": self.valid_pair_rate,
        }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute a mean over no values")
    return sum(values) / len(values)


def _relative_reduction(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
) -> float | None:
    baseline_total = sum(baseline_values)
    if baseline_total == 0:
        return 0.0 if sum(candidate_values) == 0 else None
    return (baseline_total - sum(candidate_values)) / baseline_total


def _comparison(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
    *,
    include_reduction: bool = True,
) -> MetricComparison:
    baseline_mean = _mean(baseline_values)
    candidate_mean = _mean(candidate_values)
    return MetricComparison(
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        absolute_delta=candidate_mean - baseline_mean,
        relative_reduction=(
            _relative_reduction(baseline_values, candidate_values)
            if include_reduction
            else None
        ),
    )


def _index_records(records: Iterable[RunRecord]) -> dict[PairKey, RunRecord]:
    indexed: dict[PairKey, RunRecord] = {}
    for record in records:
        if not isinstance(record, RunRecord):
            raise TypeError("paired comparisons require RunRecord instances")
        if record.pair_key in indexed:
            raise PairingError(f"duplicate paired key: {record.pair_key!r}")
        indexed[record.pair_key] = record
    return indexed


def compare_paired_methods(
    records: Iterable[RunRecord],
    *,
    baseline_method: str,
    candidate_method: str,
) -> PairedSummary:
    """Compare two methods with strict benchmark/task/seed pairing.

    Records from unrelated methods are ignored.  Every identity observed on
    either selected method must appear exactly once on both sides.  Error
    messages report the full required four-part keys, making accidental
    seed, method, or benchmark mismatches auditable.
    """

    if baseline_method == candidate_method:
        raise ValueError("baseline and candidate methods must differ")
    selected = [
        record
        for record in records
        if record.method in {baseline_method, candidate_method}
    ]
    indexed = _index_records(selected)
    baseline_keys = {
        record.task_seed_key
        for record in selected
        if record.method == baseline_method
    }
    candidate_keys = {
        record.task_seed_key
        for record in selected
        if record.method == candidate_method
    }
    if not baseline_keys and not candidate_keys:
        raise PairingError("no records found for the requested methods")
    if baseline_keys != candidate_keys:
        missing_candidate = [
            (*key, candidate_method)
            for key in sorted(baseline_keys - candidate_keys, key=repr)
        ]
        missing_baseline = [
            (*key, baseline_method)
            for key in sorted(candidate_keys - baseline_keys, key=repr)
        ]
        raise PairingError(
            "paired record keys do not match; "
            f"missing candidate={missing_candidate}, "
            f"missing baseline={missing_baseline}"
        )

    pairs = [
        (
            indexed[(*key, baseline_method)],
            indexed[(*key, candidate_method)],
        )
        for key in sorted(baseline_keys, key=repr)
    ]
    baseline = [pair[0] for pair in pairs]
    candidate = [pair[1] for pair in pairs]

    def values(items: Sequence[RunRecord], path: str) -> list[float]:
        getters = {
            "quality": lambda item: item.quality,
            "tokens": lambda item: float(item.runtime.total_tokens),
            "nodes": lambda item: float(item.structure.node_count),
            "edges": lambda item: float(item.structure.edge_count),
            "critical_path": lambda item: float(
                item.structure.critical_path_length
            ),
            "latency": lambda item: item.runtime.latency_seconds,
        }
        return [getters[path](item) for item in items]

    ttfo_pairs = [
        (left.runtime.ttfo_seconds, right.runtime.ttfo_seconds)
        for left, right in pairs
        if left.runtime.ttfo_seconds is not None
        and right.runtime.ttfo_seconds is not None
    ]
    ttfo = (
        _comparison(
            [float(item[0]) for item in ttfo_pairs],
            [float(item[1]) for item in ttfo_pairs],
        )
        if ttfo_pairs
        else None
    )
    baseline_latency = values(baseline, "latency")
    candidate_latency = values(candidate, "latency")
    candidate_latency_total = sum(candidate_latency)
    speedup = (
        None
        if candidate_latency_total == 0
        else sum(baseline_latency) / candidate_latency_total
    )

    return PairedSummary(
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        pair_count=len(pairs),
        quality=_comparison(
            values(baseline, "quality"),
            values(candidate, "quality"),
            include_reduction=False,
        ),
        total_tokens=_comparison(
            values(baseline, "tokens"), values(candidate, "tokens")
        ),
        nodes=_comparison(
            values(baseline, "nodes"), values(candidate, "nodes")
        ),
        edges=_comparison(
            values(baseline, "edges"), values(candidate, "edges")
        ),
        critical_path=_comparison(
            values(baseline, "critical_path"),
            values(candidate, "critical_path"),
        ),
        latency=_comparison(baseline_latency, candidate_latency),
        ttfo=ttfo,
        speedup=speedup,
        valid_pair_rate=(
            sum(left.valid and right.valid for left, right in pairs)
            / len(pairs)
        ),
    )


@dataclass(frozen=True, slots=True)
class RuntimeDiagnostics:
    """Aggregate readiness opportunity and contract-safety metrics."""

    task_count: int
    task_opportunity_rate: float
    analyzed_data_edges: int
    partial_edges: int
    partial_edge_ratio: float
    readiness_p50: float | None
    readiness_p90: float | None
    readiness_p95: float | None
    first_guard_ready_item_mean_seconds: float | None
    dispatched_calls: int
    early_dispatched_calls: int
    violation_rates: Mapping[str, float]

    def to_dict(self) -> JsonObject:
        """Return a JSON-serializable representation."""

        return {
            "task_count": self.task_count,
            "task_opportunity_rate": self.task_opportunity_rate,
            "analyzed_data_edges": self.analyzed_data_edges,
            "partial_edges": self.partial_edges,
            "partial_edge_ratio": self.partial_edge_ratio,
            "readiness_p50": self.readiness_p50,
            "readiness_p90": self.readiness_p90,
            "readiness_p95": self.readiness_p95,
            "first_guard_ready_item_mean_seconds": (
                self.first_guard_ready_item_mean_seconds
            ),
            "dispatched_calls": self.dispatched_calls,
            "early_dispatched_calls": self.early_dispatched_calls,
            "violation_rates": dict(self.violation_rates),
        }


def aggregate_runtime_diagnostics(
    records: Iterable[RunRecord],
) -> RuntimeDiagnostics:
    """Aggregate opportunity, readiness, and four violation categories.

    Violation rates are violation incidences per dispatched call.  This common
    denominator covers dispatch, effect, and capacity checks without assuming
    that all violations happen on early-dispatched calls.  The aggregate rate
    can exceed one when one dispatch produces multiple disjoint violations.
    A zero denominator produces ``0.0`` rather than NaN or infinity.
    """

    items = list(records)
    analyzed_edges = sum(item.runtime.analyzed_data_edges for item in items)
    partial_edges = sum(item.runtime.partial_edges for item in items)
    gaps = [
        gap for item in items for gap in item.runtime.readiness_gaps
    ]
    first_items = [
        item.runtime.first_guard_ready_item_seconds
        for item in items
        if item.runtime.first_guard_ready_item_seconds is not None
    ]
    dispatched_calls = sum(item.runtime.dispatched_calls for item in items)
    early_calls = sum(
        item.runtime.early_dispatched_calls for item in items
    )
    violations = sum(
        (item.runtime.violations for item in items),
        start=ViolationCounts(),
    )
    denominator = float(dispatched_calls)

    def violation_rate(count: int) -> float:
        return 0.0 if denominator == 0 else count / denominator

    rates = {
        "argument_mismatch": violation_rate(
            violations.argument_mismatch
        ),
        "duplicate_dispatch": violation_rate(
            violations.duplicate_dispatch
        ),
        "effect_order": violation_rate(violations.effect_order),
        "capacity_overload": violation_rate(
            violations.capacity_overload
        ),
        "aggregate": violation_rate(violations.total),
    }
    return RuntimeDiagnostics(
        task_count=len(items),
        task_opportunity_rate=(
            0.0
            if not items
            else sum(
                item.runtime.task_has_opportunity for item in items
            )
            / len(items)
        ),
        analyzed_data_edges=analyzed_edges,
        partial_edges=partial_edges,
        partial_edge_ratio=(
            0.0 if analyzed_edges == 0 else partial_edges / analyzed_edges
        ),
        readiness_p50=percentile(gaps, 0.50),
        readiness_p90=percentile(gaps, 0.90),
        readiness_p95=percentile(gaps, 0.95),
        first_guard_ready_item_mean_seconds=(
            None if not first_items else _mean(first_items)
        ),
        dispatched_calls=dispatched_calls,
        early_dispatched_calls=early_calls,
        violation_rates=rates,
    )
