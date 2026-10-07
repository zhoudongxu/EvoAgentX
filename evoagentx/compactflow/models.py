"""Typed data models for CompactFlow's construction plane.

The models deliberately use only the Python standard library.  They are
small, JSON-friendly records that can be used by an LLM-backed planner or by
fully deterministic tests without importing the rest of EvoAgentX.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

JsonObject = dict[str, Any]


class PolicyStatus(str, Enum):
    """Lifecycle state of a compactness policy."""

    CANDIDATE = "candidate"
    VERIFIED = "verified"
    REJECTED = "rejected"


class AdmissionVerdict(str, Enum):
    """Deterministic result of paired held-out policy verification."""

    ADMIT = "admit"
    MERGE = "merge"
    REJECT = "reject"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _json_object(name: str, value: Mapping[str, Any]) -> JsonObject:
    """Validate and canonicalize a nested JSON object."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain only JSON-compatible values") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must be a JSON object")
    return decoded


@dataclass(slots=True)
class ExecutionFeedback:
    """Quality, efficiency, and validity measurements for one execution.

    Args:
        quality: Benchmark-native quality score.  Larger is better.
        token_cost: Total input and output token cost.  Smaller is better.
        latency: End-to-end latency in seconds.  Smaller is better.
        graph_cost: An optional caller-defined normalized workflow structure
            cost. When omitted, node plus edge count is used.
        valid: Whether the execution satisfies its semantic contract.
        node_count: Optional workflow node count.
        edge_count: Optional workflow edge count.
        contract_violations: Runtime contract violations observed in the run.
        metrics: Additional JSON-serializable measurements.
    """

    quality: float
    token_cost: float = 0.0
    latency: float = 0.0
    graph_cost: float | None = None
    valid: bool = True
    node_count: int | None = None
    edge_count: int | None = None
    contract_violations: tuple[str, ...] = ()
    metrics: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("quality", "token_cost", "latency"):
            value = float(getattr(self, name))
            _require_finite(name, value)
            setattr(self, name, value)
        if self.graph_cost is not None:
            self.graph_cost = float(self.graph_cost)
            _require_finite("graph_cost", self.graph_cost)
        for name in ("token_cost", "latency"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.graph_cost is not None and self.graph_cost < 0:
            raise ValueError("graph_cost must be non-negative")
        for name in ("node_count", "edge_count"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when provided")
        self.contract_violations = tuple(self.contract_violations)
        self.metrics = _json_object("metrics", self.metrics)

    @property
    def total_graph_cost(self) -> float:
        """Return the explicit graph cost or a node-plus-edge fallback."""

        if self.graph_cost is not None:
            return self.graph_cost
        return float((self.node_count or 0) + (self.edge_count or 0))

    def to_dict(self) -> JsonObject:
        """Convert this feedback record to a JSON-serializable dictionary."""

        return {
            "quality": self.quality,
            "token_cost": self.token_cost,
            "latency": self.latency,
            "graph_cost": self.graph_cost,
            "valid": self.valid,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "contract_violations": list(self.contract_violations),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionFeedback:
        """Construct feedback from a dictionary produced by :meth:`to_dict`."""

        return cls(
            quality=float(data["quality"]),
            token_cost=float(data.get("token_cost", 0.0)),
            latency=float(data.get("latency", 0.0)),
            graph_cost=(
                None
                if data.get("graph_cost") is None
                else float(data["graph_cost"])
            ),
            valid=bool(data.get("valid", True)),
            node_count=data.get("node_count"),
            edge_count=data.get("edge_count"),
            contract_violations=tuple(data.get("contract_violations", ())),
            metrics=dict(data.get("metrics", {})),
        )


@dataclass(slots=True)
class Evidence:
    """Execution-grounded evidence used to evolve compactness policies.

    ``variant`` normally identifies a ``"baseline"`` or ``"candidate"`` run.
    Paired verification matches evidence by benchmark, split, ``task_id``,
    and ``seed`` rather than relying on insertion order.
    """

    task_id: str
    feedback: ExecutionFeedback
    benchmark: str = ""
    id: str = field(default_factory=lambda: uuid4().hex)
    workflow_id: str = ""
    policy_ids: tuple[str, ...] = ()
    seed: int | str | None = None
    split: str = "train"
    variant: str = "candidate"
    trace_ref: str | None = None
    metadata: JsonObject = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evidence id must not be empty")
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if not isinstance(self.benchmark, str):
            raise TypeError("benchmark must be a string")
        if not isinstance(self.split, str) or not self.split:
            raise ValueError("split must be a non-empty string")
        if not isinstance(self.feedback, ExecutionFeedback):
            raise TypeError("feedback must be an ExecutionFeedback instance")
        self.policy_ids = tuple(dict.fromkeys(self.policy_ids))
        self.metadata = _json_object("metadata", self.metadata)

    @property
    def pair_key(self) -> tuple[str, str, str, int | str | None]:
        """Key used to match baseline and candidate held-out executions."""

        return self.benchmark, self.split, self.task_id, self.seed

    def to_dict(self) -> JsonObject:
        """Convert this evidence record to a JSON-serializable dictionary."""

        return {
            "id": self.id,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "workflow_id": self.workflow_id,
            "policy_ids": list(self.policy_ids),
            "seed": self.seed,
            "split": self.split,
            "variant": self.variant,
            "trace_ref": self.trace_ref,
            "feedback": self.feedback.to_dict(),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Evidence:
        """Construct evidence from a dictionary produced by :meth:`to_dict`."""

        return cls(
            id=str(data["id"]),
            benchmark=str(data.get("benchmark", "")),
            task_id=str(data["task_id"]),
            workflow_id=str(data.get("workflow_id", "")),
            policy_ids=tuple(data.get("policy_ids", ())),
            seed=data.get("seed"),
            split=str(data.get("split", "train")),
            variant=str(data.get("variant", "candidate")),
            trace_ref=data.get("trace_ref"),
            feedback=ExecutionFeedback.from_dict(data["feedback"]),
            metadata=dict(data.get("metadata", {})),
            created_at=str(data.get("created_at", _utc_now())),
        )


@dataclass(slots=True)
class CompactnessPolicy:
    """A reusable, execution-verified workflow construction rule.

    ``precondition`` and ``operation`` are intentionally generic typed JSON
    objects.  This keeps the storage layer independent of a particular
    workflow graph representation while preserving enough structure for
    applicability checks and planners.
    """

    id: str
    description: str
    precondition: JsonObject = field(default_factory=dict)
    operation: JsonObject = field(default_factory=dict)
    expected_effect: JsonObject = field(default_factory=dict)
    utility: float = 0.5
    confidence: float = 0.5
    conflicts_with: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    negative_evidence_ids: tuple[str, ...] = ()
    status: PolicyStatus = PolicyStatus.CANDIDATE
    version: int = 1
    metadata: JsonObject = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("policy id must not be empty")
        if not self.description.strip():
            raise ValueError("policy description must not be empty")
        for name in ("utility", "confidence"):
            value = float(getattr(self, name))
            _require_finite(name, value)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            setattr(self, name, value)
        if self.version < 1:
            raise ValueError("policy version must be at least one")
        if not isinstance(self.status, PolicyStatus):
            self.status = PolicyStatus(self.status)
        self.precondition = _json_object("precondition", self.precondition)
        self.operation = _json_object("operation", self.operation)
        self.expected_effect = _json_object(
            "expected_effect", self.expected_effect
        )
        self.conflicts_with = tuple(dict.fromkeys(self.conflicts_with))
        self.evidence_ids = tuple(dict.fromkeys(self.evidence_ids))
        self.negative_evidence_ids = tuple(
            dict.fromkeys(self.negative_evidence_ids)
        )
        self.metadata = _json_object("metadata", self.metadata)

    def retrieval_text(self) -> str:
        """Return stable text used by the default policy embedder."""

        tags = self.metadata.get("tags", ())
        if isinstance(tags, str):
            tags = (tags,)
        operation_name = self.operation.get("type", self.operation.get("name", ""))
        return " ".join(
            part
            for part in (
                self.description,
                str(operation_name),
                " ".join(str(tag) for tag in tags),
            )
            if part
        )

    def with_evidence(self, evidence: Evidence, *, positive: bool) -> None:
        """Attach an evidence identifier and update the modification time."""

        if positive:
            self.evidence_ids = tuple(
                dict.fromkeys((*self.evidence_ids, evidence.id))
            )
        else:
            self.negative_evidence_ids = tuple(
                dict.fromkeys((*self.negative_evidence_ids, evidence.id))
            )
        self.updated_at = _utc_now()

    def to_dict(self) -> JsonObject:
        """Convert this policy to a JSON-serializable dictionary."""

        return {
            "id": self.id,
            "description": self.description,
            "precondition": dict(self.precondition),
            "operation": dict(self.operation),
            "expected_effect": dict(self.expected_effect),
            "utility": self.utility,
            "confidence": self.confidence,
            "conflicts_with": list(self.conflicts_with),
            "evidence_ids": list(self.evidence_ids),
            "negative_evidence_ids": list(self.negative_evidence_ids),
            "status": self.status.value,
            "version": self.version,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompactnessPolicy:
        """Construct a policy from a dictionary produced by :meth:`to_dict`."""

        return cls(
            id=str(data["id"]),
            description=str(data["description"]),
            precondition=dict(data.get("precondition", {})),
            operation=dict(data.get("operation", {})),
            expected_effect=dict(data.get("expected_effect", {})),
            utility=float(data.get("utility", 0.5)),
            confidence=float(data.get("confidence", 0.5)),
            conflicts_with=tuple(data.get("conflicts_with", ())),
            evidence_ids=tuple(data.get("evidence_ids", ())),
            negative_evidence_ids=tuple(data.get("negative_evidence_ids", ())),
            status=PolicyStatus(data.get("status", PolicyStatus.CANDIDATE.value)),
            version=int(data.get("version", 1)),
            metadata=dict(data.get("metadata", {})),
            # Released seed policies from schema v1 predate timestamp fields.
            # Use a stable epoch marker when those fields are absent so loading
            # the same immutable library in a later process has the same digest
            # and can safely pass the evolution runner's resume lock.
            created_at=str(data.get("created_at", "1970-01-01T00:00:00+00:00")),
            updated_at=str(data.get("updated_at", "1970-01-01T00:00:00+00:00")),
        )
