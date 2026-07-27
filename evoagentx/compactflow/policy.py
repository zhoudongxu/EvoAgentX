"""Policy storage, deterministic retrieval, and compatibility selection."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from .models import CompactnessPolicy, Evidence, PolicyStatus


@runtime_checkable
class TextEmbedder(Protocol):
    """Protocol for a pluggable text embedding implementation."""

    def embed(self, text: str) -> Sequence[float]:
        """Return one finite numeric vector for ``text``."""


EmbedderLike = TextEmbedder | Callable[[str], Sequence[float]]
ApplicabilityFn = Callable[[CompactnessPolicy, Mapping[str, Any]], float]
ConflictChecker = Callable[
    [CompactnessPolicy, CompactnessPolicy], bool | str | None
]


class DeterministicTextEmbedder:
    """Dependency-free hashing-vector embedder.

    This embedder is designed as a deterministic default and testing fallback,
    not as a replacement for a semantic embedding model.  It hashes normalized
    word unigrams and adjacent bigrams into a fixed-size signed vector and
    L2-normalizes the result.  Python's randomized ``hash`` is deliberately not
    used, so vectors remain stable across processes and machines.
    """

    _TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension

    def embed(self, text: str) -> tuple[float, ...]:
        """Embed text into a deterministic unit vector."""

        normalized = unicodedata.normalize("NFKC", text).casefold()
        tokens = self._TOKEN_RE.findall(normalized)
        features = list(tokens)
        features.extend(
            f"{left}\u241f{right}" for left, right in pairwise(tokens)
        )
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=16, person=b"compactflow"
            ).digest()
            index = int.from_bytes(digest[:8], "big") % self.dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)


def _embed(embedder: EmbedderLike, text: str) -> tuple[float, ...]:
    values = embedder.embed(text) if hasattr(embedder, "embed") else embedder(text)
    vector = tuple(float(value) for value in values)
    if not vector:
        raise ValueError("embedder returned an empty vector")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("embedder returned a non-finite vector")
    return vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity, returning zero for a zero vector."""

    if len(left) != len(right):
        raise ValueError(
            f"embedding dimensions differ: {len(left)} != {len(right)}"
        )
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _value_match(expected: Any, actual: Any) -> float:
    """Return a deterministic structured precondition match in ``[0, 1]``."""

    if isinstance(expected, Mapping):
        operators = set(expected) & {
            "$eq",
            "$in",
            "$contains",
            "$all",
            "$gte",
            "$lte",
            "$exists",
        }
        if operators:
            results: list[bool] = []
            if "$exists" in expected:
                results.append((actual is not None) is bool(expected["$exists"]))
            if "$eq" in expected:
                results.append(actual == expected["$eq"])
            if "$in" in expected:
                try:
                    results.append(actual in expected["$in"])
                except TypeError:
                    results.append(False)
            if "$contains" in expected:
                try:
                    results.append(expected["$contains"] in actual)
                except TypeError:
                    results.append(False)
            if "$all" in expected:
                try:
                    results.append(set(expected["$all"]).issubset(set(actual)))
                except TypeError:
                    results.append(False)
            if "$gte" in expected:
                try:
                    results.append(actual >= expected["$gte"])
                except TypeError:
                    results.append(False)
            if "$lte" in expected:
                try:
                    results.append(actual <= expected["$lte"])
                except TypeError:
                    results.append(False)
            return sum(results) / len(results) if results else 0.0
        if not isinstance(actual, Mapping):
            return 0.0
        if not expected:
            return 1.0
        return sum(
            _value_match(value, actual.get(key))
            for key, value in expected.items()
        ) / len(expected)
    if isinstance(expected, (list, tuple, set, frozenset)):
        if not expected:
            return 1.0
        if not isinstance(actual, (list, tuple, set, frozenset)):
            return 0.0
        return sum(
            any(value == actual_value for actual_value in actual)
            for value in expected
        ) / len(expected)
    return 1.0 if expected == actual else 0.0


def default_applicability(
    policy: CompactnessPolicy, context: Mapping[str, Any]
) -> float:
    """Score a policy precondition against task/tool structural context.

    Plain nested mappings are matched recursively.  Required list values are
    treated as a subset requirement.  The optional operators ``$eq``, ``$in``,
    ``$contains``, ``$all``, ``$gte``, ``$lte``, and ``$exists`` support
    explicit predicates while remaining JSON serializable.
    """

    if not policy.precondition:
        return 1.0
    return _value_match(policy.precondition, context)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Configurable semantic prefilter and multi-signal reranking settings.

    Reusable planning defaults to policies that passed held-out verification.
    Candidate and rejected policies are available only through explicit
    exploration flags.
    """

    semantic_top_k0: int = 20
    top_k: int = 5
    semantic_weight: float = 0.55
    applicability_weight: float = 0.25
    utility_weight: float = 0.10
    confidence_weight: float = 0.10
    minimum_semantic_score: float = -1.0
    minimum_applicability: float = 0.0
    include_candidates: bool = False
    include_rejected: bool = False

    def __post_init__(self) -> None:
        if self.semantic_top_k0 < 1 or self.top_k < 1:
            raise ValueError("semantic_top_k0 and top_k must be positive")
        if self.semantic_top_k0 < self.top_k:
            raise ValueError("semantic_top_k0 must be greater than or equal to top_k")
        weights = (
            self.semantic_weight,
            self.applicability_weight,
            self.utility_weight,
            self.confidence_weight,
        )
        if any(weight < 0 or not math.isfinite(weight) for weight in weights):
            raise ValueError("retrieval weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one retrieval weight must be positive")
        if not -1.0 <= self.minimum_semantic_score <= 1.0:
            raise ValueError("minimum_semantic_score must be in [-1, 1]")
        if not 0.0 <= self.minimum_applicability <= 1.0:
            raise ValueError("minimum_applicability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PolicyMatch:
    """A retrieved policy with decomposed ranking scores."""

    policy: CompactnessPolicy
    semantic_score: float
    applicability_score: float
    total_score: float


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Thresholds for compatibility-aware greedy policy selection."""

    max_policies: int = 3
    minimum_score: float = 0.0
    minimum_applicability: float = 0.5

    def __post_init__(self) -> None:
        if self.max_policies < 1:
            raise ValueError("max_policies must be positive")
        if not math.isfinite(self.minimum_score):
            raise ValueError("minimum_score must be finite")
        if not 0.0 <= self.minimum_applicability <= 1.0:
            raise ValueError("minimum_applicability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SkippedPolicy:
    """A candidate omitted during compatibility-aware selection."""

    policy_id: str
    reason: str
    conflicts_with: str | None = None


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Ordered compatible policy matches and explanations for skipped items."""

    selected: tuple[PolicyMatch, ...]
    skipped: tuple[SkippedPolicy, ...]

    @property
    def policies(self) -> tuple[CompactnessPolicy, ...]:
        """Return only selected policy objects in planner priority order."""

        return tuple(match.policy for match in self.selected)


class PolicyLibrary:
    """Thread-safe in-memory policy/evidence store with atomic JSON persistence."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        policies: Iterable[CompactnessPolicy] = (),
        *,
        evidence: Iterable[Evidence] = (),
        path: str | Path | None = None,
        embedder: EmbedderLike | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.embedder: EmbedderLike = embedder or DeterministicTextEmbedder()
        self._policies: dict[str, CompactnessPolicy] = {}
        self._evidence: dict[str, Evidence] = {}
        self._lock = RLock()
        for policy in policies:
            self.add(policy)
        for item in evidence:
            self.record_evidence(item)

    def __len__(self) -> int:
        return len(self._policies)

    def __iter__(self) -> Iterator[CompactnessPolicy]:
        return iter(self.all())

    def all(self) -> tuple[CompactnessPolicy, ...]:
        """Return policies in deterministic identifier order."""

        with self._lock:
            return tuple(self._policies[key] for key in sorted(self._policies))

    def get(self, policy_id: str) -> CompactnessPolicy | None:
        """Return a policy by identifier, or ``None`` when absent."""

        with self._lock:
            return self._policies.get(policy_id)

    def require(self, policy_id: str) -> CompactnessPolicy:
        """Return a policy by identifier or raise ``KeyError``."""

        policy = self.get(policy_id)
        if policy is None:
            raise KeyError(policy_id)
        return policy

    def add(self, policy: CompactnessPolicy, *, replace: bool = False) -> None:
        """Add a policy, optionally replacing an existing identifier."""

        if not isinstance(policy, CompactnessPolicy):
            raise TypeError("policy must be a CompactnessPolicy")
        with self._lock:
            if policy.id in self._policies and not replace:
                raise ValueError(f"policy {policy.id!r} already exists")
            self._policies[policy.id] = policy

    def remove(self, policy_id: str) -> CompactnessPolicy:
        """Remove and return one policy."""

        with self._lock:
            return self._policies.pop(policy_id)

    def record_evidence(
        self,
        evidence: Evidence,
        *,
        policy_id: str | None = None,
        positive: bool | None = None,
    ) -> None:
        """Store execution evidence and optionally attach it to one policy.

        ``positive`` must be provided when ``policy_id`` is provided so callers
        explicitly distinguish supporting and negative evidence.
        """

        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence instance")
        if policy_id is not None and positive is None:
            raise ValueError("positive is required when attaching evidence")
        with self._lock:
            previous = self._evidence.get(evidence.id)
            if previous is not None and previous.to_dict() != evidence.to_dict():
                raise ValueError(f"evidence {evidence.id!r} already exists")
            self._evidence[evidence.id] = evidence
            if policy_id is not None:
                self.require(policy_id).with_evidence(
                    evidence, positive=bool(positive)
                )

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Return evidence by identifier, or ``None`` when absent."""

        with self._lock:
            return self._evidence.get(evidence_id)

    def all_evidence(self) -> tuple[Evidence, ...]:
        """Return all evidence in deterministic identifier order."""

        with self._lock:
            return tuple(self._evidence[key] for key in sorted(self._evidence))

    def evidence_for(
        self, policy_id: str, *, positive: bool | None = None
    ) -> tuple[Evidence, ...]:
        """Return positive, negative, or all attached evidence for a policy."""

        policy = self.require(policy_id)
        if positive is True:
            ids = policy.evidence_ids
        elif positive is False:
            ids = policy.negative_evidence_ids
        else:
            ids = tuple(
                dict.fromkeys(
                    (*policy.evidence_ids, *policy.negative_evidence_ids)
                )
            )
        return tuple(
            item
            for evidence_id in ids
            if (item := self.get_evidence(evidence_id)) is not None
        )

    def merge(
        self,
        existing_id: str,
        candidate: CompactnessPolicy,
        *,
        status: PolicyStatus = PolicyStatus.VERIFIED,
    ) -> CompactnessPolicy:
        """Merge candidate evidence/statistics into an existing policy.

        The existing rule's description, precondition, and operation remain
        authoritative.  This avoids silently changing verified semantics.
        Evidence identifiers, conflicts, metadata, and expected effects are
        accumulated; utility and confidence are evidence-count-weighted.
        """

        with self._lock:
            existing = self.require(existing_id)
            old_weight = max(
                1,
                len(existing.evidence_ids)
                + len(existing.negative_evidence_ids),
            )
            new_weight = max(
                1,
                len(candidate.evidence_ids)
                + len(candidate.negative_evidence_ids),
            )
            denominator = old_weight + new_weight
            existing.utility = (
                existing.utility * old_weight + candidate.utility * new_weight
            ) / denominator
            existing.confidence = (
                existing.confidence * old_weight
                + candidate.confidence * new_weight
            ) / denominator
            existing.evidence_ids = tuple(
                dict.fromkeys((*existing.evidence_ids, *candidate.evidence_ids))
            )
            existing.negative_evidence_ids = tuple(
                dict.fromkeys(
                    (
                        *existing.negative_evidence_ids,
                        *candidate.negative_evidence_ids,
                    )
                )
            )
            existing.conflicts_with = tuple(
                dict.fromkeys(
                    (*existing.conflicts_with, *candidate.conflicts_with)
                )
            )
            existing.expected_effect.update(candidate.expected_effect)
            existing.metadata.update(candidate.metadata)
            existing.status = status
            existing.version += 1
            from datetime import datetime, timezone

            existing.updated_at = datetime.now(timezone.utc).isoformat()
            return existing

    def retrieve(
        self,
        query: str,
        context: Mapping[str, Any] | None = None,
        *,
        config: RetrievalConfig | None = None,
        applicability: ApplicabilityFn | None = None,
    ) -> tuple[PolicyMatch, ...]:
        """Convenience wrapper around :class:`PolicyRetriever`."""

        return PolicyRetriever(
            self,
            embedder=self.embedder,
            config=config,
            applicability=applicability,
        ).retrieve(query, context or {})

    def nearest(
        self,
        policy: CompactnessPolicy,
        *,
        include_rejected: bool = False,
    ) -> tuple[CompactnessPolicy | None, float]:
        """Return the nearest policy description and cosine similarity."""

        query_vector = _embed(self.embedder, policy.retrieval_text())
        best: CompactnessPolicy | None = None
        best_score = -1.0
        for existing in self.all():
            if existing.id == policy.id:
                continue
            if not include_rejected and existing.status is PolicyStatus.REJECTED:
                continue
            score = cosine_similarity(
                query_vector,
                _embed(self.embedder, existing.retrieval_text()),
            )
            if best is None or score > best_score or (
                score == best_score
                and existing.id < best.id
            ):
                best, best_score = existing, score
        return best, best_score if best is not None else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-serializable library payload."""

        return {
            "schema_version": self.SCHEMA_VERSION,
            "policies": [policy.to_dict() for policy in self.all()],
            "evidence": [item.to_dict() for item in self.all_evidence()],
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Atomically persist policies and evidence as UTF-8 JSON."""

        destination = Path(path) if path is not None else self.path
        if destination is None:
            raise ValueError("a path is required to save the policy library")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(destination)
        self.path = destination
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embedder: EmbedderLike | None = None,
    ) -> PolicyLibrary:
        """Load a policy library from JSON, rejecting unknown future schemas."""

        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        version = int(payload.get("schema_version", 0))
        if version != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported policy library schema {version}; "
                f"expected {cls.SCHEMA_VERSION}"
            )
        return cls(
            (CompactnessPolicy.from_dict(item) for item in payload["policies"]),
            evidence=(
                Evidence.from_dict(item) for item in payload.get("evidence", ())
            ),
            path=source,
            embedder=embedder,
        )


class PolicyRetriever:
    """Two-stage semantic retrieval and applicability-aware reranking."""

    def __init__(
        self,
        library: PolicyLibrary,
        *,
        embedder: EmbedderLike | None = None,
        config: RetrievalConfig | None = None,
        applicability: ApplicabilityFn | None = None,
    ) -> None:
        self.library = library
        self.embedder = embedder or library.embedder
        self.config = config or RetrievalConfig()
        self.applicability = applicability or default_applicability
        self._embedding_cache: dict[
            tuple[str, int, str], tuple[float, ...]
        ] = {}

    def _policy_vector(self, policy: CompactnessPolicy) -> tuple[float, ...]:
        cache_key = policy.id, policy.version, policy.retrieval_text()
        vector = self._embedding_cache.get(cache_key)
        if vector is None:
            vector = _embed(self.embedder, policy.retrieval_text())
            self._embedding_cache[cache_key] = vector
        return vector

    def retrieve(
        self, query: str, context: Mapping[str, Any]
    ) -> tuple[PolicyMatch, ...]:
        """Retrieve semantic Top-K0 and rerank the final Top-K deterministically."""

        query_vector = _embed(self.embedder, query)
        semantic_candidates: list[tuple[CompactnessPolicy, float]] = []
        for policy in self.library.all():
            if (
                policy.status is PolicyStatus.CANDIDATE
                and not self.config.include_candidates
            ):
                continue
            if (
                policy.status is PolicyStatus.REJECTED
                and not self.config.include_rejected
            ):
                continue
            semantic_score = cosine_similarity(
                query_vector, self._policy_vector(policy)
            )
            if semantic_score >= self.config.minimum_semantic_score:
                semantic_candidates.append((policy, semantic_score))
        semantic_candidates.sort(key=lambda item: (-item[1], item[0].id))
        semantic_candidates = semantic_candidates[: self.config.semantic_top_k0]

        weight_sum = (
            self.config.semantic_weight
            + self.config.applicability_weight
            + self.config.utility_weight
            + self.config.confidence_weight
        )
        matches: list[PolicyMatch] = []
        for policy, semantic_score in semantic_candidates:
            applicability_score = float(self.applicability(policy, context))
            if not math.isfinite(applicability_score):
                raise ValueError("applicability scorer returned a non-finite value")
            applicability_score = max(0.0, min(1.0, applicability_score))
            if applicability_score < self.config.minimum_applicability:
                continue
            total_score = (
                self.config.semantic_weight * semantic_score
                + self.config.applicability_weight * applicability_score
                + self.config.utility_weight * policy.utility
                + self.config.confidence_weight * policy.confidence
            ) / weight_sum
            matches.append(
                PolicyMatch(
                    policy=policy,
                    semantic_score=semantic_score,
                    applicability_score=applicability_score,
                    total_score=total_score,
                )
            )
        matches.sort(key=lambda item: (-item.total_score, item.policy.id))
        return tuple(matches[: self.config.top_k])


class CompatibilitySelector:
    """Greedily select an ordered, applicable, mutually compatible prefix."""

    def __init__(
        self,
        config: SelectionConfig | None = None,
        *,
        conflict_checker: ConflictChecker | None = None,
    ) -> None:
        self.config = config or SelectionConfig()
        self.conflict_checker = conflict_checker

    def _conflict_reason(
        self, left: CompactnessPolicy, right: CompactnessPolicy
    ) -> str | None:
        if (
            right.id in left.conflicts_with
            or left.id in right.conflicts_with
        ):
            return "explicit policy conflict"
        left_group = left.metadata.get("exclusive_group")
        right_group = right.metadata.get("exclusive_group")
        if left_group is not None and left_group == right_group:
            return f"exclusive group {left_group!r}"
        if self.conflict_checker is not None:
            result = self.conflict_checker(left, right)
            if isinstance(result, str) and result:
                return result
            if result:
                return "custom conflict"
        return None

    def select(self, matches: Iterable[PolicyMatch]) -> SelectionResult:
        """Select policies by score while explaining every skipped candidate."""

        ordered = sorted(
            matches, key=lambda item: (-item.total_score, item.policy.id)
        )
        selected: list[PolicyMatch] = []
        skipped: list[SkippedPolicy] = []
        for match in ordered:
            if len(selected) >= self.config.max_policies:
                skipped.append(
                    SkippedPolicy(match.policy.id, "selection limit reached")
                )
                continue
            if match.total_score < self.config.minimum_score:
                skipped.append(
                    SkippedPolicy(match.policy.id, "score below threshold")
                )
                continue
            if (
                match.applicability_score
                < self.config.minimum_applicability
            ):
                skipped.append(
                    SkippedPolicy(
                        match.policy.id, "precondition is insufficiently supported"
                    )
                )
                continue
            conflict = next(
                (
                    (chosen.policy.id, reason)
                    for chosen in selected
                    if (
                        reason := self._conflict_reason(
                            match.policy, chosen.policy
                        )
                    )
                ),
                None,
            )
            if conflict is not None:
                skipped.append(
                    SkippedPolicy(
                        match.policy.id,
                        conflict[1],
                        conflicts_with=conflict[0],
                    )
                )
                continue
            selected.append(match)
        return SelectionResult(tuple(selected), tuple(skipped))
