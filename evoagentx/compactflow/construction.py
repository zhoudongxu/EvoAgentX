"""Construction-plane orchestration, Pareto evidence, and policy admission."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .models import (
    AdmissionVerdict,
    CompactnessPolicy,
    Evidence,
    PolicyStatus,
)
from .policy import (
    ApplicabilityFn,
    CompatibilitySelector,
    ConflictChecker,
    PolicyLibrary,
    PolicyMatch,
    PolicyRetriever,
    RetrievalConfig,
    SelectionConfig,
    SelectionResult,
)


@dataclass(frozen=True, slots=True)
class ParetoConfig:
    """Comparison settings for valid quality-efficiency frontiers."""

    dominance_epsilon: float = 1e-12
    require_valid: bool = True

    def __post_init__(self) -> None:
        if self.dominance_epsilon < 0 or not math.isfinite(
            self.dominance_epsilon
        ):
            raise ValueError("dominance_epsilon must be finite and non-negative")


class ParetoArchive:
    """Execution evidence archive with task-aware Pareto frontier queries.

    Quality is maximized; token, latency, and graph costs are minimized.
    Evidence remains in the archive even when dominated so future cost
    normalizers and diagnostics can be replayed.
    """

    def __init__(
        self,
        evidence: Iterable[Evidence] = (),
        *,
        config: ParetoConfig | None = None,
    ) -> None:
        self.config = config or ParetoConfig()
        self._evidence: dict[str, Evidence] = {}
        self.extend(evidence)

    def __len__(self) -> int:
        return len(self._evidence)

    def add(self, evidence: Evidence) -> None:
        """Record one execution, rejecting an identifier collision."""

        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence instance")
        previous = self._evidence.get(evidence.id)
        if previous is not None and previous.to_dict() != evidence.to_dict():
            raise ValueError(f"evidence {evidence.id!r} already exists")
        self._evidence[evidence.id] = evidence

    record = add

    def extend(self, evidence: Iterable[Evidence]) -> None:
        """Record multiple execution evidence items."""

        for item in evidence:
            self.add(item)

    def get(self, evidence_id: str) -> Evidence | None:
        """Return evidence by identifier."""

        return self._evidence.get(evidence_id)

    def all(self) -> tuple[Evidence, ...]:
        """Return all evidence in deterministic identifier order."""

        return tuple(self._evidence[key] for key in sorted(self._evidence))

    def query(
        self,
        *,
        benchmark: str | None = None,
        task_id: str | None = None,
        split: str | None = None,
        variant: str | None = None,
        policy_id: str | None = None,
        valid: bool | None = None,
    ) -> tuple[Evidence, ...]:
        """Filter evolution evidence without discarding insertion history."""

        return tuple(
            item
            for item in self.all()
            if (benchmark is None or item.benchmark == benchmark)
            and (task_id is None or item.task_id == task_id)
            and (split is None or item.split == split)
            and (variant is None or item.variant == variant)
            and (policy_id is None or policy_id in item.policy_ids)
            and (valid is None or item.feedback.valid is valid)
        )

    def dominates(self, left: Evidence, right: Evidence) -> bool:
        """Return whether ``left`` weakly improves every objective and one strictly."""

        epsilon = self.config.dominance_epsilon
        left_metrics = (
            left.feedback.quality,
            -left.feedback.token_cost,
            -left.feedback.latency,
            -left.feedback.total_graph_cost,
        )
        right_metrics = (
            right.feedback.quality,
            -right.feedback.token_cost,
            -right.feedback.latency,
            -right.feedback.total_graph_cost,
        )
        weakly_better = all(
            left_value >= right_value - epsilon
            for left_value, right_value in zip(left_metrics, right_metrics)
        )
        strictly_better = any(
            left_value > right_value + epsilon
            for left_value, right_value in zip(left_metrics, right_metrics)
        )
        return weakly_better and strictly_better

    def frontier(
        self,
        *,
        benchmark: str | None = None,
        task_id: str | None = None,
        split: str | None = None,
        variant: str | None = None,
    ) -> tuple[Evidence, ...]:
        """Return the non-dominated frontier for the selected comparable runs."""

        candidates = list(
            self.query(
                benchmark=benchmark,
                task_id=task_id,
                split=split,
                variant=variant,
                valid=True if self.config.require_valid else None,
            )
        )
        result = [
            candidate
            for candidate in candidates
            if not any(
                other.id != candidate.id and self.dominates(other, candidate)
                for other in candidates
            )
        ]
        return tuple(sorted(result, key=lambda item: item.id))

    def frontiers(
        self,
        *,
        benchmark: str | None = None,
        split: str | None = None,
        variant: str | None = None,
    ) -> dict[tuple[str, str], tuple[Evidence, ...]]:
        """Return one Pareto frontier per benchmark/task identifier."""

        task_keys = {
            (item.benchmark, item.task_id)
            for item in self.query(
                benchmark=benchmark,
                split=split,
                variant=variant,
            )
        }
        return {
            task_key: self.frontier(
                benchmark=task_key[0],
                task_id=task_key[1],
                split=split,
                variant=variant,
            )
            for task_key in sorted(task_keys)
        }


@dataclass(frozen=True, slots=True)
class PairedExecution:
    """Baseline and candidate runs sharing a held-out task and random seed."""

    baseline: Evidence
    candidate: Evidence

    @property
    def pair_key(self) -> tuple[str, str, str, int | str | None]:
        """Return the common benchmark/split/task/seed key."""

        return self.baseline.pair_key


def pair_evidence(
    baseline: Iterable[Evidence],
    candidate: Iterable[Evidence],
    *,
    strict: bool = True,
) -> tuple[PairedExecution, ...]:
    """Match held-out baseline and candidate evidence by task and seed.

    Args:
        baseline: Runs without the candidate policy.
        candidate: Runs with the candidate policy.
        strict: Raise when either side has missing or duplicate pair keys.
    """

    def index(items: Iterable[Evidence], label: str) -> dict[tuple, Evidence]:
        result: dict[tuple, Evidence] = {}
        for item in items:
            key = item.pair_key
            if key in result:
                raise ValueError(f"duplicate {label} pair key: {key!r}")
            result[key] = item
        return result

    baseline_by_key = index(baseline, "baseline")
    candidate_by_key = index(candidate, "candidate")
    if strict and baseline_by_key.keys() != candidate_by_key.keys():
        missing_candidate = sorted(
            map(str, baseline_by_key.keys() - candidate_by_key.keys())
        )
        missing_baseline = sorted(
            map(str, candidate_by_key.keys() - baseline_by_key.keys())
        )
        raise ValueError(
            "held-out evidence keys do not match; "
            f"missing candidate={missing_candidate}, "
            f"missing baseline={missing_baseline}"
        )
    common = baseline_by_key.keys() & candidate_by_key.keys()
    return tuple(
        PairedExecution(baseline_by_key[key], candidate_by_key[key])
        for key in sorted(common, key=repr)
    )


@dataclass(frozen=True, slots=True)
class CostWeights:
    """Weights for normalized token, latency, and graph cost reduction."""

    token: float = 1.0 / 3.0
    latency: float = 1.0 / 3.0
    graph: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        values = (self.token, self.latency, self.graph)
        if any(value < 0 or not math.isfinite(value) for value in values):
            raise ValueError("cost weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one cost weight must be positive")

    @property
    def normalized(self) -> tuple[float, float, float]:
        """Return weights normalized to sum to one."""

        total = self.token + self.latency + self.graph
        return self.token / total, self.latency / total, self.graph / total


@dataclass(frozen=True, slots=True)
class AdmissionConfig:
    """All thresholds used by deterministic paired policy admission."""

    quality_tolerance: float = 0.0
    minimum_cost_reduction: float = 0.0
    merge_similarity_threshold: float = 0.90
    minimum_candidate_valid_rate: float = 1.0
    normalization_floor: float = 1e-9
    cost_weights: CostWeights = field(default_factory=CostWeights)
    require_matching_pair_keys: bool = True
    require_valid_baseline: bool = True

    def __post_init__(self) -> None:
        if self.quality_tolerance < 0 or not math.isfinite(
            self.quality_tolerance
        ):
            raise ValueError("quality_tolerance must be finite and non-negative")
        if not math.isfinite(self.minimum_cost_reduction):
            raise ValueError("minimum_cost_reduction must be finite")
        if not -1.0 <= self.merge_similarity_threshold <= 1.0:
            raise ValueError("merge_similarity_threshold must be in [-1, 1]")
        if not 0.0 <= self.minimum_candidate_valid_rate <= 1.0:
            raise ValueError("minimum_candidate_valid_rate must be in [0, 1]")
        if self.normalization_floor <= 0 or not math.isfinite(
            self.normalization_floor
        ):
            raise ValueError("normalization_floor must be finite and positive")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Auditable Admit/Merge/Reject verdict and its paired measurements."""

    verdict: AdmissionVerdict
    delta_quality: float
    delta_cost: float
    candidate_valid_rate: float
    pair_count: int
    nearest_policy_id: str | None = None
    nearest_similarity: float = 0.0
    reason: str = ""


class PairedPolicyAdmission:
    """Apply the paper's deterministic paired held-out admission rule."""

    def __init__(
        self,
        library: PolicyLibrary,
        *,
        config: AdmissionConfig | None = None,
    ) -> None:
        self.library = library
        self.config = config or AdmissionConfig()

    def _relative_reduction(self, baseline: float, candidate: float) -> float:
        scale = max(abs(baseline), self.config.normalization_floor)
        return (baseline - candidate) / scale

    def _cost_reduction(self, pair: PairedExecution) -> float:
        baseline = pair.baseline.feedback
        candidate = pair.candidate.feedback
        component_reductions = (
            self._relative_reduction(
                baseline.token_cost, candidate.token_cost
            ),
            self._relative_reduction(baseline.latency, candidate.latency),
            self._relative_reduction(
                baseline.total_graph_cost, candidate.total_graph_cost
            ),
        )
        return sum(
            weight * reduction
            for weight, reduction in zip(
                self.config.cost_weights.normalized, component_reductions
            )
        )

    def evaluate(
        self,
        candidate_policy: CompactnessPolicy,
        pairs: Iterable[PairedExecution],
    ) -> AdmissionDecision:
        """Evaluate quality preservation, cost reduction, and near duplication."""

        paired = tuple(pairs)
        if not paired:
            raise ValueError("at least one paired held-out execution is required")
        if self.config.require_matching_pair_keys:
            for pair in paired:
                if pair.baseline.pair_key != pair.candidate.pair_key:
                    raise ValueError(
                        "baseline and candidate task/seed keys must match"
                    )
        if self.config.require_valid_baseline and any(
            not pair.baseline.feedback.valid for pair in paired
        ):
            raise ValueError("paired admission requires valid baseline executions")

        delta_quality = sum(
            pair.candidate.feedback.quality - pair.baseline.feedback.quality
            for pair in paired
        ) / len(paired)
        delta_cost = sum(self._cost_reduction(pair) for pair in paired) / len(
            paired
        )
        candidate_valid_rate = sum(
            pair.candidate.feedback.valid for pair in paired
        ) / len(paired)

        existing_same_id = self.library.get(candidate_policy.id)
        if existing_same_id is not None:
            nearest, similarity = existing_same_id, 1.0
        else:
            nearest, similarity = self.library.nearest(candidate_policy)

        quality_ok = delta_quality >= -self.config.quality_tolerance
        cost_ok = delta_cost >= self.config.minimum_cost_reduction
        validity_ok = (
            candidate_valid_rate >= self.config.minimum_candidate_valid_rate
        )
        if not validity_ok:
            verdict = AdmissionVerdict.REJECT
            reason = (
                f"candidate valid rate {candidate_valid_rate:.6g} is below "
                f"{self.config.minimum_candidate_valid_rate:.6g}"
            )
        elif not quality_ok:
            verdict = AdmissionVerdict.REJECT
            reason = (
                f"quality change {delta_quality:.6g} is below allowed "
                f"{-self.config.quality_tolerance:.6g}"
            )
        elif not cost_ok:
            verdict = AdmissionVerdict.REJECT
            reason = (
                f"cost reduction {delta_cost:.6g} is below "
                f"{self.config.minimum_cost_reduction:.6g}"
            )
        elif (
            nearest is not None
            and similarity >= self.config.merge_similarity_threshold
        ):
            verdict = AdmissionVerdict.MERGE
            reason = (
                f"quality/cost thresholds passed; nearest similarity "
                f"{similarity:.6g} triggers merge"
            )
        else:
            verdict = AdmissionVerdict.ADMIT
            reason = "quality, cost, validity, and novelty thresholds passed"

        return AdmissionDecision(
            verdict=verdict,
            delta_quality=delta_quality,
            delta_cost=delta_cost,
            candidate_valid_rate=candidate_valid_rate,
            pair_count=len(paired),
            nearest_policy_id=nearest.id if nearest is not None else None,
            nearest_similarity=similarity,
            reason=reason,
        )

    def apply(
        self,
        candidate_policy: CompactnessPolicy,
        decision: AdmissionDecision,
    ) -> CompactnessPolicy | None:
        """Apply an evaluated verdict to the policy library."""

        if decision.verdict is AdmissionVerdict.REJECT:
            candidate_policy.status = PolicyStatus.REJECTED
            return None
        candidate_policy.status = PolicyStatus.VERIFIED
        if decision.verdict is AdmissionVerdict.ADMIT:
            self.library.add(candidate_policy)
            return candidate_policy
        if decision.nearest_policy_id is None:
            raise ValueError("a merge decision requires a nearest policy")
        return self.library.merge(
            decision.nearest_policy_id,
            candidate_policy,
            status=PolicyStatus.VERIFIED,
        )


@dataclass(frozen=True, slots=True)
class ConstructionConfig:
    """Grouped configurable thresholds for the construction policy pipeline."""

    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    admission: AdmissionConfig = field(default_factory=AdmissionConfig)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)


@dataclass(frozen=True, slots=True)
class PolicyGuidance:
    """Policy candidates and compatible subset supplied to a planner."""

    candidates: tuple[PolicyMatch, ...]
    selection: SelectionResult

    @property
    def selected_policies(self) -> tuple[CompactnessPolicy, ...]:
        """Return compatible policies in planner priority order."""

        return self.selection.policies


class ConstructionPlane:
    """Facade for retrieval, selection, evidence, and policy evolution.

    Workflow planning remains pluggable: callers pass ``selected_policies`` to
    their EvoAgentX planner, execute the graph, and record resulting
    :class:`Evidence` through this facade.
    """

    def __init__(
        self,
        library: PolicyLibrary,
        *,
        config: ConstructionConfig | None = None,
        applicability: ApplicabilityFn | None = None,
        conflict_checker: ConflictChecker | None = None,
        archive: ParetoArchive | None = None,
    ) -> None:
        self.library = library
        self.config = config or ConstructionConfig()
        self.retriever = PolicyRetriever(
            library,
            config=self.config.retrieval,
            applicability=applicability,
        )
        self.selector = CompatibilitySelector(
            self.config.selection, conflict_checker=conflict_checker
        )
        self.archive = archive or ParetoArchive(config=self.config.pareto)
        self.admission = PairedPolicyAdmission(
            library, config=self.config.admission
        )

    def prepare(
        self, query: str, context: Mapping[str, Any]
    ) -> PolicyGuidance:
        """Retrieve and compatibly select policies for one planner request."""

        candidates = self.retriever.retrieve(query, context)
        return PolicyGuidance(candidates, self.selector.select(candidates))

    def record_evidence(
        self,
        evidence: Evidence,
        *,
        policy_id: str | None = None,
        positive: bool | None = None,
    ) -> None:
        """Record one execution in both the evolution and persistent archives."""

        self.archive.add(evidence)
        self.library.record_evidence(
            evidence, policy_id=policy_id, positive=positive
        )

    def comparable_frontier(
        self,
        task_id: str,
        *,
        benchmark: str | None = None,
        split: str | None = None,
    ) -> tuple[Evidence, ...]:
        """Return valid Pareto evidence eligible for task-level distillation."""

        return self.archive.frontier(
            benchmark=benchmark,
            task_id=task_id,
            split=split,
        )

    def evolve(
        self,
        candidate_policy: CompactnessPolicy,
        pairs: Iterable[PairedExecution],
        *,
        persist: bool = False,
    ) -> AdmissionDecision:
        """Record paired evidence, evaluate, and apply one candidate policy."""

        paired = tuple(pairs)
        decision = self.admission.evaluate(candidate_policy, paired)
        candidate_evidence = tuple(pair.candidate for pair in paired)
        baseline_evidence = tuple(pair.baseline for pair in paired)
        for item in (*baseline_evidence, *candidate_evidence):
            self.record_evidence(item)

        evidence_ids = tuple(item.id for item in candidate_evidence)
        if decision.verdict is AdmissionVerdict.REJECT:
            candidate_policy.negative_evidence_ids = tuple(
                dict.fromkeys(
                    (*candidate_policy.negative_evidence_ids, *evidence_ids)
                )
            )
        else:
            candidate_policy.evidence_ids = tuple(
                dict.fromkeys((*candidate_policy.evidence_ids, *evidence_ids))
            )
        self.admission.apply(candidate_policy, decision)
        if persist:
            self.library.save()
        return decision
