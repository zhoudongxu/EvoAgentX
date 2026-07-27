"""Deterministic unit tests for the CompactFlow construction plane."""

from __future__ import annotations

import json

import pytest

from evoagentx.compactflow.construction import (
    AdmissionConfig,
    ConstructionConfig,
    ConstructionPlane,
    CostWeights,
    PairedExecution,
    PairedPolicyAdmission,
    ParetoArchive,
    pair_evidence,
)
from evoagentx.compactflow.models import (
    AdmissionVerdict,
    CompactnessPolicy,
    Evidence,
    ExecutionFeedback,
    PolicyStatus,
)
from evoagentx.compactflow.policy import (
    CompatibilitySelector,
    DeterministicTextEmbedder,
    PolicyLibrary,
    PolicyMatch,
    PolicyRetriever,
    RetrievalConfig,
    SelectionConfig,
    cosine_similarity,
)


def policy(
    policy_id: str,
    description: str,
    *,
    precondition: dict | None = None,
    utility: float = 0.5,
    confidence: float = 0.5,
    conflicts_with: tuple[str, ...] = (),
    metadata: dict | None = None,
) -> CompactnessPolicy:
    return CompactnessPolicy(
        id=policy_id,
        description=description,
        precondition=precondition or {},
        operation={"type": description.split()[0]},
        utility=utility,
        confidence=confidence,
        conflicts_with=conflicts_with,
        metadata=metadata or {},
    )


def evidence(
    evidence_id: str,
    *,
    benchmark: str = "synthetic",
    task_id: str = "task-1",
    seed: int = 1,
    variant: str = "candidate",
    quality: float = 1.0,
    tokens: float = 100.0,
    latency: float = 10.0,
    graph: float = 10.0,
    valid: bool = True,
    policy_ids: tuple[str, ...] = (),
) -> Evidence:
    return Evidence(
        id=evidence_id,
        benchmark=benchmark,
        task_id=task_id,
        seed=seed,
        split="validation",
        variant=variant,
        policy_ids=policy_ids,
        feedback=ExecutionFeedback(
            quality=quality,
            token_cost=tokens,
            latency=latency,
            graph_cost=graph,
            valid=valid,
        ),
    )


class AxisEmbedder:
    """Small controllable embedder used to expose two-stage ranking behavior."""

    def embed(self, text: str) -> tuple[float, float, float]:
        lowered = text.lower()
        return (
            float("fusion" in lowered),
            float("pruning" in lowered),
            float("routing" in lowered),
        )


def held_out_pair(
    *,
    candidate_quality: float = 0.99,
    candidate_tokens: float = 70.0,
    candidate_latency: float = 7.0,
    candidate_graph: float = 7.0,
    candidate_valid: bool = True,
    seed: int = 1,
    suffix: str = "",
) -> PairedExecution:
    return PairedExecution(
        baseline=evidence(
            f"baseline-{seed}{suffix}",
            seed=seed,
            variant="baseline",
            quality=1.0,
            tokens=100.0,
            latency=10.0,
            graph=10.0,
        ),
        candidate=evidence(
            f"candidate-{seed}{suffix}",
            seed=seed,
            variant="candidate",
            quality=candidate_quality,
            tokens=candidate_tokens,
            latency=candidate_latency,
            graph=candidate_graph,
            valid=candidate_valid,
        ),
    )


def test_default_embedder_is_deterministic() -> None:
    first = DeterministicTextEmbedder(64)
    second = DeterministicTextEmbedder(64)

    a = first.embed("Fuse repeated retrieval agents")
    b = second.embed("Fuse repeated retrieval agents")
    other = first.embed("Prune an unused edge")

    assert a == b
    assert cosine_similarity(a, b) == pytest.approx(1.0)
    assert cosine_similarity(a, other) < 1.0


def test_semantic_top_k0_then_applicability_utility_confidence_rerank() -> None:
    fusion = policy(
        "fusion",
        "fusion repeated roles",
        precondition={"tools": ["search"]},
        utility=0.2,
        confidence=0.2,
    )
    pruning = policy(
        "pruning",
        "fusion pruning unused outputs",
        precondition={"tools": ["search"]},
        utility=1.0,
        confidence=1.0,
    )
    routing = policy(
        "routing",
        "routing simplification",
        utility=1.0,
        confidence=1.0,
    )
    for item in (fusion, pruning, routing):
        item.status = PolicyStatus.VERIFIED
    library = PolicyLibrary(
        [fusion, pruning, routing], embedder=AxisEmbedder()
    )
    retriever = PolicyRetriever(
        library,
        config=RetrievalConfig(
            semantic_top_k0=2,
            top_k=2,
            semantic_weight=0.1,
            applicability_weight=0.2,
            utility_weight=0.35,
            confidence_weight=0.35,
        ),
    )

    matches = retriever.retrieve(
        "fusion repeated agents", {"tools": ["search", "summarize"]}
    )

    assert [match.policy.id for match in matches] == ["pruning", "fusion"]
    assert all(match.policy.id != "routing" for match in matches)
    assert all(match.applicability_score == 1.0 for match in matches)
    assert matches[0].total_score > matches[1].total_score


def test_retrieval_requires_verified_policies_unless_exploring() -> None:
    candidate = policy("candidate", "fusion candidate")
    verified = policy("verified", "fusion verified")
    verified.status = PolicyStatus.VERIFIED
    library = PolicyLibrary([candidate, verified], embedder=AxisEmbedder())

    safe_matches = PolicyRetriever(
        library,
        config=RetrievalConfig(semantic_top_k0=2, top_k=2),
    ).retrieve("fusion", {})
    exploration_matches = PolicyRetriever(
        library,
        config=RetrievalConfig(
            semantic_top_k0=2,
            top_k=2,
            include_candidates=True,
        ),
    ).retrieve("fusion", {})

    assert [item.policy.id for item in safe_matches] == ["verified"]
    assert {item.policy.id for item in exploration_matches} == {
        "candidate",
        "verified",
    }


def test_compatibility_selector_skips_explicit_and_group_conflicts() -> None:
    first = policy(
        "first",
        "fusion one",
        conflicts_with=("second",),
        metadata={"exclusive_group": "graph-rewrite"},
    )
    second = policy("second", "fusion two")
    third = policy(
        "third",
        "routing three",
        metadata={"exclusive_group": "graph-rewrite"},
    )
    matches = [
        PolicyMatch(first, 1.0, 1.0, 0.9),
        PolicyMatch(second, 0.9, 1.0, 0.8),
        PolicyMatch(third, 0.8, 1.0, 0.7),
    ]

    result = CompatibilitySelector(
        SelectionConfig(
            max_policies=3,
            minimum_score=0.0,
            minimum_applicability=0.5,
        )
    ).select(matches)

    assert [item.policy.id for item in result.selected] == ["first"]
    skipped = {item.policy_id: item for item in result.skipped}
    assert skipped["second"].conflicts_with == "first"
    assert skipped["third"].conflicts_with == "first"
    assert "explicit" in skipped["second"].reason
    assert "exclusive group" in skipped["third"].reason


def test_paired_admission_admits_and_rejects_from_configured_thresholds() -> None:
    library = PolicyLibrary()
    admission = PairedPolicyAdmission(
        library,
        config=AdmissionConfig(
            quality_tolerance=0.02,
            minimum_cost_reduction=0.20,
            merge_similarity_threshold=0.95,
            cost_weights=CostWeights(token=1.0, latency=1.0, graph=1.0),
        ),
    )
    candidate = policy("candidate", "fusion compatible chain")

    admit = admission.evaluate(candidate, [held_out_pair()])
    assert admit.verdict is AdmissionVerdict.ADMIT
    assert admit.delta_quality == pytest.approx(-0.01)
    assert admit.delta_cost == pytest.approx(0.30)
    assert admission.apply(candidate, admit) is candidate
    assert library.require("candidate").status is PolicyStatus.VERIFIED

    invalid_candidate = policy("invalid", "pruning required verifier")
    reject = admission.evaluate(
        invalid_candidate,
        [
            held_out_pair(
                candidate_quality=1.0,
                candidate_tokens=50.0,
                candidate_valid=False,
                suffix="-invalid",
            )
        ],
    )
    assert reject.verdict is AdmissionVerdict.REJECT
    assert admission.apply(invalid_candidate, reject) is None
    assert library.get("invalid") is None
    assert invalid_candidate.status is PolicyStatus.REJECTED


def test_merge_updates_existing_policy_and_records_evolution_evidence() -> None:
    existing = policy("existing", "fusion repeated chain")
    existing.status = PolicyStatus.VERIFIED
    library = PolicyLibrary([existing])
    plane = ConstructionPlane(
        library,
        config=ConstructionConfig(
            retrieval=RetrievalConfig(semantic_top_k0=2, top_k=1),
            selection=SelectionConfig(max_policies=1),
            admission=AdmissionConfig(
                quality_tolerance=0.02,
                minimum_cost_reduction=0.20,
                merge_similarity_threshold=0.99,
            ),
        ),
    )
    candidate = policy("new-candidate", "fusion repeated chain")
    pair = held_out_pair()

    decision = plane.evolve(candidate, [pair])

    assert decision.verdict is AdmissionVerdict.MERGE
    assert decision.nearest_policy_id == "existing"
    assert len(library) == 1
    merged = library.require("existing")
    assert merged.version == 2
    assert pair.candidate.id in merged.evidence_ids
    assert library.get_evidence(pair.baseline.id) == pair.baseline
    assert library.get_evidence(pair.candidate.id) == pair.candidate
    assert {item.id for item in plane.archive.all()} == {
        pair.baseline.id,
        pair.candidate.id,
    }


def test_policy_library_json_persistence_round_trip(tmp_path) -> None:
    stored_policy = policy(
        "融合策略",
        "fusion 重复检索角色",
        precondition={"tools": ["search"]},
        utility=0.8,
        confidence=0.9,
    )
    stored_policy.status = PolicyStatus.VERIFIED
    stored_evidence = evidence(
        "evidence-1", policy_ids=(stored_policy.id,)
    )
    library_path = tmp_path / "policies.json"
    library = PolicyLibrary([stored_policy], path=library_path)
    library.record_evidence(
        stored_evidence, policy_id=stored_policy.id, positive=True
    )

    saved = library.save()
    restored = PolicyLibrary.load(saved)

    assert saved == library_path
    assert restored.to_dict() == library.to_dict()
    assert restored.evidence_for(stored_policy.id) == (stored_evidence,)
    raw = json.loads(library_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == PolicyLibrary.SCHEMA_VERSION
    assert raw["policies"][0]["id"] == "融合策略"


def test_pareto_archive_excludes_dominated_and_invalid_runs() -> None:
    dominant = evidence(
        "dominant", quality=1.0, tokens=80, latency=8, graph=8
    )
    dominated = evidence(
        "dominated", quality=0.9, tokens=100, latency=10, graph=10
    )
    quality_tradeoff = evidence(
        "tradeoff", quality=1.1, tokens=120, latency=12, graph=12
    )
    invalid = evidence(
        "invalid",
        quality=2.0,
        tokens=1,
        latency=1,
        graph=1,
        valid=False,
    )
    other_task = evidence(
        "other-task",
        task_id="task-2",
        quality=3.0,
        tokens=1,
        latency=1,
        graph=1,
    )
    archive = ParetoArchive(
        [dominant, dominated, quality_tradeoff, invalid, other_task]
    )

    frontier = archive.frontier(task_id="task-1")

    assert {item.id for item in frontier} == {"dominant", "tradeoff"}
    assert ("synthetic", "task-1") in archive.frontiers()
    assert archive.query(valid=False) == (invalid,)


def test_pair_evidence_matches_task_and_seed_not_input_order() -> None:
    baseline_one = evidence("b1", task_id="a", seed=1, variant="baseline")
    baseline_two = evidence("b2", task_id="a", seed=2, variant="baseline")
    candidate_one = evidence("c1", task_id="a", seed=1)
    candidate_two = evidence("c2", task_id="a", seed=2)

    pairs = pair_evidence(
        [baseline_two, baseline_one], [candidate_one, candidate_two]
    )

    assert [(pair.baseline.id, pair.candidate.id) for pair in pairs] == [
        ("b1", "c1"),
        ("b2", "c2"),
    ]


def test_pair_evidence_does_not_cross_benchmarks_or_splits() -> None:
    baseline = [
        evidence(
            "mbpp-base",
            benchmark="MBPP",
            task_id="1",
            seed=1,
            variant="baseline",
        ),
        evidence(
            "math-base",
            benchmark="MATH",
            task_id="1",
            seed=1,
            variant="baseline",
        ),
    ]
    candidate = [
        evidence(
            "math-candidate",
            benchmark="MATH",
            task_id="1",
            seed=1,
        ),
        evidence(
            "mbpp-candidate",
            benchmark="MBPP",
            task_id="1",
            seed=1,
        ),
    ]

    pairs = pair_evidence(baseline, candidate)

    assert [
        (pair.baseline.id, pair.candidate.id) for pair in pairs
    ] == [
        ("math-base", "math-candidate"),
        ("mbpp-base", "mbpp-candidate"),
    ]


def test_explicit_zero_graph_cost_does_not_fall_back_to_node_count() -> None:
    feedback = ExecutionFeedback(
        quality=1.0,
        graph_cost=0.0,
        node_count=3,
        edge_count=2,
    )
    implicit = ExecutionFeedback(
        quality=1.0,
        node_count=3,
        edge_count=2,
    )

    assert feedback.total_graph_cost == 0.0
    assert implicit.total_graph_cost == 5.0
