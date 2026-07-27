"""Offline tests for CompactFlow metrics and experiment artifacts."""

from __future__ import annotations

import asyncio
import csv
import json
import re
from pathlib import Path

import pytest

from evoagentx.compactflow.experiments import (
    ArtifactWriter,
    ConfigValidationError,
    assert_anonymous_manifest,
    build_anonymous_manifest,
    load_experiment_config,
    sanitize_for_manifest,
    stable_hash_split,
    validate_experiment_config,
    write_experiment_artifacts,
)
from evoagentx.compactflow.metrics import (
    PairingError,
    RunRecord,
    RuntimeRecord,
    StructureRecord,
    ViolationCounts,
    aggregate_runtime_diagnostics,
    compare_paired_methods,
    percentile,
)

REPOSITORY_ROOT = Path(__file__).parents[3]
SMOKE_CONFIG = (
    REPOSITORY_ROOT / "examples/compactflow/configs/smoke.json"
)
PAPER_TEMPLATE = (
    REPOSITORY_ROOT / "examples/compactflow/configs/paper_template.json"
)
IMPLEMENTATION_README = (
    REPOSITORY_ROOT / "examples/compactflow/README.md"
)


def run_record(
    method: str,
    *,
    task_id: str = "task-1",
    seed: int = 7,
    quality: float = 1.0,
    tokens: tuple[int, int] = (60, 40),
    nodes: int = 4,
    edges: int = 3,
    critical_path: int = 3,
    latency: float = 10.0,
    ttfo: float | None = 4.0,
    analyzed_edges: int = 0,
    partial_edges: int = 0,
    opportunity: bool = False,
    gaps: tuple[float, ...] = (),
    first_guard_ready_item: float | None = None,
    dispatched_calls: int | None = None,
    early_calls: int = 0,
    violations: ViolationCounts | None = None,
) -> RunRecord:
    return RunRecord(
        benchmark="synthetic",
        task_id=task_id,
        seed=seed,
        method=method,
        quality=quality,
        structure=StructureRecord(
            node_count=nodes,
            edge_count=edges,
            critical_path_length=critical_path,
        ),
        runtime=RuntimeRecord(
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            latency_seconds=latency,
            ttfo_seconds=ttfo,
            analyzed_data_edges=analyzed_edges,
            partial_edges=partial_edges,
            task_has_opportunity=opportunity,
            readiness_gaps=gaps,
            first_guard_ready_item_seconds=first_guard_ready_item,
            dispatched_calls=(
                early_calls
                if dispatched_calls is None
                else dispatched_calls
            ),
            early_dispatched_calls=early_calls,
            violations=violations or ViolationCounts(),
        ),
    )


def test_structure_record_measures_dag_critical_path() -> None:
    structure = StructureRecord.from_edges(
        ["a", "b", "c", "d"],
        [("a", "b"), ("a", "c"), ("c", "d")],
    )

    assert structure == StructureRecord(
        node_count=4,
        edge_count=3,
        critical_path_length=3,
    )
    with pytest.raises(ValueError, match="cyclic"):
        StructureRecord.from_edges(["a", "b"], [("a", "b"), ("b", "a")])


def test_strict_paired_comparison_uses_full_record_key() -> None:
    records = [
        run_record(
            "baseline",
            task_id="a",
            quality=0.8,
            tokens=(60, 40),
            latency=10.0,
            ttfo=5.0,
        ),
        run_record(
            "candidate",
            task_id="a",
            quality=0.8,
            tokens=(45, 35),
            nodes=3,
            edges=2,
            critical_path=2,
            latency=5.0,
            ttfo=2.0,
        ),
        run_record(
            "baseline",
            task_id="b",
            quality=0.6,
            tokens=(60, 40),
            latency=8.0,
            ttfo=4.0,
        ),
        run_record(
            "candidate",
            task_id="b",
            quality=0.7,
            tokens=(40, 30),
            nodes=3,
            edges=2,
            critical_path=2,
            latency=4.0,
            ttfo=2.0,
        ),
    ]

    assert records[0].pair_key == ("synthetic", "a", 7, "baseline")
    summary = compare_paired_methods(
        records,
        baseline_method="baseline",
        candidate_method="candidate",
    )

    assert summary.pair_count == 2
    assert summary.delta_quality == pytest.approx(0.05)
    assert summary.total_tokens.relative_reduction == pytest.approx(0.25)
    assert summary.nodes.relative_reduction == pytest.approx(0.25)
    assert summary.latency.relative_reduction == pytest.approx(0.5)
    assert summary.speedup == pytest.approx(2.0)
    assert summary.ttfo is not None
    assert summary.ttfo.relative_reduction == pytest.approx(5.0 / 9.0)


def test_strict_pairing_rejects_missing_and_duplicate_keys() -> None:
    missing = [
        run_record("baseline", task_id="present-only-on-baseline"),
        run_record("candidate", task_id="present-only-on-candidate"),
    ]
    with pytest.raises(PairingError, match="missing candidate") as error:
        compare_paired_methods(
            missing,
            baseline_method="baseline",
            candidate_method="candidate",
        )
    assert "('synthetic', 'present-only-on-baseline', 7, 'candidate')" in str(
        error.value
    )

    duplicate = [
        run_record("baseline"),
        run_record("baseline"),
        run_record("candidate"),
    ]
    with pytest.raises(PairingError, match="duplicate paired key"):
        compare_paired_methods(
            duplicate,
            baseline_method="baseline",
            candidate_method="candidate",
        )


def test_percentiles_use_linear_interpolation() -> None:
    values = [0.0, 2.0, 4.0, 6.0, 10.0]

    assert percentile(values, 0.50) == pytest.approx(4.0)
    assert percentile(values, 0.90) == pytest.approx(8.4)
    assert percentile(values, 0.95) == pytest.approx(9.2)
    assert percentile([], 0.50) is None


def test_runtime_diagnostics_and_four_violation_rates() -> None:
    records = [
        run_record(
            "guarded",
            task_id="a",
            analyzed_edges=6,
            partial_edges=3,
            opportunity=True,
            gaps=(0.1, 0.2),
            early_calls=4,
            violations=ViolationCounts(
                argument_mismatch=1,
                duplicate_dispatch=1,
            ),
        ),
        run_record(
            "guarded",
            task_id="b",
            analyzed_edges=4,
            partial_edges=2,
            opportunity=False,
            gaps=(0.4, 0.8),
            early_calls=6,
            violations=ViolationCounts(
                effect_order=1,
                capacity_overload=1,
            ),
        ),
    ]

    diagnostics = aggregate_runtime_diagnostics(records)

    assert diagnostics.task_opportunity_rate == pytest.approx(0.5)
    assert diagnostics.partial_edge_ratio == pytest.approx(0.5)
    assert diagnostics.readiness_p50 == pytest.approx(0.3)
    assert diagnostics.readiness_p90 == pytest.approx(0.68)
    assert diagnostics.readiness_p95 == pytest.approx(0.74)
    assert diagnostics.violation_rates == pytest.approx(
        {
            "argument_mismatch": 0.1,
            "duplicate_dispatch": 0.1,
            "effect_order": 0.1,
            "capacity_overload": 0.1,
            "aggregate": 0.4,
        }
    )
    assert diagnostics.dispatched_calls == 10


def test_violation_incidence_uses_all_dispatches_not_early_calls() -> None:
    record = run_record(
        "guarded",
        dispatched_calls=2,
        early_calls=0,
        violations=ViolationCounts(
            argument_mismatch=1,
            effect_order=1,
            capacity_overload=1,
        ),
    )

    diagnostics = aggregate_runtime_diagnostics([record])

    assert diagnostics.early_dispatched_calls == 0
    assert diagnostics.dispatched_calls == 2
    assert diagnostics.violation_rates == pytest.approx(
        {
            "argument_mismatch": 0.5,
            "duplicate_dispatch": 0.0,
            "effect_order": 0.5,
            "capacity_overload": 0.5,
            "aggregate": 1.5,
        }
    )


def test_zero_denominators_are_json_safe() -> None:
    diagnostics = aggregate_runtime_diagnostics(
        [run_record("guarded", ttfo=None)]
    )

    assert diagnostics.partial_edge_ratio == 0.0
    assert set(diagnostics.violation_rates.values()) == {0.0}
    assert diagnostics.readiness_p50 is None
    assert diagnostics.readiness_p90 is None
    assert diagnostics.readiness_p95 is None
    json.dumps(diagnostics.to_dict(), allow_nan=False)


def test_hash_split_is_order_independent_and_append_stable() -> None:
    identifiers = [f"task-{index}" for index in range(100)]
    first = stable_hash_split(
        identifiers,
        source_fraction=0.6,
        validation_fraction=0.2,
        salt="test-protocol",
    )
    reversed_order = stable_hash_split(
        reversed(identifiers),
        source_fraction=0.6,
        validation_fraction=0.2,
        salt="test-protocol",
    )
    extended = stable_hash_split(
        [*identifiers, "new-task"],
        source_fraction=0.6,
        validation_fraction=0.2,
        salt="test-protocol",
    )

    assert first == reversed_order
    assert {task_id: extended[task_id] for task_id in identifiers} == first
    assert set(first.values()) == {"source", "validation", "target"}


def test_manifest_removes_identity_secret_fields_and_absolute_paths() -> None:
    config = load_experiment_config(SMOKE_CONFIG, for_run=True)
    manifest = build_anonymous_manifest(
        config,
        dataset_ids={"synthetic": ["a", "b"]},
        git_commit="0123456789abcdef",
        created_at="2026-07-27T00:00:00+00:00",
        extra={
            "username": "private-user",
            "email": "private@example.invalid",
            "maintainer": "private-maintainer",
            "owner": "private-owner",
            "home": "/home/private-user",
            "cwd": "/home/private-user/project",
            "hostname": "private-host",
            "env": {"API_KEY": "sk-not-for-artifacts"},
            "output_path": "/home/private-user/project/result.json",
            "command": "python /home/private-user/project/run.py",
            "safe_note": "offline smoke",
        },
    )
    encoded = json.dumps(manifest, sort_keys=True)

    assert "private-user" not in encoded
    assert "example.invalid" not in encoded
    assert "private-maintainer" not in encoded
    assert "private-owner" not in encoded
    assert "private-host" not in encoded
    assert "/home/" not in encoded
    assert "sk-not-for-artifacts" not in encoded
    assert "<redacted-path>" in encoded
    assert manifest["extra"]["safe_note"] == "offline smoke"
    assert_anonymous_manifest(manifest)

    with pytest.raises(ValueError, match="forbidden field"):
        assert_anonymous_manifest({"username": "not-anonymous"})
    with pytest.raises(ValueError, match="forbidden field"):
        assert_anonymous_manifest({"maintainer_email": "not-anonymous"})


def test_path_redaction_handles_posix_windows_and_embedded_paths() -> None:
    sanitized = sanitize_for_manifest(
        {
            "posix_path": "/Users/person/project/config.json",
            "windows_path": "C:\\Users\\person\\project\\config.json",
            "message": "read /home/person/project/config.json before running",
            "relative_path": "outputs/compactflow/run.json",
        }
    )

    assert sanitized["posix_path"] == "<redacted-path>"
    assert sanitized["windows_path"] == "<redacted-path>"
    assert "<redacted-path>" in sanitized["message"]
    assert "person" not in json.dumps(sanitized)
    assert sanitized["relative_path"] == "outputs/compactflow/run.json"


def test_smoke_is_runnable_but_paper_template_is_not() -> None:
    smoke = load_experiment_config(SMOKE_CONFIG, for_run=True)
    template = load_experiment_config(PAPER_TEMPLATE)

    assert smoke["offline"] is True
    assert smoke["network_access"] is False
    assert smoke["runner"] == "offline_smoke"
    assert template["template"] is True
    assert template["runner"] == "protocol_template"
    assert template["model"]["name"] is None
    with pytest.raises(ConfigValidationError, match="cannot start a run"):
        load_experiment_config(PAPER_TEMPLATE, for_run=True)


def _smoke_config() -> dict:
    return json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))


def _set_path(config: dict, path: str, value) -> None:
    current = config
    components = path.split(".")
    for component in components[:-1]:
        current = current[component]
    current[components[-1]] = value


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "construction.semantic_top_k0",
            2,
            "semantic_top_k0 must be greater",
        ),
        (
            "construction.max_policies",
            4,
            "max_policies cannot exceed",
        ),
        (
            "construction.retrieval_weights.semantic",
            -0.1,
            "must be at least",
        ),
        (
            "construction.merge_similarity_threshold",
            1.1,
            "must be at most",
        ),
        (
            "execution.external_call_capacity",
            0,
            "must be at least",
        ),
        ("evaluation.repetitions", 0, "must be at least"),
        ("model.temperature", "cold", "must be a number"),
    ],
)
def test_runnable_config_rejects_invalid_numeric_fields(
    path: str,
    value,
    message: str,
) -> None:
    config = _smoke_config()
    _set_path(config, path, value)

    with pytest.raises(ConfigValidationError, match=message):
        validate_experiment_config(config, for_run=True)


def test_runnable_config_validates_weight_and_fraction_sums() -> None:
    config = _smoke_config()
    config["construction"]["retrieval_weights"] = {
        "semantic": 0.5,
        "structural": 0.5,
        "historical": 0.1,
    }
    with pytest.raises(ConfigValidationError, match="sum to more than one"):
        validate_experiment_config(config, for_run=True)

    config = _smoke_config()
    config["construction"]["source_fraction"] = 0.9
    config["construction"]["validation_fraction"] = 0.2
    with pytest.raises(ConfigValidationError, match="fractions cannot sum"):
        validate_experiment_config(config, for_run=True)


def test_runnable_config_rejects_unresolved_split_seed_and_unknown_method() -> None:
    config = _smoke_config()
    config["datasets"][0]["split"] = None
    with pytest.raises(ConfigValidationError, match="split must be resolved"):
        validate_experiment_config(config, for_run=True)

    config = _smoke_config()
    config["seed"] = None
    with pytest.raises(ConfigValidationError, match="seed must be resolved"):
        validate_experiment_config(config, for_run=True)

    config = _smoke_config()
    config["methods"]["execution"].append("percentage_threshold")
    with pytest.raises(ConfigValidationError, match="implements methods.execution"):
        validate_experiment_config(config, for_run=True)


def test_optional_protocol_numbers_are_validated_when_resolved() -> None:
    config = _smoke_config()
    config["construction"]["cost_weights"] = {
        "tokens": 0,
        "latency": 0,
        "graph": 0,
    }
    with pytest.raises(ConfigValidationError, match="positive weight"):
        validate_experiment_config(config, for_run=True)

    config = _smoke_config()
    config["evaluation"]["generation_seeds"] = [7, 7]
    with pytest.raises(ConfigValidationError, match="contains duplicates"):
        validate_experiment_config(config, for_run=True)


def test_guard_ready_metric_excludes_mutable_partial_events() -> None:
    from evoagentx.compactflow.runtime import CompactFlowRuntime
    from evoagentx.compactflow.schema import ExecutionMode
    from examples.compactflow.run_experiments import (
        _runtime_record,
        full_result_fallback_graph,
        streaming_fanout_graph,
    )

    stable_graph = streaming_fanout_graph(capacity=2)
    stable_execution = asyncio.run(
        CompactFlowRuntime(
            stable_graph, mode=ExecutionMode.GUARDED
        ).execute()
    )
    stable_record = _runtime_record(stable_graph, stable_execution)

    mutable_graph = full_result_fallback_graph(capacity=2)
    mutable_execution = asyncio.run(
        CompactFlowRuntime(
            mutable_graph, mode=ExecutionMode.GUARDED
        ).execute()
    )
    mutable_record = _runtime_record(mutable_graph, mutable_execution)

    assert stable_record.first_guard_ready_item_seconds is not None
    assert mutable_record.first_guard_ready_item_seconds is None


def test_implementation_readme_contains_no_identity_metadata() -> None:
    text = IMPLEMENTATION_README.read_text(encoding="utf-8")

    assert not re.search(
        r"(?im)^\s*(?:author|authors|maintainer|created[-_ ]by)\s*[:=]",
        text,
    )
    assert not re.search(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        text,
        flags=re.IGNORECASE,
    )
    assert not re.search(r"(?<!\w)/(?:home|Users)/[^/\s]+/", text)


def test_json_jsonl_csv_artifact_bundle(tmp_path: Path) -> None:
    config = load_experiment_config(SMOKE_CONFIG, for_run=True)
    manifest = build_anonymous_manifest(
        config,
        dataset_ids={"synthetic": ["task-1"]},
        created_at="2026-07-27T00:00:00+00:00",
    )
    records = [
        run_record("baseline"),
        run_record(
            "candidate",
            tokens=(40, 30),
            nodes=3,
            edges=2,
            critical_path=2,
            latency=5.0,
            ttfo=2.0,
        ),
    ]
    summary = compare_paired_methods(
        records,
        baseline_method="baseline",
        candidate_method="candidate",
    ).to_dict()

    paths = write_experiment_artifacts(
        ArtifactWriter(tmp_path),
        records=records,
        summary=summary,
        manifest=manifest,
    )

    assert set(paths) == {
        "records_jsonl",
        "records_csv",
        "summary",
        "manifest",
    }
    jsonl_rows = [
        json.loads(line)
        for line in paths["records_jsonl"].read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [row["method"] for row in jsonl_rows] == [
        "baseline",
        "candidate",
    ]
    with paths["records_csv"].open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        csv_rows = list(csv.DictReader(stream))
    assert [row["method"] for row in csv_rows] == [
        "baseline",
        "candidate",
    ]
    assert json.loads(paths["manifest"].read_text(encoding="utf-8"))[
        "experiment_name"
    ] == "compactflow-offline-smoke"

    with pytest.raises(ValueError, match="safe relative path"):
        ArtifactWriter(tmp_path).write_json("../escape.json", {})
