"""Configuration, stable splits, anonymous manifests, and artifact writers.

The helpers in this module make experiment bookkeeping reproducible without
requiring an LLM provider or network access.  In particular, manifests never
inspect or serialize the current user, home directory, working directory,
hostname, process environment, or credentials.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeAlias

from .metrics import RunRecord

JsonValue: TypeAlias = Any


class ConfigValidationError(ValueError):
    """Raised when an experiment configuration cannot be executed safely."""


_SENSITIVE_KEY_TOKENS = {
    "author",
    "credential",
    "cwd",
    "email",
    "env",
    "environment",
    "home",
    "host",
    "hostname",
    "key",
    "maintainer",
    "machine",
    "owner",
    "password",
    "secret",
    "token",
    "user",
    "username",
}
_SENSITIVE_KEY_FRAGMENTS = {
    "accesskey",
    "apikey",
    "email",
    "emailaddress",
    "homedir",
    "hostname",
    "maintainer",
    "owner",
    "privatekey",
    "secretkey",
    "username",
}

_ALLOWED_DATASET_SPLITS = {
    "train",
    "validation",
    "test",
    "dev",
    "source",
    "target",
}
_RUNNER_METHODS: dict[str, dict[str, frozenset[str]]] = {
    "offline_smoke": {
        "construction": frozenset({"policy_retrieval_smoke"}),
        "execution": frozenset({"complete_dependency", "guarded"}),
    }
}
_PROTOCOL_TEMPLATE_RUNNER = "protocol_template"
_MISSING = object()
_ABSOLUTE_PATH_PATTERNS = (
    re.compile(r"(?<![:/\w])/(?:[^/\s]+/)+[^/\s]*"),
    re.compile(r"(?i)(?<!\w)[a-z]:\\(?:[^\\\s]+\\)*[^\\\s]*"),
    re.compile(r"(?<!\w)~(?:/[^/\s]+)+"),
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bgh[oprsu]_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{8,}\b"),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sensitive_key(key: str) -> bool:
    tokens = {
        token
        for token in re.split(r"[^A-Za-z0-9]+", key.lower())
        if token
    }
    joined = "".join(tokens)
    return bool(tokens.intersection(_SENSITIVE_KEY_TOKENS)) or any(
        fragment in joined for fragment in _SENSITIVE_KEY_FRAGMENTS
    )


def _looks_like_absolute_path(value: str) -> bool:
    if value.startswith(("file://", "/", "~")):
        return True
    if re.match(r"(?i)^[a-z]:[\\/]", value):
        return True
    return any(pattern.search(value) for pattern in _ABSOLUTE_PATH_PATTERNS)


def _redact_string(value: str) -> str:
    redacted = value
    if _looks_like_absolute_path(redacted):
        if (
            redacted.startswith(("file://", "/", "~"))
            or re.match(r"(?i)^[a-z]:[\\/]", redacted)
        ):
            return "<redacted-path>"
        for pattern in _ABSOLUTE_PATH_PATTERNS:
            redacted = pattern.sub("<redacted-path>", redacted)
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("<redacted-secret>", redacted)
    return redacted


def sanitize_for_manifest(value: Any) -> JsonValue:
    """Remove identity/secret fields and redact absolute filesystem paths.

    Sensitive mapping entries are omitted rather than replaced so a manifest
    cannot reveal even which environment variable or credential mechanism was
    used.  Relative artifact paths remain intact.
    """

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math_is_finite(value):
            raise ValueError("manifest values must be finite")
        return value
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, Mapping):
        sanitized: dict[str, JsonValue] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            if _sensitive_key(key):
                continue
            sanitized[key] = sanitize_for_manifest(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_manifest(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return sanitize_for_manifest(value.to_dict())
    raise TypeError(
        f"manifest value of type {type(value).__name__} is not JSON-compatible"
    )


def math_is_finite(value: float) -> bool:
    """Avoid importing a numerical package for one finite-value check."""

    return math.isfinite(value)


def assert_anonymous_manifest(value: Any) -> None:
    """Raise when a manifest contains forbidden keys, paths, or secrets."""

    def visit(item: Any, location: str) -> None:
        if isinstance(item, Mapping):
            for raw_key, nested in item.items():
                key = str(raw_key)
                if _sensitive_key(key):
                    raise ValueError(
                        f"manifest contains forbidden field at "
                        f"{location}.{key}"
                    )
                visit(nested, f"{location}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{location}[{index}]")
        elif isinstance(item, str):
            if _looks_like_absolute_path(item):
                raise ValueError(
                    f"manifest contains an absolute path at {location}"
                )
            if any(pattern.search(item) for pattern in _SECRET_VALUE_PATTERNS):
                raise ValueError(
                    f"manifest contains a credential-like value at {location}"
                )

    visit(value, "$")
    _canonical_json(value)


def _stable_task_payload(task_id: int | str) -> str:
    if isinstance(task_id, bool) or not isinstance(task_id, (int, str)):
        raise TypeError("task identifiers must be integers or strings")
    return _canonical_json(
        {"type": type(task_id).__name__, "value": task_id}
    )


def _stable_hash(task_id: int | str, salt: str) -> int:
    payload = f"{salt}\0{_stable_task_payload(task_id)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def stable_hash_split(
    task_ids: Iterable[int | str],
    *,
    source_fraction: float,
    validation_fraction: float,
    salt: str = "compactflow-v1",
) -> dict[int | str, str]:
    """Assign task IDs to source/validation/target by a stable SHA-256 hash.

    Hash thresholds, rather than shuffled list positions, ensure that changing
    input order or appending unrelated tasks does not move an existing task.
    """

    for name, fraction in (
        ("source_fraction", source_fraction),
        ("validation_fraction", validation_fraction),
    ):
        if not math_is_finite(float(fraction)) or not 0 <= fraction <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    if source_fraction + validation_fraction > 1:
        raise ValueError(
            "source_fraction + validation_fraction cannot exceed one"
        )
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a non-empty string")

    result: dict[int | str, str] = {}
    denominator = float(1 << 256)
    for task_id in task_ids:
        _stable_task_payload(task_id)
        if task_id in result:
            raise ValueError(f"duplicate task identifier: {task_id!r}")
        position = _stable_hash(task_id, salt) / denominator
        if position < source_fraction:
            split = "source"
        elif position < source_fraction + validation_fraction:
            split = "validation"
        else:
            split = "target"
        result[task_id] = split
    return result


def split_task_ids(
    task_ids: Iterable[int | str],
    *,
    source_fraction: float,
    validation_fraction: float,
    salt: str = "compactflow-v1",
) -> dict[str, list[int | str]]:
    """Return stable split assignments grouped in deterministic hash order."""

    assignments = stable_hash_split(
        task_ids,
        source_fraction=source_fraction,
        validation_fraction=validation_fraction,
        salt=salt,
    )
    grouped: dict[str, list[int | str]] = {
        "source": [],
        "validation": [],
        "target": [],
    }
    for task_id, split in assignments.items():
        grouped[split].append(task_id)
    for split, identifiers in grouped.items():
        grouped[split] = sorted(
            identifiers, key=lambda item: _stable_hash(item, salt)
        )
    return grouped


def _required_value(config: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = config
    for component in dotted_path.split("."):
        if not component:
            raise ConfigValidationError(
                f"invalid required path: {dotted_path!r}"
            )
        if not isinstance(current, Mapping) or component not in current:
            raise ConfigValidationError(
                f"required config field is missing: {dotted_path}"
            )
        current = current[component]
    return current


def _value_at(config: Mapping[str, Any], dotted_path: str) -> Any:
    """Return a dotted-path value, or an internal missing sentinel."""

    current: Any = config
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current


def _numeric_value(
    config: Mapping[str, Any],
    dotted_path: str,
    *,
    integer: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
    required: bool = False,
) -> float | int | None:
    value = _value_at(config, dotted_path)
    if value is _MISSING or value is None:
        if required:
            raise ConfigValidationError(
                f"{dotted_path} must be resolved for an executable run"
            )
        return None
    valid_type = (
        isinstance(value, int)
        if integer
        else isinstance(value, (int, float))
    )
    if isinstance(value, bool) or not valid_type:
        kind = "an integer" if integer else "a number"
        raise ConfigValidationError(f"{dotted_path} must be {kind}")
    numeric = float(value)
    if not math_is_finite(numeric):
        raise ConfigValidationError(f"{dotted_path} must be finite")
    if minimum is not None and numeric < minimum:
        raise ConfigValidationError(
            f"{dotted_path} must be at least {minimum:g}"
        )
    if maximum is not None and numeric > maximum:
        raise ConfigValidationError(
            f"{dotted_path} must be at most {maximum:g}"
        )
    return int(value) if integer else numeric


def _validate_numeric_fields(
    config: Mapping[str, Any],
    *,
    for_run: bool,
) -> None:
    """Validate all numeric protocol knobs without resolving templates."""

    required = for_run
    include_candidates = _value_at(
        config, "construction.include_candidate_policies"
    )
    if include_candidates is _MISSING or include_candidates is None:
        if required:
            raise ConfigValidationError(
                "construction.include_candidate_policies must be resolved "
                "for an executable run"
            )
    elif not isinstance(include_candidates, bool):
        raise ConfigValidationError(
            "construction.include_candidate_policies must be true or false"
        )

    _numeric_value(
        config,
        "model.temperature",
        minimum=0,
        required=required,
    )
    _numeric_value(
        config,
        "model.max_completion_tokens",
        integer=True,
        minimum=1,
        required=required,
    )

    k0 = _numeric_value(
        config,
        "construction.semantic_top_k0",
        integer=True,
        minimum=1,
        required=required,
    )
    top_k = _numeric_value(
        config,
        "construction.top_k",
        integer=True,
        minimum=1,
        required=required,
    )
    max_policies = _numeric_value(
        config,
        "construction.max_policies",
        integer=True,
        minimum=1,
        required=required,
    )
    if k0 is not None and top_k is not None and k0 < top_k:
        raise ConfigValidationError(
            "construction.semantic_top_k0 must be greater than or equal "
            "to construction.top_k"
        )
    if (
        top_k is not None
        and max_policies is not None
        and max_policies > top_k
    ):
        raise ConfigValidationError(
            "construction.max_policies cannot exceed construction.top_k"
        )

    retrieval_weights = [
        _numeric_value(
            config,
            f"construction.retrieval_weights.{name}",
            minimum=0,
            required=required,
        )
        for name in ("semantic", "structural", "historical")
    ]
    resolved_retrieval_weights = [
        value for value in retrieval_weights if value is not None
    ]
    if (
        len(resolved_retrieval_weights) == len(retrieval_weights)
        and sum(resolved_retrieval_weights) > 1 + 1e-12
    ):
        raise ConfigValidationError(
            "construction retrieval weights cannot sum to more than one"
        )

    for path in (
        "construction.quality_tolerance",
        "construction.minimum_cost_reduction",
    ):
        _numeric_value(config, path, minimum=0, required=required)
    _numeric_value(
        config,
        "construction.merge_similarity_threshold",
        minimum=0,
        maximum=1,
        required=required,
    )

    cost_weights = [
        _numeric_value(
            config,
            f"construction.cost_weights.{name}",
            minimum=0,
            required=False,
        )
        for name in ("tokens", "latency", "graph")
    ]
    if any(value is not None for value in cost_weights):
        if any(value is None for value in cost_weights):
            raise ConfigValidationError(
                "construction.cost_weights must resolve tokens, latency, "
                "and graph together"
            )
        if sum(value for value in cost_weights if value is not None) <= 0:
            raise ConfigValidationError(
                "construction.cost_weights must contain a positive weight"
            )

    source_fraction = _numeric_value(
        config,
        "construction.source_fraction",
        minimum=0,
        maximum=1,
        required=required,
    )
    validation_fraction = _numeric_value(
        config,
        "construction.validation_fraction",
        minimum=0,
        maximum=1,
        required=required,
    )
    if (
        source_fraction is not None
        and validation_fraction is not None
        and source_fraction + validation_fraction > 1 + 1e-12
    ):
        raise ConfigValidationError(
            "construction source and validation fractions cannot sum "
            "to more than one"
        )

    _numeric_value(
        config,
        "execution.external_call_capacity",
        integer=True,
        minimum=1,
        required=required,
    )
    _numeric_value(
        config,
        "execution.percentage_threshold",
        minimum=0,
        maximum=1,
    )
    _numeric_value(
        config,
        "execution.materialization_batch_size",
        integer=True,
        minimum=1,
    )
    _numeric_value(
        config,
        "evaluation.sample_count_per_benchmark",
        integer=True,
        minimum=1,
    )
    _numeric_value(
        config,
        "evaluation.rollout_budget",
        integer=True,
        minimum=1,
    )
    repetitions = _numeric_value(
        config,
        "evaluation.repetitions",
        integer=True,
        minimum=1,
        required=required,
    )
    if for_run and repetitions != 1:
        raise ConfigValidationError(
            "offline_smoke currently implements exactly one repetition"
        )

    generation_seeds = _value_at(config, "evaluation.generation_seeds")
    if generation_seeds is not _MISSING and generation_seeds is not None:
        if (
            not isinstance(generation_seeds, list)
            or not generation_seeds
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, (int, str))
                or (isinstance(seed, str) and not seed.strip())
                for seed in generation_seeds
            )
        ):
            raise ConfigValidationError(
                "evaluation.generation_seeds must be a non-empty list "
                "of integer or non-empty string seeds"
            )
        if len({_canonical_json(seed) for seed in generation_seeds}) != len(
            generation_seeds
        ):
            raise ConfigValidationError(
                "evaluation.generation_seeds contains duplicates"
            )


def _validate_runner_methods(
    config: Mapping[str, Any],
    *,
    for_run: bool,
) -> None:
    runner = config.get("runner")
    allowed_runners = {*_RUNNER_METHODS, _PROTOCOL_TEMPLATE_RUNNER}
    if runner not in allowed_runners:
        raise ConfigValidationError(
            "runner must be offline_smoke or protocol_template"
        )
    if runner == _PROTOCOL_TEMPLATE_RUNNER:
        if not config["template"] or config["runnable"]:
            raise ConfigValidationError(
                "protocol_template is a non-runnable experiment scaffold"
            )
        return
    if config["template"]:
        raise ConfigValidationError(
            "an executable runner cannot be used by a template"
        )
    if not for_run:
        return
    expected = _RUNNER_METHODS[str(runner)]
    methods = config["methods"]
    for plane, expected_methods in expected.items():
        configured = set(methods.get(plane, ()))
        if configured != expected_methods:
            raise ConfigValidationError(
                f"runner {runner} implements methods.{plane}="
                f"{sorted(expected_methods)}, got {sorted(configured)}"
            )


def validate_experiment_config(
    config: Mapping[str, Any],
    *,
    for_run: bool = False,
) -> dict[str, Any]:
    """Validate and return a detached JSON-compatible configuration.

    Template files may be loaded and inspected.  They are rejected only when
    ``for_run=True``; runnable configurations additionally require every
    dotted path in ``required`` to be present and non-null.
    """

    if not isinstance(config, Mapping):
        raise ConfigValidationError("experiment config must be a JSON object")
    try:
        cloned = json.loads(_canonical_json(config))
    except (TypeError, ValueError) as error:
        raise ConfigValidationError(
            "experiment config must contain finite JSON values"
        ) from error

    if cloned.get("schema_version") != 1:
        raise ConfigValidationError("schema_version must equal 1")
    if not isinstance(cloned.get("name"), str) or not cloned["name"].strip():
        raise ConfigValidationError("name must be a non-empty string")
    for field_name in ("template", "runnable", "offline", "network_access"):
        if not isinstance(cloned.get(field_name), bool):
            raise ConfigValidationError(
                f"{field_name} must be explicitly true or false"
            )
    if cloned["offline"] and cloned["network_access"]:
        raise ConfigValidationError(
            "an offline experiment cannot enable network access"
        )
    if cloned["template"] and cloned["runnable"]:
        raise ConfigValidationError(
            "a template cannot be marked runnable"
        )
    if for_run and (cloned["template"] or not cloned["runnable"]):
        raise ConfigValidationError(
            "template/non-runnable configuration cannot start a run"
        )
    seed = cloned.get("seed")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, str))
    ):
        raise ConfigValidationError("seed must be an integer, string, or null")
    if for_run and (
        seed is None or (isinstance(seed, str) and not seed.strip())
    ):
        raise ConfigValidationError(
            "seed must be resolved for an executable run"
        )

    datasets = cloned.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ConfigValidationError("datasets must be a non-empty list")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ConfigValidationError(
                f"datasets[{index}] must be an object"
            )
        if not isinstance(dataset.get("name"), str) or not dataset["name"]:
            raise ConfigValidationError(
                f"datasets[{index}].name must be non-empty"
            )
        if cloned["offline"] and dataset.get("source") not in {
            "synthetic",
            "replay",
            "local",
        }:
            raise ConfigValidationError(
                "offline datasets must use synthetic, replay, or local source"
            )
        split = dataset.get("split")
        if split is not None and split not in _ALLOWED_DATASET_SPLITS:
            raise ConfigValidationError(
                f"datasets[{index}].split must be one of "
                f"{sorted(_ALLOWED_DATASET_SPLITS)} or null in a template"
            )
        if for_run and split is None:
            raise ConfigValidationError(
                f"datasets[{index}].split must be resolved for a run"
            )
        task_ids = dataset.get("task_ids", _MISSING)
        if task_ids is not _MISSING:
            if (
                not isinstance(task_ids, list)
                or not task_ids
                or any(
                    isinstance(task_id, bool)
                    or not isinstance(task_id, (int, str))
                    or (isinstance(task_id, str) and not task_id.strip())
                    for task_id in task_ids
                )
            ):
                raise ConfigValidationError(
                    f"datasets[{index}].task_ids must be a non-empty list "
                    "of integer or non-empty string identifiers"
                )
            if len({_canonical_json(item) for item in task_ids}) != len(
                task_ids
            ):
                raise ConfigValidationError(
                    f"datasets[{index}].task_ids contains duplicates"
                )
        elif for_run and dataset.get("source") == "synthetic":
            raise ConfigValidationError(
                f"datasets[{index}].task_ids is required for synthetic data"
            )

    methods = cloned.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ConfigValidationError("methods must be a non-empty object")
    for plane, plane_methods in methods.items():
        if plane not in {"construction", "execution"}:
            raise ConfigValidationError(f"unknown method plane: {plane}")
        if (
            not isinstance(plane_methods, list)
            or not plane_methods
            or any(
                not isinstance(method, str) or not method
                for method in plane_methods
            )
        ):
            raise ConfigValidationError(
                f"methods.{plane} must contain non-empty method names"
            )
        if len(set(plane_methods)) != len(plane_methods):
            raise ConfigValidationError(
                f"methods.{plane} contains duplicate methods"
            )

    required = cloned.get("required")
    if (
        not isinstance(required, list)
        or any(not isinstance(path, str) or not path for path in required)
    ):
        raise ConfigValidationError(
            "required must be a list of dotted field paths"
        )
    if len(set(required)) != len(required):
        raise ConfigValidationError("required contains duplicate paths")

    _validate_runner_methods(cloned, for_run=for_run)

    if for_run:
        unresolved = [
            path for path in required if _required_value(cloned, path) is None
        ]
        if unresolved:
            raise ConfigValidationError(
                "required config fields are unresolved: "
                + ", ".join(sorted(unresolved))
            )
    else:
        for path in required:
            _required_value(cloned, path)
    _validate_numeric_fields(cloned, for_run=for_run)
    return cloned


def load_experiment_config(
    path: str | os.PathLike[str],
    *,
    for_run: bool = False,
) -> dict[str, Any]:
    """Load and validate a JSON experiment configuration."""

    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    return validate_experiment_config(config, for_run=for_run)


def build_anonymous_manifest(
    config: Mapping[str, Any],
    *,
    dataset_ids: Mapping[str, Sequence[int | str]] | None = None,
    git_commit: str | None = None,
    created_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest without local identity metadata."""

    validated = validate_experiment_config(config, for_run=False)
    datasets = {
        str(name): list(identifiers)
        for name, identifiers in (dataset_ids or {}).items()
    }
    sanitized_config = sanitize_for_manifest(validated)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "experiment_name": validated["name"],
        "created_at": created_at
        or datetime.now(timezone.utc).isoformat(),
        "config_digest": _digest(validated),
        "config": sanitized_config,
        "dataset_fingerprint": _digest(datasets),
        "dataset_counts": {
            name: len(identifiers)
            for name, identifiers in sorted(datasets.items())
        },
        "runtime": {
            "python_version": (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            )
        },
    }
    if git_commit is not None:
        manifest["git_commit"] = _redact_string(str(git_commit))
    if extra:
        sanitized_extra = sanitize_for_manifest(extra)
        if sanitized_extra:
            manifest["extra"] = sanitized_extra
    assert_anonymous_manifest(manifest)
    return manifest


def _jsonable(value: Any) -> Any:
    if isinstance(value, RunRecord):
        return value.to_dict()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class ArtifactWriter:
    """Write deterministic JSON, JSONL, and CSV experiment artifacts."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _target(self, relative_name: str | os.PathLike[str]) -> Path:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact name must be a safe relative path")
        if not relative.parts:
            raise ValueError("artifact name must not be empty")
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(
        self, relative_name: str | os.PathLike[str], value: Any
    ) -> Path:
        """Write one pretty, deterministic JSON document."""

        target = self._target(relative_name)
        with target.open("w", encoding="utf-8") as stream:
            json.dump(
                _jsonable(value),
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        return target

    def write_jsonl(
        self,
        relative_name: str | os.PathLike[str],
        rows: Iterable[Any],
    ) -> Path:
        """Write one canonical JSON object per line."""

        target = self._target(relative_name)
        with target.open("w", encoding="utf-8") as stream:
            for row in rows:
                normalized = _jsonable(row)
                if not isinstance(normalized, Mapping):
                    raise TypeError("JSONL rows must serialize to objects")
                stream.write(_canonical_json(normalized))
                stream.write("\n")
        return target

    def write_csv(
        self,
        relative_name: str | os.PathLike[str],
        rows: Iterable[Any],
    ) -> Path:
        """Write mappings as CSV, encoding nested values as canonical JSON."""

        normalized_rows = [_jsonable(row) for row in rows]
        if any(not isinstance(row, Mapping) for row in normalized_rows):
            raise TypeError("CSV rows must serialize to objects")
        fieldnames = sorted(
            {
                str(key)
                for row in normalized_rows
                for key in row
            }
        )
        target = self._target(relative_name)
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in normalized_rows:
                encoded = {}
                for key in fieldnames:
                    value = row.get(key)
                    if isinstance(value, (Mapping, list, tuple)):
                        encoded[key] = _canonical_json(_jsonable(value))
                    elif value is None:
                        encoded[key] = ""
                    else:
                        encoded[key] = value
                writer.writerow(encoded)
        return target

    def write_manifest(
        self,
        manifest: Mapping[str, Any],
        relative_name: str | os.PathLike[str] = "manifest.json",
    ) -> Path:
        """Validate anonymity and write a manifest."""

        assert_anonymous_manifest(manifest)
        return self.write_json(relative_name, manifest)


def write_experiment_artifacts(
    writer: ArtifactWriter,
    *,
    records: Iterable[RunRecord],
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Path]:
    """Write the standard record, summary, and anonymous manifest bundle."""

    record_list = list(records)
    return {
        "records_jsonl": writer.write_jsonl("records.jsonl", record_list),
        "records_csv": writer.write_csv("records.csv", record_list),
        "summary": writer.write_json("summary.json", summary),
        "manifest": writer.write_manifest(manifest),
    }
