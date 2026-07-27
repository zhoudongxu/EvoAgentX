"""Compiler and static validation for CompactFlow's sidecar GFRG."""

from __future__ import annotations

import inspect
import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

from .schema import (
    GFRG,
    CallSpec,
    DataDependency,
    EffectDependency,
    GuardSpec,
    ResourceVector,
    SchemaContractError,
)

PathToken = str | int


def parse_field_path(path: str) -> tuple[PathToken, ...]:
    """Parse a concrete dot/bracket path or RFC 6901 JSON pointer.

    Examples:
        ``"papers[0].title"`` -> ``("papers", 0, "title")``
        ``"/papers/0/title"`` -> ``("papers", "0", "title")``; the numeric
        segment is resolved as an index when its schema/value is an array.

    Wildcards are deliberately rejected: guarded readiness must refer to a
    finite, exact field footprint.
    """

    if not isinstance(path, str):
        raise TypeError("field path must be a string")
    if path in {"", "$"}:
        return ()
    if "*" in path:
        raise ValueError(f"wildcards are not supported in field path {path!r}")
    if path.startswith("/"):
        tokens: list[PathToken] = []
        for raw in path.split("/")[1:]:
            token = raw.replace("~1", "/").replace("~0", "~")
            # RFC 6901 itself does not type path segments.  Preserve numeric
            # object keys here and resolve them as indices only when the
            # current schema/value is an array.
            tokens.append(token)
        return tuple(tokens)

    if path.startswith("$."):
        path = path[2:]
    elif path.startswith("$"):
        raise ValueError(f"invalid field path {path!r}")

    tokens = []
    index = 0
    need_token = True
    while index < len(path):
        char = path[index]
        if char == ".":
            if need_token:
                raise ValueError(f"invalid field path {path!r}")
            need_token = True
            index += 1
            continue
        if char == "[":
            end = path.find("]", index + 1)
            if end < 0:
                raise ValueError(f"unclosed bracket in field path {path!r}")
            raw = path[index + 1 : end].strip()
            if not raw:
                raise ValueError(f"empty bracket in field path {path!r}")
            if (
                len(raw) >= 2
                and raw[0] == raw[-1]
                and raw[0] in {"'", '"'}
            ):
                token: PathToken = raw[1:-1]
            elif re.fullmatch(r"0|[1-9][0-9]*", raw):
                token = int(raw)
            else:
                raise ValueError(
                    f"brackets require an index or quoted key in {path!r}"
                )
            tokens.append(token)
            need_token = False
            index = end + 1
            continue
        end = index
        while end < len(path) and path[end] not in ".[":
            end += 1
        raw = path[index:end]
        if not raw:
            raise ValueError(f"invalid field path {path!r}")
        tokens.append(raw)
        need_token = False
        index = end
    if need_token:
        raise ValueError(f"field path cannot end with '.' in {path!r}")
    return tuple(tokens)


def _validate_schema(schema: Mapping[str, Any], label: str) -> None:
    """Validate a schema against the metaschema for its declared draft."""

    if not isinstance(schema, Mapping):
        raise SchemaContractError(
            f"{label} must be a JSON Schema mapping"
        )
    try:
        validator_for(schema).check_schema(schema)
    except SchemaError as error:
        location = ".".join(str(token) for token in error.absolute_path)
        suffix = f" at {location}" if location else ""
        raise SchemaContractError(
            f"{label} is not a valid JSON Schema{suffix}: {error.message}"
        ) from error


def _is_cancellable_async_target(target: Any) -> bool:
    """Return whether task cancellation can stop the target itself.

    A synchronous target is run through :func:`asyncio.to_thread` by the
    runtime. Cancelling the awaiting asyncio task does not stop that worker
    thread, so such a consumer must not start on speculative partial output.
    This intentionally rejects ambiguous sync wrappers that merely return an
    awaitable; they can opt in by exposing a native coroutine/async-generator
    function.
    """

    return inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(
        target
    )


def _resolve_local_ref(
    root_schema: Mapping[str, Any], reference: str
) -> Mapping[str, Any] | None:
    if not reference.startswith("#/"):
        return None
    value: Any = root_schema
    for raw in reference[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or token not in value:
            return None
        value = value[token]
    return value if isinstance(value, Mapping) else None


def _schema_accepts_tokens(
    schema: Mapping[str, Any],
    tokens: tuple[PathToken, ...],
    root_schema: Mapping[str, Any],
    seen_refs: frozenset[str] = frozenset(),
) -> bool:
    if not tokens:
        return True
    reference = schema.get("$ref")
    if isinstance(reference, str):
        if reference in seen_refs:
            return False
        resolved = _resolve_local_ref(root_schema, reference)
        if resolved is None:
            # External refs cannot be proven invalid locally.
            return not reference.startswith("#")
        return _schema_accepts_tokens(
            resolved, tokens, root_schema, seen_refs | {reference}
        )

    variants = schema.get("anyOf") or schema.get("oneOf")
    if variants:
        return any(
            _schema_accepts_tokens(branch, tokens, root_schema, seen_refs)
            for branch in variants
        )
    if schema.get("allOf"):
        return any(
            _schema_accepts_tokens(branch, tokens, root_schema, seen_refs)
            for branch in schema["allOf"]
        )

    token, rest = tokens[0], tokens[1:]
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        schema_types = set(schema_type)
    elif schema_type is None:
        schema_types = set()
    else:
        schema_types = {schema_type}

    if isinstance(token, str) and token.isdigit() and "array" in schema_types:
        token = int(token)
    if isinstance(token, int):
        if schema_types and "array" not in schema_types:
            return False
        prefix_items = schema.get("prefixItems")
        if (
            isinstance(prefix_items, Sequence)
            and not isinstance(prefix_items, (str, bytes))
            and token < len(prefix_items)
            and isinstance(prefix_items[token], Mapping)
        ):
            return _schema_accepts_tokens(
                prefix_items[token], rest, root_schema, seen_refs
            )
        items = schema.get("items")
        if isinstance(items, Mapping):
            return _schema_accepts_tokens(items, rest, root_schema, seen_refs)
        # An unconstrained/empty schema accepts any concrete continuation.
        return not schema or items is True

    if schema_types and "object" not in schema_types:
        return False
    properties = schema.get("properties", {})
    if token in properties and isinstance(properties[token], Mapping):
        return _schema_accepts_tokens(
            properties[token], rest, root_schema, seen_refs
        )
    additional = schema.get("additionalProperties", not schema)
    if isinstance(additional, Mapping):
        return _schema_accepts_tokens(additional, rest, root_schema, seen_refs)
    return bool(additional)


def schema_accepts_path(schema: Mapping[str, Any], path: str) -> bool:
    """Return whether ``path`` can be resolved by the JSON schema."""

    tokens = parse_field_path(path)
    if not schema:
        # The root of an unconstrained value is valid, but named paths cannot
        # be statically proven.  This keeps ordinary schema-free calls usable
        # while requiring real schemas for field-level guarded execution.
        return not tokens
    return _schema_accepts_tokens(schema, tokens, schema)


def _schema_types_at_tokens(
    schema: Mapping[str, Any],
    tokens: tuple[PathToken, ...],
    root_schema: Mapping[str, Any],
    seen_refs: frozenset[str] = frozenset(),
) -> set[str]:
    reference = schema.get("$ref")
    if isinstance(reference, str):
        if reference in seen_refs:
            return set()
        resolved = _resolve_local_ref(root_schema, reference)
        if resolved is None:
            return set()
        return _schema_types_at_tokens(
            resolved, tokens, root_schema, seen_refs | {reference}
        )
    variants = schema.get("anyOf") or schema.get("oneOf")
    if variants:
        result: set[str] = set()
        for branch in variants:
            result.update(
                _schema_types_at_tokens(branch, tokens, root_schema, seen_refs)
            )
        return result
    if schema.get("allOf"):
        result = set()
        for branch in schema["allOf"]:
            result.update(
                _schema_types_at_tokens(branch, tokens, root_schema, seen_refs)
            )
        return result
    if not tokens:
        schema_type = schema.get("type")
        if isinstance(schema_type, str):
            return {schema_type}
        if isinstance(schema_type, Sequence) and not isinstance(
            schema_type, (str, bytes)
        ):
            return {value for value in schema_type if isinstance(value, str)}
        return set()

    token, rest = tokens[0], tokens[1:]
    schema_type = schema.get("type")
    schema_types = (
        {schema_type}
        if isinstance(schema_type, str)
        else set(schema_type or ())
    )
    if isinstance(token, str) and token.isdigit() and "array" in schema_types:
        token = int(token)
    if isinstance(token, int):
        prefix_items = schema.get("prefixItems")
        if (
            isinstance(prefix_items, Sequence)
            and not isinstance(prefix_items, (str, bytes))
            and token < len(prefix_items)
            and isinstance(prefix_items[token], Mapping)
        ):
            return _schema_types_at_tokens(
                prefix_items[token], rest, root_schema, seen_refs
            )
        items = schema.get("items")
        if isinstance(items, Mapping):
            return _schema_types_at_tokens(
                items, rest, root_schema, seen_refs
            )
        return set()

    properties = schema.get("properties", {})
    if isinstance(properties, Mapping) and isinstance(
        properties.get(token), Mapping
    ):
        return _schema_types_at_tokens(
            properties[token], rest, root_schema, seen_refs
        )
    additional = schema.get("additionalProperties")
    if isinstance(additional, Mapping):
        return _schema_types_at_tokens(
            additional, rest, root_schema, seen_refs
        )
    return set()


def schema_types_at_path(
    schema: Mapping[str, Any], path: str
) -> set[str]:
    """Return the JSON-schema types at a concrete path, when known."""

    if not schema:
        return set()
    return _schema_types_at_tokens(
        schema, parse_field_path(path), schema
    )


def _schemas_are_directionally_compatible(
    producer_types: set[str], consumer_types: set[str]
) -> bool:
    if not producer_types or not consumer_types:
        return True
    for producer_type in producer_types:
        if producer_type in consumer_types:
            continue
        if producer_type == "integer" and "number" in consumer_types:
            continue
        return False
    return True


class GFRGCompiler:
    """Validate and compile exact field/effect dependencies.

    The compiler never infers stability from a schema or from a value merely
    appearing in a partial result.  A guard is emitted only when all of these
    are explicit:

    * the data edge allows early use;
    * the producer has a monotone stream contract naming the exact path;
    * the consumer declares itself early-safe; and
    * the consumer is a natively cancellable async callable; and
    * no completion-only effect edge exists for the same producer/consumer.

    Synchronous callables run in worker threads. Since cancelling an asyncio
    wrapper cannot stop the underlying thread, they conservatively retain the
    producer-completion barrier even when ``early_safe=True``.
    """

    def __init__(
        self, resource_capacity: ResourceVector | Mapping[str, float] | None = None
    ) -> None:
        if isinstance(resource_capacity, ResourceVector):
            self.resource_capacity = resource_capacity
        else:
            self.resource_capacity = ResourceVector(resource_capacity)

    def compile(
        self,
        calls: Sequence[CallSpec] | Mapping[str, CallSpec],
        data_dependencies: Sequence[DataDependency] = (),
        effect_dependencies: Sequence[EffectDependency] = (),
        *,
        resource_capacity: ResourceVector | Mapping[str, float] | None = None,
    ) -> GFRG:
        if isinstance(calls, Mapping):
            call_items = tuple(calls.values())
        else:
            call_items = tuple(calls)
        data_items = tuple(data_dependencies)
        effect_items = tuple(effect_dependencies)
        if not call_items:
            raise ValueError("a GFRG must contain at least one call")

        call_map: dict[str, CallSpec] = {}
        for call in call_items:
            if not isinstance(call, CallSpec):
                raise TypeError("calls must contain CallSpec instances")
            if call.id in call_map:
                raise ValueError(f"duplicate call id {call.id!r}")
            call_map[call.id] = call
            _validate_schema(call.input_schema, f"{call.id}.input_schema")
            _validate_schema(call.output_schema, f"{call.id}.output_schema")
            if call.stream_contract is not None:
                for path in (
                    call.stream_contract.stable_fields
                    + call.stream_contract.mutable_fields
                ):
                    parse_field_path(path)
                    if not schema_accepts_path(call.output_schema, path):
                        raise ValueError(
                            f"stream path {path!r} is absent from output schema "
                            f"of {call.id!r}"
                        )

        target_writers: set[tuple[str, str]] = set()
        for dependency in data_items:
            if not isinstance(dependency, DataDependency):
                raise TypeError(
                    "data_dependencies must contain DataDependency instances"
                )
            parse_field_path(dependency.source_path)
            parse_field_path(dependency.target_path)
            if dependency.consumer not in call_map:
                raise ValueError(
                    f"unknown data consumer {dependency.consumer!r}"
                )
            if dependency.producer is not None:
                if dependency.producer not in call_map:
                    raise ValueError(
                        f"unknown data producer {dependency.producer!r}"
                    )
                if dependency.producer == dependency.consumer:
                    raise ValueError("self data dependencies are not allowed")
                producer = call_map[dependency.producer]
                if not schema_accepts_path(
                    producer.output_schema, dependency.source_path
                ):
                    raise ValueError(
                        f"source path {dependency.source_path!r} is absent "
                        f"from output schema of {dependency.producer!r}"
                    )
            consumer = call_map[dependency.consumer]
            if not schema_accepts_path(
                consumer.input_schema, dependency.target_path
            ):
                raise ValueError(
                    f"target path {dependency.target_path!r} is absent from "
                    f"input schema of {dependency.consumer!r}"
                )
            if dependency.producer is not None:
                producer_types = schema_types_at_path(
                    call_map[dependency.producer].output_schema,
                    dependency.source_path,
                )
                consumer_types = schema_types_at_path(
                    consumer.input_schema, dependency.target_path
                )
                if not _schemas_are_directionally_compatible(
                    producer_types, consumer_types
                ):
                    raise ValueError(
                        f"incompatible schemas for "
                        f"{dependency.producer}.{dependency.source_path} -> "
                        f"{dependency.consumer}.{dependency.target_path}: "
                        f"{sorted(producer_types)} cannot satisfy "
                        f"{sorted(consumer_types)}"
                    )
            writer = (dependency.consumer, dependency.target_path)
            if writer in target_writers:
                raise ValueError(
                    "multiple dependencies write target path "
                    f"{dependency.consumer}.{dependency.target_path}"
                )
            target_writers.add(writer)

        for dependency in effect_items:
            if not isinstance(dependency, EffectDependency):
                raise TypeError(
                    "effect_dependencies must contain EffectDependency instances"
                )
            if dependency.producer not in call_map:
                raise ValueError(
                    f"unknown effect producer {dependency.producer!r}"
                )
            if dependency.consumer not in call_map:
                raise ValueError(
                    f"unknown effect consumer {dependency.consumer!r}"
                )
            if dependency.producer == dependency.consumer:
                raise ValueError("self effect dependencies are not allowed")
            if dependency.effect not in call_map[dependency.producer].effects:
                raise ValueError(
                    f"effect {dependency.effect!r} is not declared by "
                    f"producer {dependency.producer!r}"
                )
            if dependency.allow_partial and not _is_cancellable_async_target(
                call_map[dependency.consumer].target
            ):
                raise ValueError(
                    "partial effect dependencies require a native async "
                    f"consumer; {dependency.consumer!r} would run in a "
                    "non-cancellable worker thread"
                )

        capacity = (
            resource_capacity
            if resource_capacity is not None
            else self.resource_capacity
        )
        if not isinstance(capacity, ResourceVector):
            capacity = ResourceVector(capacity)
        for call in call_items:
            for resource, demand in call.resources.items():
                if (
                    resource in capacity.amounts
                    and demand > capacity.get(resource)
                ):
                    raise ValueError(
                        f"call {call.id!r} demands {demand:g} {resource}, "
                        f"exceeding capacity {capacity.get(resource):g}"
                    )

        topological_order = self._topological_order(
            tuple(call_map), data_items, effect_items
        )
        unsafe_effect_pairs = {
            (dependency.producer, dependency.consumer)
            for dependency in effect_items
            if not dependency.allow_partial
        }
        guards: list[GuardSpec] = []
        for dependency in data_items:
            if dependency.producer is None or not dependency.allow_early:
                continue
            producer = call_map[dependency.producer]
            consumer = call_map[dependency.consumer]
            contract = producer.stream_contract
            if (
                consumer.early_safe
                and _is_cancellable_async_target(consumer.target)
                and contract is not None
                and contract.explicitly_stabilizes(dependency.source_path)
                and (
                    dependency.producer,
                    dependency.consumer,
                )
                not in unsafe_effect_pairs
            ):
                guards.append(GuardSpec.from_dependency(dependency))

        return GFRG(
            calls=call_items,
            data_dependencies=data_items,
            effect_dependencies=effect_items,
            guards=tuple(guards),
            resource_capacity=capacity,
            topological_order=topological_order,
        )

    @staticmethod
    def _topological_order(
        call_ids: tuple[str, ...],
        data_dependencies: tuple[DataDependency, ...],
        effect_dependencies: tuple[EffectDependency, ...],
    ) -> tuple[str, ...]:
        adjacency: dict[str, set[str]] = {call_id: set() for call_id in call_ids}
        indegree = {call_id: 0 for call_id in call_ids}
        edges = [
            (dependency.producer, dependency.consumer)
            for dependency in data_dependencies
            if dependency.producer is not None
        ] + [
            (dependency.producer, dependency.consumer)
            for dependency in effect_dependencies
        ]
        for producer, consumer in edges:
            if consumer in adjacency[producer]:
                continue
            adjacency[producer].add(consumer)
            indegree[consumer] += 1
        queue = deque(call_id for call_id in call_ids if indegree[call_id] == 0)
        result: list[str] = []
        while queue:
            call_id = queue.popleft()
            result.append(call_id)
            for consumer in adjacency[call_id]:
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    queue.append(consumer)
        if len(result) != len(call_ids):
            cyclic = sorted(
                call_id for call_id, degree in indegree.items() if degree
            )
            raise ValueError(
                "combined data/effect dependency graph must be acyclic; "
                f"cycle involves {cyclic}"
            )
        return tuple(result)


def compile_gfrg(
    calls: Sequence[CallSpec] | Mapping[str, CallSpec],
    data_dependencies: Sequence[DataDependency] = (),
    effect_dependencies: Sequence[EffectDependency] = (),
    *,
    resource_capacity: ResourceVector | Mapping[str, float] | None = None,
) -> GFRG:
    """Convenience wrapper around :class:`GFRGCompiler`."""

    return GFRGCompiler(resource_capacity).compile(
        calls,
        data_dependencies,
        effect_dependencies,
    )


__all__ = [
    "GFRGCompiler",
    "PathToken",
    "compile_gfrg",
    "parse_field_path",
    "schema_accepts_path",
    "schema_types_at_path",
]
