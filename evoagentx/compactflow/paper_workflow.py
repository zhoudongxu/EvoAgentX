"""Model-driven typed workflow construction and exact-field streaming execution."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from .compiler import GFRGCompiler
from .replay import canonical
from .schema import CallSpec, Complete, DataDependency, Partial, StreamContract

PROMPTS = Path(__file__).resolve().parents[2] / "examples" / "compactflow" / "prompts"


def prompt(name: str) -> str:
    return (PROMPTS / (name + ".txt")).read_text()


def object_schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def validate_spec(spec: dict, public_input: dict, selected_ids: set[str], max_nodes: int = 12) -> None:
    from jsonschema import Draft202012Validator
    schema = json.loads((PROMPTS.parent / "schemas/workflow.schema.json").read_text())
    error = next(iter(Draft202012Validator(schema).iter_errors(spec)), None)
    if error is not None:
        raise ValueError(f"workflow schema at {list(error.path)}: {error.message}")
    if set(spec) - {"nodes", "sinks", "applied_policies", "unapplied_policies"}:
        raise ValueError("unknown workflow fields")
    if not isinstance(spec.get("nodes"), list) or not 1 <= len(spec["nodes"]) <= max_nodes:
        raise ValueError("node count outside planner bound")
    ids = set()
    for node in spec["nodes"]:
        if set(node) != {"id", "tool", "instruction", "inputs", "outputs"}:
            raise ValueError("each node needs exactly id/tool/instruction/inputs/outputs")
        if not isinstance(node["id"], str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", node["id"]) or node["id"] in ids:
            raise ValueError("invalid/duplicate node ID")
        ids.add(node["id"])
        if node["tool"] not in {"llm", "retrieve_context"}:
            raise ValueError("unregistered tool")
        if not isinstance(node["instruction"], str) or not isinstance(node["inputs"], dict):
            raise TypeError("invalid node instruction/inputs")
        if not node["outputs"] or len(set(node["outputs"])) != len(node["outputs"]):
            raise ValueError("node needs unique outputs")
        if any(not isinstance(x, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", x) for x in node["outputs"]):
            raise ValueError("outputs must be simple field names")
        if node["tool"] == "retrieve_context" and node["inputs"] != {"context": "$input.context"}:
            raise ValueError("retrieve_context requires exactly the supplied context")
    if not spec.get("sinks") or set(spec["sinks"]) - ids or len(spec["sinks"]) != 1:
        raise ValueError("reference answer evaluator requires one declared sink")
    sinks = [n for n in spec["nodes"] if n["id"] in spec["sinks"]]
    if "answer" not in sinks[0]["outputs"]:
        raise ValueError("sink must expose answer")
    applied = spec.get("applied_policies", [])
    applied_ids = {p["id"] if isinstance(p, dict) else p for p in applied}
    if applied_ids - selected_ids:
        raise ValueError("planner claimed an unselected policy")
    mapping = {n["id"]: n for n in spec["nodes"]}
    ancestors = set(spec["sinks"])
    pending = list(ancestors)
    while pending:
        for reference in mapping[pending.pop()]["inputs"].values():
            if not isinstance(reference, str) or "." not in reference:
                raise ValueError("input binding must be producer.field or $input.field")
            producer, field = reference.split(".", 1)
            if producer == "$input":
                if field not in public_input:
                    raise ValueError(f"unknown workflow input {field}")
            else:
                if producer not in mapping or field not in mapping[producer]["outputs"]:
                    raise ValueError(f"unknown input binding {reference}")
                if producer not in ancestors:
                    ancestors.add(producer)
                    pending.append(producer)
    if ancestors != ids:
        raise ValueError("every planned node must contribute to the answer sink")


def compile_spec(spec: dict, public_input: dict, client, *, seed: int, capacity: int = 4, recorder=None):
    calls, dependencies = [], []
    for node in spec["nodes"]:
        properties = {}
        for argument, reference in node["inputs"].items():
            producer, source = reference.split(".", 1)
            if producer == "$input":
                value = public_input[source]
                field_type = "array" if isinstance(value, list) else "object" if isinstance(value, dict) else "string"
                producer = None
            else:
                field_type = "string"
            properties[argument] = {"type": field_type}
            dependencies.append(DataDependency(producer, node["id"], source, argument))
        output_schema = object_schema({name: {"type": "string"} for name in node["outputs"]})
        if node["tool"] == "retrieve_context":
            async def target(context, _fields=tuple(node["outputs"])):
                accumulated = {}
                # Real materialization points only: no artificial latency is added.
                # Each field exposes one independently immutable dataset record.
                for i, field in enumerate(_fields):
                    accumulated[field] = canonical(context[i]) if i < len(context) else ""
                    yield Partial({field: accumulated[field]}, (field,))
                yield Complete(accumulated)
        else:
            frozen_node = copy.deepcopy(node)
            async def target(_node=frozen_node, **arguments):
                buffer, values = "", {}
                request = canonical({"instruction": _node["instruction"], "inputs": arguments,
                                     "output_fields": _node["outputs"]})
                async for chunk in client.stream(prompt("execute"), request, component="executor", seed=seed):
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        if set(item) != {"field", "value"} or item["field"] not in _node["outputs"] or not isinstance(item["value"], str):
                            raise ValueError("invalid materialized output field")
                        if item["field"] in values:
                            raise ValueError("a stable output field was emitted twice")
                        values[item["field"]] = item["value"]
                        yield Partial({item["field"]: item["value"]}, (item["field"],))
                if buffer.strip():
                    item = json.loads(buffer)
                    if set(item) != {"field", "value"} or item["field"] not in _node["outputs"] or not isinstance(item["value"], str) or item["field"] in values:
                        raise ValueError("invalid final output field")
                    values[item["field"]] = item["value"]
                    yield Partial({item["field"]: item["value"]}, (item["field"],))
                if set(values) != set(_node["outputs"]):
                    raise ValueError("executor omitted declared output fields")
                yield Complete(values)
        call = CallSpec(node["id"], target, input_schema=object_schema(properties), output_schema=output_schema,
                        early_safe=True, resources={"external_call": 1},
                        stream_contract=StreamContract(stable_fields=tuple(node["outputs"])),
                        metadata={"tool": node["tool"], "effect_class": "pure", "explicit_input_bindings": True,
                                  "footprint_method": "explicit_typed_input_bindings_v1"})
        calls.append(recorder.wrap(call) if recorder else call)
    return GFRGCompiler({"external_call": capacity}).compile(calls, dependencies)


async def plan_workflow(client, task, policies, config, *, seed: int):
    selected = [p.to_dict() for p in policies]
    data = {"task": task.public_input(), "selected_policies": selected, "max_nodes": config["max_nodes"]}
    data["workflow_schema"] = json.loads((PROMPTS.parent / "schemas/workflow.schema.json").read_text())
    errors = []
    spec = None
    original_ids = {p.id for p in policies}
    for attempt in range(config["max_repairs"] + 1):
        if attempt:
            data["repair"] = {"error": errors[-1], "previous_workflow": spec}
        try:
            spec = await client.json(prompt("plan"), canonical(data), component="planner", seed=seed + attempt)
            validate_spec(spec, task.public_input(), original_ids, config["max_nodes"])
            compile_spec(spec, task.public_input(), client, seed=seed)
            return spec, {"repairs": attempt, "errors": errors, "fallback": False}
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
    if policies and config["fallback"] == "base_planner":
        spec, trace = await plan_workflow(client, task, [], {**config, "fallback": "fail"}, seed=seed)
        return spec, {**trace, "fallback": True, "conditioned_errors": errors}
    raise ValueError("workflow planning failed: " + "; ".join(errors))
