"""Model-driven typed workflow construction and exact-field streaming execution."""
from __future__ import annotations
from contextlib import aclosing
from .llm import format_options

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
    from .gaia import WORKFLOW_TOOLS
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
        allowed = {"llm", "retrieve_context"} | (WORKFLOW_TOOLS if public_input.get("benchmark") == "GAIA" else set())
        if node["tool"] not in allowed:
            raise ValueError("unregistered tool")
        if not isinstance(node["instruction"], str) or not isinstance(node["inputs"], dict):
            raise TypeError("invalid node instruction/inputs")
        if not node["outputs"] or len(set(node["outputs"])) != len(node["outputs"]):
            raise ValueError("node needs unique outputs")
        if any(not isinstance(x, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", x) for x in node["outputs"]):
            raise ValueError("outputs must be simple field names")
        if node["tool"] in WORKFLOW_TOOLS - {"gaia_agent"} and set(node["inputs"]) != {"arguments"}:
            raise ValueError("GAIA primitive requires exactly one input named arguments, bound to an llm builder's JSON string; tool argument keys such as asset_id/unit/query belong INSIDE that JSON, not in workflow bindings")
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
        raise ValueError("every planned node must contribute to the answer sink; unused nodes: " + str(sorted(ids-ancestors)) + "; bind their outputs in a downstream node or omit unused nodes")


def compile_spec(spec: dict, public_input: dict, client, *, seed: int, capacity: int = 4, recorder=None, tool_session=None):
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
            dependencies.append(DataDependency(producer, node["id"], source, argument, allow_early=True))
        output_schema = object_schema({name: {"type": "string"} for name in node["outputs"]})
        if node["tool"].startswith("gaia_"):
            frozen_node = copy.deepcopy(node)
            async def target(_node=frozen_node, **arguments):
                if tool_session is None:
                    raise ValueError("GAIA tool session is required at execution")
                async with aclosing(tool_session.workflow_stream(client, _node, arguments)) as events:
                    async for event in events:
                        yield event
        elif node["tool"] == "retrieve_context":
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
            from .gaia import TOOL_SCHEMAS
            output_contracts = []
            for consumer in spec["nodes"]:
                if not consumer["tool"].startswith("gaia_") or consumer["tool"] == "gaia_agent":
                    continue
                reference = consumer["inputs"].get("arguments", "")
                if reference.split(".", 1)[0] != node["id"]:
                    continue
                fields = list(consumer["outputs"])
                output_contracts.append({"output_field":reference.split(".",1)[1],
                    "tool":consumer["tool"], "tool_arguments":TOOL_SCHEMAS[consumer["tool"][5:]],
                    "required_observation_fields":fields,
                    "encoding": "Emit a JSON string containing ordinary tool arguments" if len(fields)==1 else
                        "Emit a JSON string containing exactly {units:[{field:NAME,arguments:TOOL_ARGUMENTS},...]}; include every required_observation_field exactly once. Do not emit just one unit's arguments."})
            async def target(_node=frozen_node, _contracts=copy.deepcopy(output_contracts), **arguments):
                buffer, values = "", {}
                payload = {"instruction": _node["instruction"], "inputs": arguments,
                           "output_fields": _node["outputs"]}
                if _contracts:
                    payload["gaia_output_contracts"] = _contracts
                request = canonical(payload)
                async with aclosing(client.stream(prompt("execute"), request, component="executor", seed=seed,
                        **format_options(client, "stream", {"type":"ndjson_fields", "fields":_node["outputs"]}))) as stream:
                    async for chunk in stream:
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
                        early_safe=not node["tool"].startswith("gaia_"), resources={"external_call": 1},
                        stream_contract=StreamContract(stable_fields=() if node["tool"].startswith("gaia_") else tuple(node["outputs"])),
                        metadata={"tool": node["tool"], "effect_class": "pure", "explicit_input_bindings": True,
                                  "footprint_method": "explicit_typed_input_bindings_v1"})
        calls.append(recorder.wrap(call) if recorder else call)
    graph = GFRGCompiler({"external_call": capacity}).compile(calls, dependencies)
    from .gaia_contracts import annotate_graph
    return annotate_graph(graph, {n["id"]:n["tool"] for n in spec["nodes"] if n["tool"].startswith("gaia_")})


def normalize_single_sink(spec):
    """Canonical answer port alias only; no node, edge, or value is invented."""
    if not isinstance(spec,dict) or not isinstance(spec.get("sinks"),list) or len(spec["sinks"]) != 1:
        return spec, []
    nodes=spec.get("nodes",[])
    if not isinstance(nodes,list):return spec, []
    sinks=[n for n in nodes if isinstance(n,dict) and n.get("id")==spec["sinks"][0]]
    if len(sinks)!=1:return spec, []
    fields=sinks[0].get("outputs")
    if not isinstance(fields,list) or len(fields)!=1 or not isinstance(fields[0],str) or fields[0]=="answer" or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*",fields[0]):
        return spec, []
    result=copy.deepcopy(spec);cid=sinks[0]["id"];old=fields[0]
    for n in result["nodes"]:
        if n.get("id")==cid:n["outputs"]=["answer"]
        if isinstance(n.get("inputs"),dict):
            n["inputs"]={k:(cid+".answer" if v==cid+"."+old else v) for k,v in n["inputs"].items()}
    return result, [{"kind":"single_sink_output_alias","node":cid,"from":old,"to":"answer"}]


async def plan_workflow(client, task, policies, config, *, seed: int):
    selected = [p.to_dict() for p in policies]
    data = {"task": task.public_input(), "selected_policies": selected, "max_nodes": config["max_nodes"]}
    data["workflow_schema"] = json.loads((PROMPTS.parent / "schemas/workflow.schema.json").read_text())
    if config.get("typed_binding_guidance"):
        public_fields = "|".join(re.escape(k) for k in task.public_input())
        binding = data["workflow_schema"]["properties"]["nodes"]["items"]["properties"]["inputs"]["additionalProperties"]
        binding["pattern"] = r"^(\$input\.(?:" + public_fields + r")|[a-zA-Z][a-zA-Z0-9_]*\.[a-zA-Z][a-zA-Z0-9_]*)$"
        data["binding_rules"] = {
            "workflow_inputs": ["$input."+k for k in task.public_input()],
            "node_output_example": "If node first outputs result, bind it as first.result, never $input.result or $first.result.",
            "argument_names": "Argument keys are simple identifiers such as question or previous; never $input.question or a dotted path.",
            "context_example": "If retrieve_context outputs doc0, a consumer uses retrieve_context.doc0. Binding $input.context does not depend on that node.",
            "connectivity": "Every emitted node must reach the sink through explicit input references. Omit nodes that no later node uses."
        }
    if task.benchmark == "GAIA":
        from .gaia import TOOL_SCHEMAS, WORKFLOW_TOOLS
        data["workflow_schema"]["properties"]["nodes"]["items"]["properties"]["tool"]["enum"] = ["llm", "retrieve_context", *sorted(WORKFLOW_TOOLS)]
        data["gaia_agent_internal_tools"] = TOOL_SCHEMAS
        data["gaia_guidance"] = ("Use gaia_agent for adaptive tool reasoning, or registered gaia_* primitives for explicit operations. "
            "A primitive binds arguments to a JSON string produced by an llm node; that builder must bind question/assets. "
            "One output uses ordinary tool arguments. Multiple outputs require {units:[{field:declared_name,arguments:{...}},...]}. "
            "Each unit is a separate budgeted tool call and commits only its complete result; each output field maps to exactly one unit. "
            "Use read_file unit selectors for PDF pages, slides, paragraphs and spreadsheet rows; never split provisional answers into supposedly stable fields. "
            "All consumers need explicit input bindings. Asset-mutating operations also obey completion effect barriers. "
            "No requirement to introduce multiple units when the task does not need them.")
        data["gaia_example"] = {"nodes":[{"id":"solve", "tool":"gaia_agent",
            "instruction":"Use the authorized tools to solve the task, inspecting the supplied assets as needed.",
            "inputs":{"question":"$input.question", "assets":"$input.assets"}, "outputs":["answer"]}],
            "sinks":["solve"],"applied_policies":[],"unapplied_policies":[]}
        data["gaia_primitive_example"] = {
            "nodes": [
                {"id":"build_args","tool":"llm","instruction":"Choose the relevant authorized asset_id from assets. Emit arguments as a JSON string containing the tool arguments, for example an object with asset_id and limit=12000. Do not emit workflow references in that string.",
                 "inputs":{"question":"$input.question","assets":"$input.assets"},"outputs":["arguments"]},
                {"id":"read","tool":"gaia_read_file","instruction":"Read the selected asset.",
                 "inputs":{"arguments":"build_args.arguments"},"outputs":["observation"]},
                {"id":"answer","tool":"llm","instruction":"Answer the question from the observation.",
                 "inputs":{"question":"$input.question","observation":"read.observation"},"outputs":["answer"]}],
            "sinks":["answer"],"applied_policies":[],"unapplied_policies":[]}
    errors = []
    spec = None
    original_ids = {p.id for p in policies}
    for attempt in range(config["max_repairs"] + 1):
        if attempt:
            data["repair"] = {"error": errors[-1], "previous_workflow": spec}
            if config.get("typed_binding_guidance") and isinstance(spec,dict):
                data["repair"]["valid_references"] = ["$input."+k for k in task.public_input()] + [
                    n["id"]+"."+f for n in spec.get("nodes",[]) if isinstance(n,dict) and isinstance(n.get("id"),str)
                    for f in n.get("outputs",[]) if isinstance(f,str)]
                data["repair"]["instruction"] = "Correct the reported binding/connectivity error. Return the entire corrected graph, not the same invalid graph."

        try:
            spec = await client.json(prompt("plan"), canonical(data), component="planner", seed=seed + attempt,
                                     **format_options(client, "json", {"type":"json_schema", "schema":data["workflow_schema"]}))
            normalizations = []
            if config.get("typed_binding_guidance"):
                spec, normalizations = normalize_single_sink(spec)
            validate_spec(spec, task.public_input(), original_ids, config["max_nodes"])
            compile_spec(spec, task.public_input(), client, seed=seed)
            return spec, {"repairs": attempt, "errors": errors, "fallback": False, "normalizations": normalizations}
        except (ValueError, TypeError, KeyError) as error:
            errors.append(str(error))
    if policies and config["fallback"] == "base_planner":
        spec, trace = await plan_workflow(client, task, [], {**config, "fallback": "fail"}, seed=seed)
        return spec, {**trace, "fallback": True, "conditioned_errors": errors}
    raise ValueError("workflow planning failed: " + "; ".join(errors))
