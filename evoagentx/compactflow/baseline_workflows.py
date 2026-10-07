"""Bounded exports for native baseline DAGs; generated Python is never executed."""

from __future__ import annotations
import ast
import copy
from .compiler import GFRGCompiler
from .replay import digest
from .schema import CallSpec, DataDependency, EffectDependency, ResourceVector


class UnsupportedWorkflow(ValueError):
    pass


SUPPORTED_OPERATORS = {
    "GAIAToolAgent",
    "Custom",
    "CustomCodeGenerate",
    "AnswerGenerate",
    "ScEnsemble",
    "QAScEnsemble",
}


def export_aflow(graph_source, prompt_source="", *, max_nodes=12, operator_specs=None):
    """Export explicit awaited calls/gather; reject dynamic control flow."""
    constants = {}
    for s in ast.parse(prompt_source).body:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue
        if (
            not isinstance(s, ast.Assign)
            or len(s.targets) != 1
            or not isinstance(s.targets[0], ast.Name)
        ):
            raise UnsupportedWorkflow("prompt module requires literal assignments")
        try:
            constants[s.targets[0].id] = ast.literal_eval(s.value)
        except (ValueError, TypeError) as e:
            raise UnsupportedWorkflow("nonliteral prompt constant") from e
    workflow = None
    for s in ast.parse(graph_source).body:
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in s.names]
                if isinstance(s, ast.Import)
                else [s.module or ""]
            )
            if not all(
                n == "asyncio"
                or n.startswith(("evoagentx.", "examples.aflow."))
                or n.endswith(".prompt")
                for n in names
            ):
                raise UnsupportedWorkflow("unapproved import")
        elif isinstance(s, ast.ClassDef) and s.name == "Workflow" and workflow is None:
            workflow = s
        elif isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue
        else:
            raise UnsupportedWorkflow("unsupported top-level statement")
    if workflow is None or workflow.bases or workflow.decorator_list:
        raise UnsupportedWorkflow("plain Workflow class required")
    methods = {
        s.name: s
        for s in workflow.body
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if set(methods) != {"__init__", "__call__"} or any(
        not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        for s in workflow.body
    ):
        raise UnsupportedWorkflow("only __init__ and __call__ permitted")
    aliases = {}
    for s in methods["__init__"].body:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue
        if not isinstance(s, ast.Assign) or len(s.targets) != 1:
            raise UnsupportedWorkflow("unsupported constructor statement")
        t, v = s.targets[0], s.value
        if (
            not isinstance(t, ast.Attribute)
            or not isinstance(t.value, ast.Name)
            or t.value.id != "self"
        ):
            raise UnsupportedWorkflow("constructor must assign self attributes")
        if (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and ast.unparse(v.func.value) == "operator"
        ):
            allowed = set(operator_specs) if operator_specs is not None else SUPPORTED_OPERATORS
            if v.func.attr not in allowed:
                raise UnsupportedWorkflow(
                    f"operator {v.func.attr} lacks a common-runtime export"
                )
            if (
                v.keywords
                or len(v.args) != 1
                or ast.unparse(v.args[0]) not in {"self.llm", "llm"}
            ):
                raise UnsupportedWorkflow("custom operator constructor")
            aliases[t.attr] = v.func.attr
        elif t.attr in {"name", "benchmark", "llm"} and ast.unparse(v) in {
            "name",
            "benchmark",
            "llm",
            "create_llm_instance(llm_config)",
        }:
            pass
        else:
            raise UnsupportedWorkflow("unsupported constructor attribute")
    call = methods["__call__"]
    if (
        not isinstance(call, ast.AsyncFunctionDef)
        or call.decorator_list
        or call.args.vararg
        or call.args.kwarg
        or call.args.defaults
        or call.args.kwonlyargs
    ):
        raise UnsupportedWorkflow("fixed async __call__ signature required")
    args = [x.arg for x in call.args.args if x.arg != "self"]
    if not args or set(args) - {
        "problem",
        "question",
        "inputs",
        "input",
        "entry_point",
    }:
        raise UnsupportedWorkflow("unsupported public input signature")
    bindings = {
        a: {"input": "entry_point" if a == "entry_point" else "problem"} for a in args
    }
    nodes, edges, barriers, pending = [], set(), set(), set()

    def expr(n):
        if isinstance(n, ast.Constant) and isinstance(
            n.value, (str, int, float, bool, type(None))
        ):
            return {"literal": n.value}
        if isinstance(n, ast.Name) and n.id in bindings:
            return copy.deepcopy(bindings[n.id])
        if (
            isinstance(n, ast.Attribute)
            and ast.unparse(n.value) == "prompt_custom"
            and n.attr in constants
        ):
            return {"literal": constants[n.attr]}
        if isinstance(n, (ast.List, ast.Tuple)):
            return {"list": [expr(v) for v in n.elts]}
        if isinstance(n, ast.Dict) and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in n.keys
        ):
            return {"dict": {k.value: expr(v) for k, v in zip(n.keys, n.values)}}
        if isinstance(n, ast.Subscript):
            return {"index": [expr(n.value), expr(n.slice)]}
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return {"add": [expr(n.left), expr(n.right)]}
        if isinstance(n, ast.JoinedStr):
            parts = []
            for p in n.values:
                if isinstance(p, ast.Constant):
                    parts.append(expr(p))
                elif (
                    isinstance(p, ast.FormattedValue)
                    and p.conversion in {-1, 115}
                    and p.format_spec is None
                ):
                    parts.append({"str": expr(p.value)})
                else:
                    raise UnsupportedWorkflow("unsupported f-string format")
            return {"join": parts}
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "join"
            and isinstance(n.func.value, ast.Constant)
            and isinstance(n.func.value.value, str)
            and len(n.args) == 1
            and not n.keywords
        ):
            return {"join_with": [n.func.value.value, expr(n.args[0])]}
        raise UnsupportedWorkflow(f"unsupported expression: {type(n).__name__}")

    def refs(e):
        found = set()
        if isinstance(e, dict):
            if "node" in e:
                found.add(e["node"])
            for v in e.values():
                found.update(refs(v))
        elif isinstance(e, list):
            for v in e:
                found.update(refs(v))
        return found

    def add_call(n, previous):
        if (
            not isinstance(n, ast.Call)
            or not isinstance(n.func, ast.Attribute)
            or ast.unparse(n.func.value) != "self"
            or n.func.attr not in aliases
        ):
            raise UnsupportedWorkflow(
                "only approved self.operator calls may be awaited"
            )
        if n.args or any(k.arg is None for k in n.keywords):
            raise UnsupportedWorkflow("explicit keyword arguments required")
        inputs = {k.arg: expr(k.value) for k in n.keywords}
        op = aliases[n.func.attr]
        required = set(operator_specs[op]) if operator_specs is not None else {
            "GAIAToolAgent": {"input", "instruction"},
            "Custom": {"input", "instruction"},
            "CustomCodeGenerate": {"problem", "entry_point", "instruction"},
            "AnswerGenerate": {"input"},
            "ScEnsemble": {"solutions", "problem"},
            "QAScEnsemble": {"solutions"},
        }[op]
        if set(inputs) != required:
            raise UnsupportedWorkflow(f"{op} arguments differ from native interface")
        ident = f"call_{len(nodes):03d}"
        dependencies = refs(inputs)
        edges.update((d, ident) for d in dependencies)
        barriers.update((p, ident) for p in previous if p not in dependencies)
        nodes.append({"id": ident, "operator": op, "arguments": inputs})
        if len(nodes) > max_nodes:
            raise UnsupportedWorkflow("shared max_nodes exceeded")
        return {"node": ident}

    result = None
    for s in call.body:
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
            continue
        if result is not None:
            raise UnsupportedWorkflow("statements after return")
        if isinstance(s, ast.Return):
            result = (
                add_call(s.value.value, pending)
                if isinstance(s.value, ast.Await)
                else expr(s.value)
            )
            continue
        if not isinstance(s, ast.Assign) or len(s.targets) != 1:
            raise UnsupportedWorkflow(f"dynamic statement: {type(s).__name__}")
        t, v = s.targets[0], s.value
        if isinstance(v, ast.Await):
            inner = v.value
            if (
                isinstance(inner, ast.Call)
                and ast.unparse(inner.func) == "asyncio.gather"
            ):
                if (
                    inner.keywords
                    or not isinstance(t, (ast.Tuple, ast.List))
                    or len(t.elts) != len(inner.args)
                ):
                    raise UnsupportedWorkflow("gather requires fixed unpacking")
                values = [add_call(n, pending) for n in inner.args]
                for target, value in zip(t.elts, values):
                    if not isinstance(target, ast.Name):
                        raise UnsupportedWorkflow("invalid gather target")
                    bindings[target.id] = value
                pending = {value["node"] for value in values}
                continue
            value = add_call(inner, pending)
            pending = {value["node"]}
        else:
            value = expr(v)
        if not isinstance(t, ast.Name):
            raise UnsupportedWorkflow("single-variable assignment required")
        bindings[t.id] = value
    if result is None or not nodes:
        raise UnsupportedWorkflow("calls and explicit return required")
    return {
        "format": "aflow_static_v1",
        "nodes": nodes,
        "data_edges": sorted(map(list, edges)),
        "control_edges": sorted(map(list, barriers)),
        "result": result,
        "source_sha256": digest([graph_source, prompt_source]),
        "operator_library": {
            op: (operator_specs[op] if operator_specs and op in operator_specs else None)
            for op in {n["operator"] for n in nodes}
        },
    }


def expression_value(e, inputs, outputs):
    if "literal" in e:
        return copy.deepcopy(e["literal"])
    if "input" in e:
        return inputs[e["input"]]
    if "node" in e:
        return outputs[e["node"]]
    if "index" in e:
        a, b = e["index"]
        return expression_value(a, inputs, outputs)[
            expression_value(b, inputs, outputs)
        ]
    if "list" in e:
        return [expression_value(v, inputs, outputs) for v in e["list"]]
    if "dict" in e:
        return {k: expression_value(v, inputs, outputs) for k, v in e["dict"].items()}
    if "add" in e:
        a, b = e["add"]
        return expression_value(a, inputs, outputs) + expression_value(
            b, inputs, outputs
        )
    if "str" in e:
        return str(expression_value(e["str"], inputs, outputs))
    if "join" in e:
        return "".join(expression_value(v, inputs, outputs) for v in e["join"])
    if "join_with" in e:
        sep, values = e["join_with"]
        return sep.join(expression_value(values, inputs, outputs))
    raise UnsupportedWorkflow("invalid canonical expression")


def lower_aflow(artifact, public_input, operator_factory, capacity):
    calls, deps, effects = [], [], []
    for node in artifact["nodes"]:
        ident = node["id"]
        incoming = [a for a, b in artifact["data_edges"] if b == ident]
        controls = [a for a, b in artifact["control_edges"] if b == ident]
        outgoing = [
            f"done:{a}->{b}" for a, b in artifact["control_edges"] if a == ident
        ]

        async def execute_node(_node=node, **kwargs):
            arguments = {
                k: expression_value(v, public_input, kwargs)
                for k, v in _node["arguments"].items()
            }
            return {
                "result": await operator_factory(_node["operator"], _node["id"])(
                    **arguments
                )
            }

        for p in incoming:
            deps.append(DataDependency(p, ident, "result", p, allow_early=False))
        for p in controls:
            effects.append(
                EffectDependency(p, ident, f"done:{p}->{ident}", allow_partial=False)
            )
        calls.append(
            CallSpec(
                id=ident,
                target=execute_node,
                input_schema={
                    "type": "object",
                    "properties": {p: {"type": "object"} for p in incoming},
                    "required": incoming,
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "properties": {"result": {"type": "object"}},
                    "required": ["result"],
                },
                effects=tuple(outgoing),
                resources=ResourceVector({"external_call": 1}),
            )
        )
    graph = GFRGCompiler({"external_call": capacity}).compile(calls, deps, effects)
    from .gaia_contracts import annotate_graph
    return annotate_graph(graph, {n["id"]:"gaia_agent" for n in artifact["nodes"] if n["operator"]=="GAIAToolAgent"}, footprint_source="static_export_full_result_bindings")


def graph_topology(artifact):
    if artifact["format"] == "aflow_static_v1":
        ids = {n["id"]: i for i, n in enumerate(artifact["nodes"])}
        return {
            "nodes": [n.get("operator_label", n["operator"]) for n in artifact["nodes"]],
            "edges": sorted(
                {
                    (ids[a], ids[b])
                    for a, b in artifact["data_edges"] + artifact["control_edges"]
                }
            ),
        }
    tasks = artifact["graph"]["tasks"]
    producers = {p["name"]: i for i, t in enumerate(tasks) for p in t["outputs"]}
    edges = {
        (producers[p["name"]], i)
        for i, t in enumerate(tasks)
        for p in t["inputs"]
        if p["name"] in producers
    }
    edges.update((i, i + 1) for i in range(len(tasks) - 1))
    return {"nodes": [t["name"] for t in tasks], "edges": sorted(edges)}


def topology_statistics(topologies, *, timeout=1.0):
    """Mean normalized labeled directed GED; timed-out pairs stay undefined."""
    import time
    from itertools import combinations
    import networkx as nx

    def graph(t):
        g = nx.DiGraph()
        g.add_nodes_from((i, {"label": label}) for i, label in enumerate(t["nodes"]))
        g.add_edges_from(t["edges"])
        return g

    values = []
    timed_out = 0
    pairs = 0
    for a, b in combinations(topologies, 2):
        pairs += 1
        if a == b:
            values.append(0.0)
            continue
        ga, gb = graph(a), graph(b)
        started = time.perf_counter()
        cost = nx.graph_edit_distance(
            ga, gb, node_match=lambda x, y: x["label"] == y["label"], timeout=timeout
        )
        if cost is None or time.perf_counter() - started >= timeout * 0.99:
            timed_out += 1
            continue
        # Explicit unit-edit normalization bounded by deleting/adding both graphs.
        denominator = max(
            1, len(ga) + len(gb) + ga.number_of_edges() + gb.number_of_edges()
        )
        values.append(cost / denominator)
    return {
        "value": sum(values) / len(values) if values and not timed_out else None,
        "pairs": pairs,
        "timed_out": timed_out,
    }


def topology_instability(topologies):
    return topology_statistics(topologies)["value"]
