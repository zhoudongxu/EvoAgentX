"""Deterministic paper study allocations, typed interventions, and offline replay."""
from __future__ import annotations
import asyncio
import copy
import itertools
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

from .benchmarks import _group_tasks, assign_exact_partitions, validate_partitions
from .evolution import atomic_json
from .paper_workflow import validate_spec
from .replay import ReplayBundle, digest, graph_from_replay
from .runtime import CompactFlowRuntime
from .safety import audit_execution

STUDIES = ("construction_main", "construction_diagnostic", "construction_ablations",
           "construction_sensitivity", "execution_main", "execution_ablations",
           "execution_sensitivity", "library_dynamics", "transfer")
MODES = {"sequential": "sequential", "independent_only": "independent_only",
         "complete_dependency": "complete", "percentage_threshold": "percentage_threshold", "guarded": "guarded"}


def manifest_rows(tasks):
    from dataclasses import asdict
    return [{"benchmark": t.benchmark, "task_id": t.task_id, "family_id": t.family_id,
             "family_method": t.metadata.get("family_method", "provided"), "split": t.split,
             "validation_fold": t.metadata.get("validation_fold"), "leakage_keys": t.metadata.get("leakage_keys", []),
             "partition_keys": t.metadata.get("partition_keys", []), "partition_group_id": t.metadata.get("partition_group_id"),
             "dataset_revision": t.metadata.get("dataset_revision"), "level": t.metadata.get("level"),
             "assets": t.metadata.get("gaia_assets", []), "task_digest": digest(asdict(t))}
            for t in sorted(tasks, key=lambda t: (t.benchmark, t.task_id))]


def source_only_pilot(formal_tasks, config, benchmarks):
    """Whole-family DP draws 9/3/3 only from the already frozen formal source."""
    result = []
    seed = config["partition"]["sample_seed"]
    for benchmark in benchmarks:
        source = [t for t in formal_tasks if t.benchmark == benchmark and t.split == "source"]
        groups = sorted(_group_tasks(source), key=lambda g: digest([seed, sorted(t.task_id for t in g)]))
        capacity = tuple(config["partition"]["nominal_counts_per_benchmark"][k] for k in ("source", "validation", "target"))
        states = {(0, 0, 0): []}
        for index, group in enumerate(groups):
            for counts, chosen in list(states.items()):
                for slot in range(3):
                    if counts[slot] + len(group) > capacity[slot]:
                        continue
                    next_counts = list(counts); next_counts[slot] += len(group)
                    states.setdefault(tuple(next_counts), chosen + [(index, slot)])
            if capacity in states:
                break
        if capacity not in states:
            raise ValueError(benchmark + ": cannot form exact source-only pilot without splitting families")
        cohort = []
        for index, _ in states[capacity]:
            for original in groups[index]:
                t = copy.deepcopy(original)
                t.metadata["formal_split"] = "source"
                t.metadata["formal_partition_group_id"] = original.metadata.get("partition_group_id", original.family_id)
                t.metadata.pop("validation_fold", None)
                t.split = ""
                cohort.append(t)
        assign_exact_partitions(cohort, seed=config["partition"]["seed"], source_count=capacity[0],
                                validation_count=capacity[1], target_count=capacity[2], validation_folds=1)
        result.extend(cohort)
    validate_partitions(result)
    blocked = {(t.benchmark, t.task_id) for t in formal_tasks if t.split != "source"}
    if blocked & {(t.benchmark, t.task_id) for t in result}:
        raise ValueError("pilot intersects formal held-out tasks")
    return result


def construction_variants(config, studies):
    variants = {"base": {"kind": "base", "axis": None, "value": None, "config": copy.deepcopy(config)}}
    if "construction_ablations" in studies:
        for name in config["ablations"]["construction"]:
            cfg = copy.deepcopy(config); cfg["study_variant"] = name
            c = cfg["construction"]
            if name in {"without_structural_match", "without_historical_utility"}:
                w = c["retrieval_weights"]
                for key in (["structural"] if name == "without_structural_match" else ["historical", "confidence"]):
                    w[key] = 0.0
                total = sum(w.values())
                c["retrieval_weights"] = {k: v / total for k, v in w.items()}
            variants[name] = {"kind": "ablation", "axis": name, "value": None, "config": cfg}
    if "construction_sensitivity" in studies:
        for axis, field, values in [
            ("quality_tolerance", "quality_tolerance", config["ablations"]["quality_tolerance_sweep"]),
            ("retrieval_k0", "semantic_top_k0", config["ablations"]["retrieval_k0_sweep"]),
            ("selected_k", "max_policies", config["ablations"]["selected_k_sweep"]),
        ]:
            for value in values:
                key = axis + "_" + str(value).replace(".", "p")
                cfg = copy.deepcopy(config); cfg["construction"][field] = value
                cfg["study_variant"] = key
                variants[key] = {"kind": "sensitivity", "axis": axis, "value": value,
                                 "alias": "base" if value == config["construction"][field] else None, "config": cfg}
    return variants


def allocation(config, benchmarks, studies):
    pilot = config["profile"] == "pilot"
    counts = config["partition"]["nominal_counts_per_benchmark"]
    seeds = config["evaluation"]["generation_seeds"]
    return {"schema_version": 1, "studies": list(studies), "benchmarks": list(benchmarks),
            "construction_main": {"tasks": counts["target"], "seeds": seeds,
                                  "methods": len(config.get("baseline_runner", {}).get("required_methods", ("aflow","evoagentx","base_planner","expert_policies","static_library","compactflow"))),
                                  "expected_records": len(benchmarks)*counts["target"]*len(seeds)*len(config.get("baseline_runner", {}).get("required_methods", ("aflow","evoagentx","base_planner","expert_policies","static_library","compactflow")))},
            "construction_diagnostic": {"tasks": 3 if pilot else 50,
                "seeds": [42] if pilot else config["study_allocation"]["construction_diagnostic"]["generation_seeds"],
                "max_interventions": 4 if pilot else 12},
            "execution": {"tasks": 2 if pilot else 25,
                "seeds": [42] if pilot else config["study_allocation"]["execution_diagnostic"]["generation_seeds"],
                "warmups": 1, "repetitions": 2 if pilot else 5},
            "variant_target_tasks": 1 if pilot else counts["target"],
            "external_missing": [] if config.get("paper_experiments", {}).get("llmorch_reimplementation") else ["llmorch"],
            "execution_external": __import__("evoagentx.compactflow.execution_baselines", fromlist=["execution_coverage"]).execution_coverage(),
            "pilot_source_only": pilot}


def _prune(spec):
    nodes = {n["id"]: n for n in spec["nodes"]}
    wanted = set(spec["sinks"]); pending = list(wanted)
    while pending:
        for ref in nodes[pending.pop()]["inputs"].values():
            p = ref.split(".", 1)[0]
            if p != "$input" and p not in wanted:
                wanted.add(p); pending.append(p)
    spec["nodes"] = [n for n in spec["nodes"] if n["id"] in wanted]
    return spec


def motif_signature(spec, kind, touched):
    """Isomorphism-invariant local roles/typed footprints; no wording or entities."""
    nodes = {n["id"]: n for n in spec["nodes"]}
    labels = {k: digest([n["tool"], len(n["outputs"]), len(n["inputs"])]) for k, n in nodes.items()}
    for _ in range(len(nodes)):
        labels = {k: digest([labels[k], sorted(labels[r.split('.')[0]] for r in n['inputs'].values()
                    if r.split('.')[0] in nodes)]) for k,n in nodes.items()}
    roles = sorted([nodes[k]["tool"], len(nodes[k]["inputs"]), len(nodes[k]["outputs"]), labels[k]] for k in touched if k in nodes)
    return {"operation": kind, "roles": roles, "signature": digest([kind, roles])}


def interventions(spec, public, limit=12):
    proposals, seen, reasons = [], set(), {}
    nodes = {n["id"]: n for n in spec["nodes"]}
    def add(kind, proposed, touched):
        try:
            proposed = _prune(proposed)
            validate_spec(proposed, public, set(), 12)
        except (ValueError, KeyError, TypeError):
            return
        key = digest(proposed)
        if key == digest(spec) or key in seen:
            return
        seen.add(key)
        proposals.append({"kind": kind, "workflow": proposed, "nodes": touched,
                          "motif": motif_signature(spec, kind, touched)})
    for n in spec["nodes"]:
        if n["tool"] == "llm" and n["id"] not in spec["sinks"] and len(n["outputs"]) == len(n["inputs"]) == 1:
            ref = n["id"]+"."+n["outputs"][0]; replacement = next(iter(n["inputs"].values()))
            trial = copy.deepcopy(spec)
            for child in trial["nodes"]:
                child["inputs"] = {k: replacement if v == ref else v for k,v in child["inputs"].items()}
            trial["nodes"] = [x for x in trial["nodes"] if x["id"] != n["id"]]
            add("node_bypass", trial, [n["id"]])
        if n["tool"] == "llm" and len(n["inputs"]) > 1:
            for arg, ref in sorted(n["inputs"].items()):
                if ref.startswith("$input."):
                    continue
                trial = copy.deepcopy(spec)
                next(x for x in trial["nodes"] if x["id"] == n["id"])["inputs"].pop(arg)
                add("edge_removal", trial, [ref.split('.')[0], n["id"]])
        if n["id"] not in spec["sinks"] and "answer" in n["outputs"]:
            trial = copy.deepcopy(spec); trial["sinks"] = [n["id"]]
            add("early_stopping", trial, [n["id"], spec["sinks"][0]])
    for a,b in itertools.permutations(spec["nodes"], 2):
        if a["tool"] != "llm" or b["tool"] != "llm" or a["id"] in spec["sinks"]:
            continue
        if not any(v.startswith(a["id"]+".") for v in b["inputs"].values()):
            continue
        if any(v.startswith(a["id"]+".") for n in spec["nodes"] if n["id"] != b["id"] for v in n["inputs"].values()):
            continue
        trial = copy.deepcopy(spec); fused = next(n for n in trial["nodes"] if n["id"] == b["id"])
        external = {"upstream_"+k:v for k,v in a["inputs"].items()}
        external.update({k:v for k,v in b["inputs"].items() if not v.startswith(a["id"]+".")})
        fused["inputs"] = external
        fused["instruction"] = "Perform these two pure reasoning steps in order in one call. First: " + a["instruction"] + ". Then: " + b["instruction"] + ". Return the downstream fields only. Upstream inputs use the upstream_ prefix. Internal bindings: " + json.dumps(b["inputs"], sort_keys=True)
        trial["nodes"] = [n for n in trial["nodes"] if n["id"] != a["id"]]
        add("node_fusion", trial, [a["id"], b["id"]])
    # Round-robin operation classes avoids the budget being consumed by one kind.
    kinds = ["node_bypass", "edge_removal", "node_fusion", "early_stopping"]
    selected = []
    buckets = {k: sorted([p for p in proposals if p["kind"] == k], key=lambda p:digest(p["workflow"])) for k in kinds}
    for k in kinds:
        if not buckets[k]: reasons[k] = "no type-compatible eligible structure"
    while any(buckets.values()) and len(selected) < limit:
        for k in kinds:
            if buckets[k] and len(selected) < limit: selected.append(buckets[k].pop(0))
    return selected, reasons


def replay_cases(config, studies):
    result = []
    if "execution_main" in studies:
        result += [{"study":"execution_main", "name": k, "mode": v} for k,v in MODES.items()]
        if config.get("paper_experiments", {}).get("native_llmcompiler", True):
            result.append({"study":"execution_main", "name":"llmcompiler", "backend":"llmcompiler", "mode":"complete"})
        if config.get("paper_experiments", {}).get("llmorch_reimplementation", False):
            result.append({"study":"execution_main", "name":"llmorch", "backend":"llmorch", "mode":"complete"})
    if "execution_ablations" in studies:
        result += [{"study":"execution_ablations", "name":k, "mode":"complete" if k=="without_partial" else "guarded"}
                   for k in config["ablations"]["execution"]]
    if "execution_sensitivity" in studies:
        for axis, values in config["execution"]["sweeps"].items():
            result += [{"study":"execution_sensitivity", "name":axis+"_"+str(v), "axis":axis,"value":v,
                        "mode":"percentage_threshold" if axis=="percentage_threshold" else "guarded"} for v in values]
    return result


def readiness(bundle, graph):
    records = {r["call_id"]:r for r in bundle.records.values()}
    gaps = []
    for dep in graph.data_dependencies:
        if dep.producer is None or not graph.has_guard(dep) or dep.producer not in records: continue
        events = records[dep.producer]["events"]
        complete = next((e['at'] for e in events if e['kind']=='complete'), None)
        ready = next((e['at'] for e in events if e['kind']=='partial' and dep.source_path in e.get('stable_fields', [])), None)
        if complete is not None and ready is not None and complete > 0:
            gaps.append(max(0., (complete-ready)/complete))
    return {"eligible_edges":len(gaps), "opportunity":any(g>0 for g in gaps), "normalized_readiness_gaps":gaps}


async def replay_one(record, public, config, case):
    """Only binds recorded outputs. Disabled guards cannot invoke a live target."""
    if not record.get("graph"):
        return {"status":"incomplete", "reason":"missing captured graph"}
    raw = record["replay"]
    bundle = ReplayBundle(raw["workflow_id"], records=raw["records"], metadata=raw.get("metadata"))
    original = graph_from_replay(record["graph"], bundle)
    graph_data = copy.deepcopy(record["graph"])
    name, axis = case["name"], case.get("axis")
    capacity = dict(graph_data["resource_capacity"])
    if axis == "external_call_capacity": capacity["external_call"] = case["value"]
    if name == "without_effect_guard": graph_data["effect_dependencies"] = []
    if name == "without_resource_guard":
        capacity = {k:sum(c["resources"].get(k,0) for c in graph_data["calls"])+1 for k in capacity}
    if name == "coarse_footprint":
        for dep in graph_data["data_dependencies"]: dep["allow_early"] = False
    batch = case["value"] if axis == "materialization_batch_size" else config["execution"]["materialization_batch_size"]
    threshold = case["value"] if axis == "percentage_threshold" else config["execution"]["percentage_threshold"]
    compilation_metrics = {}
    started = time.perf_counter()
    graph = graph_from_replay(graph_data, bundle, capacity=capacity, batch_size=batch,
                              compilation_metrics=compilation_metrics)
    compilation = time.perf_counter()-started
    provenance = None
    if case.get("backend") in {"llmcompiler", "llmorch"}:
        from .execution_baselines import replay_llmcompiler, replay_llmorch, llmorch_identity, upstream_identity
        kwargs = {}
        if case['backend'] == 'llmorch':
            settings=config.get('paper_experiments', {}).get('llmorch', {})
            kwargs={'processors':settings.get('processors', config['execution']['external_call_capacity']),
                    'kinds':{c.id:settings.get('call_kinds', {}).get(c.id, settings.get('default_kind', 'inout')) for c in graph.calls}}
            provenance={**llmorch_identity(), **kwargs}
            runner=replay_llmorch
        else:
            runner=replay_llmcompiler;provenance=upstream_identity()
        result = await runner(graph, public, sinks=tuple(record["graph"]["sink_ids"]),
            call_timeout=config['execution']['call_timeout_seconds'], workflow_timeout=config['execution']['workflow_timeout_seconds'], **kwargs)
    else:
        runtime = CompactFlowRuntime(graph, mode=case["mode"], sink_ids=tuple(record["graph"]["sink_ids"]),
            percentage_threshold=threshold, profile_controls=True, call_timeout=config['execution']['call_timeout_seconds'],
            workflow_timeout=config['execution']['workflow_timeout_seconds'])
        result = await runtime.execute(public)
    audit_graph = original if name in {"without_effect_guard", "without_resource_guard"} else graph
    safety = audit_execution(audit_graph, result, reference_arguments=record.get('arguments'))
    answer = result.outputs.get(record['graph']['sink_ids'][0],{}).get('answer','')
    bounded = ('timeout', 'timed out', 'budget exhausted', 'step budget', 'tool-call budget')
    captured_bounded_failure = bool(record.get('errors')) and all(any(t in str(error).lower() for t in bounded) for error in record['errors'].values())
    replayed_bounded_failure = captured_bounded_failure and bool(result.errors) and all(any(t in str(error).lower() for t in bounded) for error in result.errors.values())
    success = (not result.errors or replayed_bounded_failure) and safety['assessment_complete']
    profile_version = result.metrics.control_profile_version
    overhead = ({**compilation_metrics, **result.metrics.control_seconds}
                if profile_version is not None else None)

    return {"status":"complete" if success else "incomplete", "errors":dict(result.errors),
            "scheduler_provenance":provenance,
            "latency":result.metrics.latency, "ttfo":result.metrics.ttfo, "answer":answer,
            "outputs_equal_capture":answer==record.get('answer'), "safety":safety,
            "readiness":readiness(bundle, original), "compilation_seconds":compilation,
            "static_analysis_seconds":compilation_metrics['static_analysis'],
            "graph_setup_seconds":compilation-compilation_metrics['static_analysis'],
            "control_profile_version":profile_version,
            "control_overhead":overhead,
            "quality":record.get('evaluation',{}).get('quality') if success and answer==record.get('answer') else None}
