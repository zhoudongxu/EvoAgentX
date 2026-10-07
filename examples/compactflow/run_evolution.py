"""Run a pinned local raw-data bundle through CompactFlow policy evolution."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from jsonschema import Draft202012Validator

from evoagentx.compactflow.benchmarks import BenchmarkTask, assign_exact_partitions, evaluate, import_dataset, validate_partitions
from evoagentx.compactflow.construction import AdmissionConfig, CostWeights
from evoagentx.compactflow.evolution import DistillationResult, EvolutionConfig, EvolutionRunner, VariantResult
from evoagentx.compactflow.llm import ModelClient
from evoagentx.compactflow.models import CompactnessPolicy, Evidence, ExecutionFeedback, PolicyStatus
from evoagentx.compactflow.paper_workflow import compile_spec, plan_workflow, prompt
from evoagentx.compactflow.policy import CompatibilitySelector, PolicyLibrary, PolicyRetriever, RetrievalConfig, SelectionConfig
from evoagentx.compactflow.reproduction_config import load_reproduction_config
from evoagentx.compactflow.runtime import CompactFlowRuntime
from evoagentx.compactflow.replay import ReplayBundle, canonical, digest, graph_to_dict
from evoagentx.compactflow.safety import audit_execution
from examples.compactflow.prepare_benchmark_trial import mbpp_interface, select_tasks

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path):
    if path.suffix in {".json", ".jsonl"}:
        text = path.read_text()
        return json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    raise ValueError(f"unsupported raw data format: {path.suffix}")


@dataclass
class RawBundle:
    tasks: list[BenchmarkTask]
    identity: dict
    coverage: dict


def load_raw_bundle(data_dir, config, *, frozen_manifest=None):
    data_dir = Path(data_dir).resolve()
    manifest = data_dir / "sources.json"
    if not manifest.exists():
        raise FileNotFoundError(f"local raw bundle requires {manifest}; runner never downloads data")
    sources = json.loads(manifest.read_text())
    result, locks, coverage = [], [], {}
    frozen = [json.loads(line) for line in Path(frozen_manifest).read_text().splitlines() if line.strip()] if frozen_manifest else None
    counts = config["partition"]["nominal_counts_per_benchmark"]
    folds = config.get("runner", {}).get("validation_folds", config["construction"].get("validation_folds", 5))
    for dataset in config["datasets"]:
        name = dataset["name"]
        wanted = config.get("runner", {}).get("benchmarks")
        if (frozen is None and wanted and name not in wanted) or (frozen is not None and not any(r["benchmark"] == name for r in frozen)):
            coverage["dataset:" + name] = {"status":"incomplete", "reason":"outside explicitly frozen pilot benchmarks", "raw_pool_count":0}
            continue
        entries = [e for e in sources if e["benchmark"] == name]
        reason, pool = None, []
        if not entries:
            reason = "missing authorized local raw data"
        for entry in entries:
            relative = entry["repository_relative_path"]
            path = (data_dir / name / relative).resolve()
            if not path.is_relative_to(data_dir / name):
                raise ValueError("raw path escapes benchmark directory")
            url = urlparse(entry["url"])
            expected = f"/datasets/{dataset['repository']}/resolve/{dataset['revision']}/{relative}"
            if url.scheme != "https" or url.netloc != "huggingface.co" or url.path != expected:
                raise ValueError(f"raw dataset revision/path mismatch for {name}")
            if not path.exists():
                reason = f"missing pinned file: {relative}"
                locks.append({**entry, "available": False})
                continue
            if _sha256(path) != entry["sha256"] or ("bytes" in entry and path.stat().st_size != entry["bytes"]):
                raise ValueError(f"raw checksum/size mismatch: {name}/{relative}")
            locks.append({**entry, "available": True})
            original_split = entry.get("original_split")
            if original_split is None:
                matches = [s for s in dataset["original_splits"] if s in path.name or s in path.parts]
                if len(dataset["original_splits"]) == 1:
                    original_split = dataset["original_splits"][0]
                elif len(matches) == 1:
                    original_split = matches[0]
                else:
                    raise ValueError(f"{name}: original_split required in sources.json")
            if original_split not in dataset["original_splits"]:
                raise ValueError(f"{name}: original_split differs from pinned protocol")
            raw = _read_rows(path)
            try:
                tasks = import_dataset(name, raw, revision=dataset["revision"], original_split=original_split)
                for task, row in zip(tasks, raw, strict=True):
                    task.metadata.update(subject=row.get("type", ""), level=row.get("Level", row.get("level", "")))
                    if name == "GAIA":
                        task.metadata["level"] = int(row["Level"])
                    if name == "MBPP":
                        interface = mbpp_interface(row["code"], row["test_list"])
                        if not interface:
                            raise ValueError("cannot determine safe MBPP function interface")
                        task.metadata["original_question_sha256"] = digest(task.question)
                        task.question += "\nRequired function interface(s):\n" + "\n".join(interface)
                pool.extend(tasks)
            except (ValueError, KeyError, TypeError) as exc:
                reason = f"raw import incomplete: {exc}"
        if name == "GAIA" and reason is None:
            from evoagentx.compactflow.gaia import attach_assets, GaiaUnavailable
            try:
                assets = attach_assets(pool, data_dir / "GAIA" / "2023" / "validation")
                locks.append({"benchmark":"GAIA", "attachment_inventory":assets})
            except (ValueError, GaiaUnavailable) as exc:
                reason = str(exc)
        if reason is None:
            try:
                validate_partitions(pool)
                if frozen is None:
                    selected = select_tasks(pool, int(dataset.get("sample_count", 150)),
                                            config["partition"]["sample_seed"], stratified=name == "MATH")
                    assign_exact_partitions(
                        selected, seed=config["partition"]["seed"], source_count=counts["source"],
                        validation_count=counts["validation"], target_count=counts["target"],
                        validation_folds=folds)
                else:
                    from collections import Counter
                    rows = [r for r in frozen if r["benchmark"] == name]
                    indexed = {t.task_id: t for t in pool}
                    sizes = Counter(r["partition_group_id"] for r in rows)
                    selected = []
                    for row in rows:
                        task = indexed[str(row["task_id"])]
                        task.split = row["split"]
                        task.metadata.update(partition_seed=config["partition"]["seed"],
                            partition_group_id=row["partition_group_id"],
                            partition_group_size=sizes[row["partition_group_id"]],partition_keys=row["partition_keys"])
                        if row.get("validation_fold") is not None:
                            task.metadata["validation_fold"] = row["validation_fold"]
                        selected.append(task)
                    if Counter(t.split for t in selected) != counts:
                        raise ValueError("frozen manifest counts differ from locked protocol")
                result.extend(selected)
            except ValueError as exc:
                reason = f"whole-family cohort incomplete: {exc}"
        coverage["dataset:" + name] = {"status": "incomplete" if reason else "complete",
                                     "reason": reason, "raw_pool_count": len(pool)}
    return RawBundle(result, {"sources_manifest_sha256": _sha256(manifest), "files": locks}, coverage)


def load_raw_tasks(data_dir, config):
    bundle = load_raw_bundle(data_dir, config)
    return bundle.tasks, bundle.identity


def build_admission(config):
    c = config["construction"]
    w = c["cost_weights"]
    return AdmissionConfig(quality_tolerance=c["quality_tolerance"], minimum_cost_reduction=c["minimum_cost_reduction"],
                           merge_similarity_threshold=c["merge_similarity_threshold"], minimum_candidate_valid_rate=c["minimum_candidate_valid_rate"],
                           normalization_floor=c["normalization_floor"], cost_weights=CostWeights(w["tokens"], w["latency"], w["graph"]),
                           require_matching_pair_keys=True, require_valid_baseline=True)


class PinnedEmbedder:
    """Lazy, local-only semantic encoder; never silently substitutes a hash model."""
    def __init__(self, config):
        self.config, self.model, self.cache = config, None, {}

    def embed(self, text):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            if self.config["backend"] != "sentence_transformers":
                raise ValueError("unsupported encoder backend")
            self.model = SentenceTransformer(self.config["repository"], revision=self.config["revision"],
                                             device=self.config["device"], local_files_only=True)
        if text not in self.cache:
            self.cache[text] = tuple(float(x) for x in self.model.encode(
                text, normalize_embeddings=self.config["normalize_embeddings"]))
            if len(self.cache[text]) != self.config["dimension"]:
                raise ValueError("encoder dimension differs from pinned configuration")
        return self.cache[text]


class LiveAdapter:
    def __init__(self, config, library, *, output=None, client_factory=None):
        self.config, self.library, self.output = config, library, Path(output) if output else None
        self.client_factory = client_factory or ModelClient

    async def _select(self, client, task, policies, seed):
        if not policies:
            return (), {}
        c = self.config["construction"]
        response = await client.json(prompt("query"), canonical({"task": task.public_input(), "tools": self.config["tools"]["registry"]}),
                                     component="query", seed=seed)
        if not isinstance(response.get("query"), str):
            raise ValueError("malformed structural query")
        w = c["retrieval_weights"]
        library = PolicyLibrary(policies, embedder=self.library.embedder)
        matches = PolicyRetriever(library, config=RetrievalConfig(
            semantic_top_k0=c["semantic_top_k0"], top_k=c["top_k"], semantic_weight=w["semantic"],
            applicability_weight=w["structural"], utility_weight=w["historical"], confidence_weight=w["confidence"],
            minimum_semantic_score=c["minimum_semantic_score"], minimum_applicability=c["minimum_applicability"],
            include_candidates=True)).retrieve(response["query"], {
                "benchmark": task.benchmark, "capabilities": {"typed_workflow": True},
                "tools": self.config["tools"]["registry"]})
        if self.config.get("study_variant") == "without_policy_selection":
            chosen = tuple(m.policy for m in matches if m.applicability_score >= c["minimum_applicability"])[:c["max_policies"]]
            return chosen, {"query": response["query"], "selected": [p.id for p in chosen], "selection_ablation": True}
        selector = CompatibilitySelector(SelectionConfig(max_policies=c["max_policies"],
            minimum_score=c["minimum_selection_score"], minimum_applicability=c["minimum_applicability"]))
        selection_result = selector.select(matches)
        allowed = selection_result.policies
        retrieval = {"retrieved_policy_ids": [m.policy.id for m in matches],
                     "compatible_policy_ids": [p.id for p in allowed],
                     "declared_conflict_pairs": sum(bool(set(a.policy.conflicts_with) & {b.policy.id} or set(b.policy.conflicts_with) & {a.policy.id})
                         for i,a in enumerate(matches) for b in matches[i+1:])}
        if not allowed:
            return (), {"query": response["query"], "selected": [], **retrieval}
        choice = await client.json(prompt("select"), canonical({
            "task": task.public_input(), "candidate_policies": [p.to_dict() for p in allowed], "max_policies": c["max_policies"]}),
            component="selector", seed=seed)
        ids = choice.get("selected")
        if not isinstance(ids, list) or any(not isinstance(i,str) for i in ids) or len(ids) != len(set(ids)) or set(ids)-{p.id for p in allowed}:
            raise ValueError("selector returned invalid policy IDs")
        return tuple(next(p for p in allowed if p.id == i) for i in ids), {"query":response["query"], **choice, **retrieval}

    async def run_variant(self, task, policies, seed, variant, split, *, workflow_spec=None, workflow_observer=None):
        from evoagentx.compactflow.paper_phase import assert_target_allowed
        assert_target_allowed(self.config, split, task.benchmark)
        from evoagentx.compactflow.baseline_native import ModelSession
        from evoagentx.compactflow.baseline_controls import SessionClient
        gaia = None
        if self.output is not None:
            workspace = self.output / "runtime_calls" / digest([task.benchmark,task.task_id,split,seed,variant,[p.to_dict() for p in policies],workflow_spec])
            session = ModelSession(self.config,seed,workspace / "model_calls",client_factory=self.client_factory)
            client = SessionClient(session)
        else:
            client = self.client_factory(self.config["model"], token_budget=self.config["evaluation"]["task_token_budget"])
        if task.benchmark == "GAIA":
            from evoagentx.compactflow.gaia import GaiaToolSession
            if self.output is None:
                raise ValueError("GAIA execution requires a durable output directory")
            gaia = GaiaToolSession(self.config,task.public_input(),output=self.output,workspace=workspace,seed=seed,model_session=session)
        started = time.perf_counter()
        planning, selection, spec, score, safety = {}, {}, {}, {}, {}
        graph = result = None
        answer, errors, selected, applied = "", {}, (), ()
        infrastructure_error = False
        bundle = ReplayBundle(f"{task.benchmark}:{task.task_id}:{seed}:{variant}")
        nodes, edges, quality, valid = None, None, 0., False
        try:
            if workflow_spec is None:
                selected, selection = await self._select(client, task, policies, seed)
                spec, planning = await plan_workflow(client, task, list(selected), self.config["construction"]["planner"], seed=seed)
            else:
                from evoagentx.compactflow.paper_workflow import validate_spec
                spec = workflow_spec
                validate_spec(spec, task.public_input(), set(), self.config["construction"]["planner"]["max_nodes"])
            if workflow_observer is not None:
                workflow_observer(spec)
            applied = tuple(p["id"] if isinstance(p,dict) else p for p in spec["applied_policies"])
            graph = compile_spec(spec, task.public_input(), client, seed=seed,
                                 capacity=self.config["execution"]["external_call_capacity"], recorder=bundle, tool_session=gaia)
            result = await CompactFlowRuntime(graph, mode="complete", sink_ids=spec["sinks"],
                call_timeout=self.config["execution"]["call_timeout_seconds"],
                workflow_timeout=self.config["execution"]["workflow_timeout_seconds"]).execute(task.public_input())
            errors = dict(result.errors)
            safety = audit_execution(graph, result)
            answer = result.outputs.get(spec["sinks"][0],{}).get("answer","")
            score = await asyncio.to_thread(evaluate,task,answer,sandbox=self.config["tools"].get("mbpp_sandbox"))
            # Failed model/contract executions remain in the denominator, but
            # their quality contribution is zero per the locked metric policy.
            quality = float(score["quality"]) if not errors else 0.0
            nodes, edges = len(spec["nodes"]), sum(d.producer is not None for d in graph.data_dependencies)
            valid = not errors and safety["assessment_complete"] and not safety["all_call_failure_incidents"]
            if variant == "candidate_test":
                candidates = [p.id for p in policies if p.status == PolicyStatus.CANDIDATE]
                if any(p not in applied for p in candidates):
                    valid = False
                    errors["candidate_application"] = "candidate was not applied by the validated planner"
        except Exception as exc:
            errors["pipeline"] = f"{type(exc).__name__}: {exc}"
            from evoagentx.compactflow.baselines import BaselineUnavailable
            from evoagentx.compactflow.gaia import GaiaUnavailable
            infrastructure_error = isinstance(exc, (BaselineUnavailable, GaiaUnavailable, ImportError, OSError))
            valid = False
        accounting = client.accounting()
        infrastructure_error |= bool(gaia and gaia.incomplete)
        valid = valid and accounting["usage_complete"] and not infrastructure_error
        feedback = ExecutionFeedback(quality=quality, token_cost=float(accounting["total_tokens"]),
            latency=time.perf_counter()-started, node_count=nodes, edge_count=edges, valid=valid,
            contract_violations=tuple(errors), metrics={"usage_complete":accounting["usage_complete"], "infrastructure_complete":not infrastructure_error})
        evidence = Evidence(id=digest([task.benchmark,task.task_id,split,seed,variant,[p.id for p in policies]]),
            benchmark=task.benchmark, task_id=task.task_id, split=split, variant=variant, seed=seed,
            policy_ids=applied, feedback=feedback, metadata={"family_id":task.family_id})
        return VariantResult(evidence,{
            "graph": graph_to_dict(graph, spec["sinks"]) if graph else None,
            "runtime_metrics": {"latency":result.metrics.latency, "ttfo":result.metrics.ttfo, "peak_resources":result.metrics.peak_resources} if result else {},
            "arguments": {k: dict(v.arguments) for k,v in result.call_traces.items()} if result else {},
            "trace": [{"at":e.timestamp-result.metrics.started_at,"kind":e.kind.value,"call_id":e.call_id,"detail":dict(e.detail)} for e in result.trace] if result else [],
            "answer":answer,"errors":errors,"planning":planning,"selection":selection,"workflow":spec,
            "execution_status":"complete" if valid else "failed",
            "infrastructure_status":"incomplete" if infrastructure_error else "complete",
            "tokens":accounting,"evaluation":score,"safety":safety,"model_requests":client.records,
            "level":task.metadata.get("level"),"tool_accounting":gaia.accounting() if gaia else {},
            "replay":{"schema_version":1,"workflow_id":bundle.workflow_id,"metadata":bundle.metadata,"records":bundle.records}, "selected_policy_ids":[p.id for p in selected]})

    async def distill_candidate(self, task, baseline, current, round_index):
        if self.output is not None:
            from evoagentx.compactflow.baseline_native import ModelSession
            from evoagentx.compactflow.baseline_controls import SessionClient
            key = digest([task.benchmark, task.task_id, round_index, baseline.evidence.to_dict(), current.evidence.to_dict()])
            client = SessionClient(ModelSession(self.config, self.config["runner"]["source_seed"] + round_index,
                self.output / "distillation_calls" / key, client_factory=self.client_factory))
        else:
            client = self.client_factory(self.config["model"], token_budget=self.config["evaluation"]["task_token_budget"])
        response, contract, candidate, reason = None, None, None, ""
        try:
            response = await client.json(prompt("distill"), canonical({
                "task":task.public_input(), "source_execution": {
                    "workflow":current.record.get("workflow"), "answer":current.record.get("answer"),
                    "planning":current.record.get("planning")},
                "source_feedback":current.evidence.feedback.to_dict(),
                "paired_base_feedback":baseline.evidence.feedback.to_dict(), "round":round_index}),
                component="distiller", seed=self.config["runner"]["source_seed"]+round_index)
            schema = json.loads((ROOT/"examples/compactflow/schemas/candidate.schema.json").read_text())
            Draft202012Validator(schema).validate(response)
            data = response["candidate"]
            if data is not None:
                contract = await client.json(prompt("contract"),canonical({
                    "task":task.public_input(), "candidate":data, "tools":self.config["tools"]["registry"],
                    "library":[{"id":p.id,"description":p.description,"operation":p.operation,
                                "precondition":p.precondition} for p in self.library.all() if p.status == PolicyStatus.VERIFIED]}),
                    component="checker", seed=self.config["runner"]["source_seed"]+round_index+1000)
                if contract.get("contract_valid") is not True or contract.get("violations") != []:
                    raise ValueError("contract checker rejected candidate or returned malformed output")
                candidate = CompactnessPolicy(
                    id="policy-"+digest([data,round_index,task.benchmark,task.task_id])[:20],
                    description=data["description"],precondition=data["precondition"],operation=data["operation"],
                    expected_effect=data["expected_effect"],status=PolicyStatus.CANDIDATE,
                    metadata={"failure_modes":data["failure_modes"],"contract_check":contract,"source_task_id":task.task_id})
            else:
                reason = "distiller_returned_no_candidate"
        except Exception as exc:
            reason = f"distillation_or_contract_failed: {type(exc).__name__}: {exc}"
        return DistillationResult(candidate, {"response":response,"contract":contract,
            "tokens":client.accounting(),"model_requests":client.records},reason)


def coverage_for(config):
    coverage = {"base_planner":{"status":"complete","implementation":"in_house"},
                "compactflow":{"status":"complete","implementation":"evolution_runner"}}
    for name, value in config.get("baselines",{}).items():
        if name in coverage or name == "shared" or not isinstance(value,dict):
            continue
        if name == "external_execution":
            for child, item in value.items():
                coverage[child] = {"status":"incomplete","reason":item.get("integration_status","not executed"),"source":item.get("source")}
        else:
            coverage[name] = {"status":"incomplete","reason":value.get("integration_status","not executed by evolution runner"),"source":value.get("source")}
    return coverage


def observe_service(config):
    base = os.environ.get("COMPACTFLOW_API_BASE", config["model"]["base_url"]).rstrip("/")
    headers = {}
    key = os.environ.get(config["model"]["api_key_env"],"")
    if key:
        headers["Authorization"] = "Bearer "+key
    with urllib.request.urlopen(urllib.request.Request(base+"/models",headers=headers),timeout=10) as r:
        models = json.load(r)
    model = next((m for m in models["data"] if m["id"] == config["model"]["name"]), None)
    if not model:
        raise ValueError("configured model is not served at the model endpoint")
    if model.get("max_model_len",config["model"]["context_length"]) < config["model"]["context_length"]:
        raise ValueError("server context is smaller than configured client bound")
    return {"base_url":base,"model":model["id"],"served_root":model.get("root"),
            "max_model_len":model.get("max_model_len"),"client_context_bound":config["model"]["context_length"]}


async def run(args):
    config = load_reproduction_config(args.config,root=ROOT)
    if getattr(args,"benchmarks",None):
        config["runner"]["benchmarks"] = args.benchmarks.split(",")
    bundle = load_raw_bundle(args.data_dir,config)
    if any(t.benchmark == "GAIA" for t in bundle.tasks):
        from evoagentx.compactflow.gaia import capability_report
        capability = capability_report(config)
        bundle.coverage["capability:GAIA"] = capability
        if capability["status"] != "complete":
            bundle.coverage["dataset:GAIA"] = {**bundle.coverage["dataset:GAIA"], "status": "incomplete",
                                               "reason": "; ".join(capability["reasons"])}
            bundle.tasks = [t for t in bundle.tasks if t.benchmark != "GAIA"]
    if bundle.tasks:
        config["observed_service"] = observe_service(config)
    library = PolicyLibrary.load(ROOT/config["construction"]["library"]["initial_path"],embedder=PinnedEmbedder(config["encoder"]))
    if bundle.tasks:
        library.embedder.embed("CompactFlow local encoder preflight")
    adapter = LiveAdapter(config,library,output=args.output)
    runner = EvolutionRunner(
        bundle.tasks,config=EvolutionConfig.from_mapping(config),library=library,output=args.output,
        run_variant=adapter.run_variant,distill_candidate=adapter.distill_candidate,admission=build_admission(config),
        coverage={**coverage_for(config),**bundle.coverage},config_payload=config,data_identity=bundle.identity)
    # The adapter uses the same embedder; the runner restores its own library on resume.
    original_distill = adapter.distill_candidate
    async def distill(*args):
        adapter.library = runner.library
        return await original_distill(*args)
    runner.distill_candidate = distill
    result = await runner.run(resume=args.resume, stage=getattr(args, "stage", "all"))
    print(json.dumps(result.summary,indent=2,ensure_ascii=False))
    return 0 if result.summary["publication_status"] == "complete" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    for command in ("run","validate"):
        p = sub.add_parser(command)
        p.add_argument("--config",type=Path,required=True)
        p.add_argument("--data-dir",type=Path,required=True)
        p.add_argument("--benchmarks",help="Explicit pilot subset; does not constitute full four-benchmark coverage")
        if command == "run":
            p.add_argument("--output",type=Path,required=True)
            p.add_argument("--resume",action="store_true")
            p.add_argument("--stage", choices=["all", "freeze", "target"], default="all")
    args = parser.parse_args()
    if args.command == "validate":
        config = load_reproduction_config(args.config,root=ROOT)
        if args.benchmarks:
            config["runner"]["benchmarks"] = args.benchmarks.split(",")
        EvolutionConfig.from_mapping(config)
        bundle = load_raw_bundle(args.data_dir,config)
        print(json.dumps({"tasks":len(bundle.tasks),"coverage":bundle.coverage},indent=2))
        return 0 if all(v["status"] == "complete" for v in bundle.coverage.values()) else 2
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
