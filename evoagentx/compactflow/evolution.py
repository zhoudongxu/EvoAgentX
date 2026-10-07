"""Auditable source discovery, round-disjoint validation, and frozen target runs.

Budgets and folds are per benchmark. Completed calls are cached atomically; the
checkpoint commits library changes and decisions together. An interrupted model
request without a durable result may need to run again.
"""
from __future__ import annotations
from datetime import datetime, timezone

import collections
import copy
import csv
import fcntl
import hashlib
import inspect
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from .benchmarks import BenchmarkTask, _group_tasks, validate_partitions
from .construction import AdmissionConfig, CostWeights, PairedExecution, PairedPolicyAdmission, ParetoArchive
from .models import AdmissionVerdict, CompactnessPolicy, Evidence, ExecutionFeedback, PolicyStatus
from .policy import PolicyLibrary


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
        temporary = Path(f.name)
    temporary.replace(path)


@dataclass(frozen=True)
class EvolutionConfig:
    bootstrap_tasks: int = 20
    rounds: int = 5
    source_tasks_per_round: int = 10
    max_candidates_per_task: int = 1
    max_candidates_per_round: int = 4
    validation_folds: int = 5
    validation_tasks_per_candidate: int = 6
    heldout_seeds: tuple[int, ...] = (101, 202, 303)
    target_seeds: tuple[int, ...] = (42, 43, 44)
    source_seed: int = 42
    quality_tolerance: float = 0.0
    minimum_cost_reduction: float = 0.05
    minimum_candidate_valid_rate: float = 1.0
    cost_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)
    other_cost_tolerance: float = 0.05
    library_capacity: int = 256

    def __post_init__(self):
        if self.rounds < 1 or self.validation_folds != self.rounds:
            raise ValueError("one disjoint validation fold is required per evolution round")
        if self.bootstrap_tasks < 0 or min(self.source_tasks_per_round, self.max_candidates_per_round, self.validation_tasks_per_candidate, self.library_capacity) < 1:
            raise ValueError("evolution task/candidate budgets must be positive")
        if self.max_candidates_per_task != 1:
            raise ValueError("this runner supports exactly one candidate at most per source task")
        for seeds in (self.heldout_seeds, self.target_seeds):
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("seeds must be nonempty and unique")
        CostWeights(*self.cost_weights)
        if self.other_cost_tolerance < 0:
            raise ValueError("other_cost_tolerance must be non-negative")

    @classmethod
    def from_mapping(cls, config):
        c, r = config["construction"], config.get("runner", {})
        d = c["distillation"]
        values = {k: r.get(k, d.get(k, getattr(cls(), k))) for k in (
            "rounds", "source_tasks_per_round", "max_candidates_per_task", "max_candidates_per_round")}
        w = c["cost_weights"]
        return cls(
            **values,
            bootstrap_tasks=r.get("bootstrap_tasks", c["library"]["bootstrap_tasks"]),
            validation_folds=r.get("validation_folds", c.get("validation_folds", 5)),
            validation_tasks_per_candidate=r.get("heldout_tasks_per_candidate", c["heldout_tasks_per_candidate"]),
            heldout_seeds=tuple(r.get("heldout_seeds", c["heldout_seeds"])),
            target_seeds=tuple(r.get("target_seeds", config["evaluation"]["generation_seeds"])),
            source_seed=r.get("source_seed", config["partition"]["seed"]),
            quality_tolerance=c["quality_tolerance"],
            minimum_cost_reduction=c["minimum_cost_reduction"],
            minimum_candidate_valid_rate=c["minimum_candidate_valid_rate"],
            cost_weights=(w["tokens"], w["latency"], w["graph"]),
            other_cost_tolerance=c.get("other_cost_tolerance", .05),
            library_capacity=c["library"]["capacity"],
        )


@dataclass
class VariantResult:
    evidence: Evidence
    record: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillationResult:
    candidate: CompactnessPolicy | None
    record: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class EvolutionResult:
    output: Path
    summary: dict[str, Any]
    coverage: dict[str, Any]


class EvolutionRunner:
    def __init__(self, tasks: Iterable[BenchmarkTask], *, config: EvolutionConfig,
                 library: PolicyLibrary, output: str | Path, run_variant: Callable,
                 distill_candidate: Callable | None = None, admission: AdmissionConfig | None = None,
                 coverage: Mapping | None = None, config_payload: Mapping | None = None,
                 code_commit: str | None = None, data_identity: Any = None):
        self.tasks = tuple(tasks)
        self.config, self.library, self.output = config, library, Path(output)
        self.run_variant, self.distill_candidate = run_variant, distill_candidate
        self.admission = admission or AdmissionConfig(
            quality_tolerance=config.quality_tolerance,
            minimum_cost_reduction=config.minimum_cost_reduction,
            minimum_candidate_valid_rate=config.minimum_candidate_valid_rate,
            cost_weights=CostWeights(*config.cost_weights))
        self.coverage = copy.deepcopy(dict(coverage or {}))
        self.config_payload = dict(config_payload or {})
        self.code_commit, self.data_identity = code_commit, data_identity
        self._records = {}
        self._decisions = []
        self._completed_rounds = 0
        self._stage = "source"
        self._initial = copy.deepcopy(library.to_dict())
        self._validate_tasks()

    @property
    def _source_records(self):
        return [r for r in self._records.values() if r["split"] == "source"]

    @property
    def _validation_records(self):
        return [r for r in self._records.values() if r["split"] == "validation"]

    @property
    def _target_records(self):
        return [r for r in self._records.values() if r["split"] == "target"]

    def _partition(self, split, benchmark=None):
        return sorted((t for t in self.tasks if t.split == split and (benchmark is None or t.benchmark == benchmark)),
                      key=lambda t: (t.benchmark, digest([self.config.source_seed, t.task_id])))

    def _fold(self, benchmark, round_index):
        return [t for t in self._partition("validation", benchmark) if t.metadata.get("validation_fold") == round_index]

    def _validate_tasks(self):
        validate_partitions(list(self.tasks))
        for b in sorted({t.benchmark for t in self.tasks}):
            need = self.config.bootstrap_tasks + self.config.rounds * self.config.source_tasks_per_round
            if len(self._partition("source", b)) < need:
                raise ValueError(f"{b}: source budget requires {need} tasks")
            for r in range(1, self.config.rounds + 1):
                if len(self._fold(b, r)) != self.config.validation_tasks_per_candidate:
                    raise ValueError(f"{b}: wrong validation fold size for round {r}")
            if len(self._partition("validation", b)) != self.config.rounds * self.config.validation_tasks_per_candidate:
                raise ValueError(f"{b}: unused or duplicate validation folds")
            if not self._partition("target", b):
                raise ValueError(f"{b}: no target tasks")
            for group in _group_tasks(self._partition("validation", b)):
                if len({t.metadata["validation_fold"] for t in group}) != 1:
                    raise ValueError(f"{b}: validation family crosses round folds")

    async def _invoke(self, callback, *args):
        result = callback(*args)
        return await result if inspect.isawaitable(result) else result

    def _manifest(self):
        return [{
            "benchmark": t.benchmark, "task_id": t.task_id, "family_id": t.family_id,
            "family_method": t.metadata.get("family_method", "provided"),
            "split": t.split, "validation_fold": t.metadata.get("validation_fold"),
            "leakage_keys": t.metadata.get("leakage_keys", []),
            "partition_keys": t.metadata.get("partition_keys", []),
            "partition_group_id": t.metadata.get("partition_group_id"),
            "dataset_revision": t.metadata.get("dataset_revision"),
            "level": t.metadata.get("level"),
            "assets": t.metadata.get("gaia_assets", []),
            "task_digest": digest(asdict(t)),
        } for t in sorted(self.tasks, key=lambda t: (t.benchmark, t.task_id))]

    def _code_identity(self):
        root = Path(__file__).resolve().parents[2]
        try:
            commit = self.code_commit or subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = self.code_commit
        # Include uncommitted implementation changes, schemas, prompts and seeds.
        files = {}
        for relative in ("evoagentx/compactflow", "examples/compactflow"):
            for p in sorted((root / relative).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".json", ".txt"}:
                    files[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return {"commit": commit, "files_digest": digest(files)}

    def _prepare(self, resume):
        expected = {
            "schema_version": 2, "config": digest([self.config_payload, asdict(self.config)]),
            "manifest": digest(self._manifest()), "data": digest(self.data_identity),
            "code": self._code_identity(), "initial_library": digest(self._initial),
        }
        lock = self.output / "run.lock.json"
        if lock.exists():
            if not resume:
                raise FileExistsError("output is immutable; use --resume")
            if json.loads(lock.read_text()) != expected:
                raise ValueError("immutable run lock mismatch: config/data/manifest/code/initial library")
        else:
            if any(self.output.iterdir()):
                # The process lock itself is expected.
                if set(p.name for p in self.output.iterdir()) != {".process.lock"}:
                    raise ValueError("refusing nonempty output without a run lock")
            atomic_json(lock, expected)
            atomic_json(self.output / "config.lock.json", {"protocol": self.config_payload, "runner": asdict(self.config)})
            atomic_json(self.output / "environment.lock.json", {
                "python": platform.python_version(), "platform": platform.platform(),
                "code": expected["code"], "model_service": self.config_payload.get("observed_service", {}),
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                             if d.metadata.get("Name") in {"torch", "transformers", "sentence-transformers", "requests",
                                                          "jsonschema", "pyarrow", "numpy", "scipy", "scikit-learn"}},
            })
            atomic_json(self.output / "raw_sources.lock.json", self.data_identity)
        manifest_file = self.output / "sample_manifest.jsonl"
        manifest_text = "".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in self._manifest())
        if manifest_file.exists() and manifest_file.read_text() != manifest_text:
            raise ValueError("frozen manifest content mismatch")
        if not manifest_file.exists():
            manifest_file.write_text(manifest_text)
        self._records = {p.stem: json.loads(p.read_text()) for p in sorted((self.output / "calls").glob("*.json"))}
        state_path = self.output / "checkpoint.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            self.library = self._library_from(state["library"])
            self._decisions = state["decisions"]
            self._completed_rounds = state["completed_rounds"]
            self._stage = state["stage"]
        if self._stage in {"frozen", "target", "complete"}:
            final = self.output / "policy_snapshots/final.json"
            if not final.exists() or json.loads(final.read_text()) != self.library.to_dict():
                raise ValueError("frozen final library snapshot changed")
            target_lock = self.output / "target_library.lock.json"
            if target_lock.exists() and json.loads(target_lock.read_text()) != {"digest": digest(self.library.to_dict())}:
                raise ValueError("target library lock mismatch")
        self._snapshot("initial", self._initial)
        if self._completed_rounds and self._stage == "source":
            # Recover a crash between committing a round and writing its snapshot.
            completed = self.output / "policy_snapshots" / f"round-{self._completed_rounds:02d}.json"
            if not completed.exists():
                self._snapshot(f"round-{self._completed_rounds:02d}")
        for name in ("source_records", "validation_records", "target_records", "candidate_decisions"):
            (self.output / (name + ".jsonl")).touch(exist_ok=True)
        if self.config_payload.get("tools",{}).get("gaia",{}).get("enabled"):
            from .gaia import initialize_gaia_run
            initialize_gaia_run(self.output,self.config_payload,self.tasks)

    def _library_from(self, value):
        return PolicyLibrary(
            [CompactnessPolicy.from_dict(p) for p in value["policies"]],
            evidence=[Evidence.from_dict(e) for e in value.get("evidence", [])],
            embedder=self.library.embedder)

    def _snapshot(self, name, value=None):
        value = value if value is not None else self.library.to_dict()
        path = self.output / "policy_snapshots" / (name + ".json")
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError(f"immutable snapshot changed: {name}")
        else:
            atomic_json(path, value)

    def _checkpoint(self):
        atomic_json(self.output / "checkpoint.json", {
            "stage": self._stage, "completed_rounds": self._completed_rounds,
            "library": self.library.to_dict(), "decisions": self._decisions,
        })
        self._export_records()

    def _export_records(self):
        for split in ("source", "validation", "target"):
            rows = sorted((r for r in self._records.values() if r["split"] == split), key=lambda r: r["call_id"])
            self._jsonl(self.output / (split + "_records.jsonl"), rows)
        self._jsonl(self.output / "candidate_decisions.jsonl", self._decisions)

    @staticmethod
    def _jsonl(path, rows):
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for r in rows))
        temporary.replace(path)

    def _policies(self, *, verified=False):
        experimental = self.config_payload.get("study_variant") == "without_heldout_verification"
        statuses = {PolicyStatus.VERIFIED} if verified and not experimental else {PolicyStatus.CANDIDATE, PolicyStatus.VERIFIED}
        return tuple(p for p in self.library.all() if p.status in statuses)

    async def _variant(self, task, policies, seed, variant, round_index, candidate_id=None):
        key = digest([task.benchmark, task.task_id, task.split, seed, variant, round_index, candidate_id,
                      [p.to_dict() for p in policies]])
        if key in self._records:
            row = self._records[key]
            return VariantResult(Evidence.from_dict(row["evidence"]), row["record"])
        value = await self._invoke(self.run_variant, copy.deepcopy(task), copy.deepcopy(tuple(policies)), seed, variant, task.split)
        if not isinstance(value, VariantResult) or value.evidence.pair_key != (task.benchmark, task.split, task.task_id, seed):
            raise ValueError("paired evidence benchmark/split/task/seed key mismatch")
        # Callback IDs need not distinguish candidates or library versions.
        value.evidence.id = key
        row = {"call_id": key, "benchmark": task.benchmark, "task_id": task.task_id, "split": task.split,
               "family_id": task.family_id, "round": round_index, "validation_fold": task.metadata.get("validation_fold"),
               "candidate_id": candidate_id, "variant": variant, "evidence": value.evidence.to_dict(), "record": value.record}
        atomic_json(self.output / "calls" / (key + ".json"), row)
        self._records[key] = row
        self._export_records()
        print(json.dumps({"phase":"evolution_task", "benchmark":task.benchmark, "task_id":task.task_id,
                          "round":round_index, "variant":variant, "completed_calls":len(self._records),
                          "execution_status":value.record.get("execution_status"),
                          "usage_complete":value.record.get("tokens",{}).get("usage_complete"),
                          "infrastructure_status":value.record.get("infrastructure_status","complete"),
                          "time":datetime.now(timezone.utc).isoformat()}), flush=True)
        return value

    async def _pair(self, task, baseline, candidate, seed, round_index, labels, candidate_id=None):
        a = await self._variant(task, baseline, seed, labels[0], round_index, candidate_id)
        b = await self._variant(task, candidate, seed, labels[1], round_index, candidate_id)
        if a.evidence.pair_key != b.evidence.pair_key:
            raise ValueError("paired evidence keys do not match")
        return a, b

    def _cost_reduction(self, a, b):
        values = [(x-y)/max(abs(x), 1e-9) for x,y in (
            (a.token_cost,b.token_cost), (a.latency,b.latency), (a.total_graph_cost,b.total_graph_cost))]
        return sum(w*x for w,x in zip(self.config.cost_weights,values))/sum(self.config.cost_weights)

    def _eligibility(self, a, b):
        archive = ParetoArchive((a.evidence, b.evidence))
        cost = self._cost_reduction(a.evidence.feedback, b.evidence.feedback)
        checks = {
            "baseline_valid": a.evidence.feedback.valid,
            "current_valid": b.evidence.feedback.valid,
            "contracts_valid": not b.evidence.feedback.contract_violations,
            "quality_preserved": b.evidence.feedback.quality >= a.evidence.feedback.quality - self.config.quality_tolerance,
            "cost_reduction_met": cost >= self.config.minimum_cost_reduction,
            "pareto_frontier": b.evidence.id in {e.id for e in archive.frontier()},
        }
        return {"eligible": all(checks.values()), "checks": checks, "cost_reduction": cost,
                "minimum_cost_reduction": self.config.minimum_cost_reduction,
                "baseline_quality": a.evidence.feedback.quality, "current_quality": b.evidence.feedback.quality}

    def _eligible(self, a, b):
        return self._eligibility(a, b)["eligible"]

    def _check_candidate(self, candidate):
        schema = json.loads((Path(__file__).resolve().parents[2] / "examples/compactflow/schemas/candidate.schema.json").read_text())
        Draft202012Validator(schema).validate({"candidate": {
            "description": candidate.description, "precondition": candidate.precondition,
            "operation": candidate.operation, "expected_effect": candidate.expected_effect,
            "failure_modes": candidate.metadata.get("failure_modes", []),
        }})
        check = candidate.metadata.get("contract_check", {})
        if check.get("contract_valid") is not True or check.get("violations") != []:
            raise ValueError("candidate lacks a successful contract/safety check")
        if candidate.id in {p.id for p in self.library.all()}:
            raise ValueError("candidate ID already exists")

    async def _distill(self, task, a, b, round_index):
        key = digest([task.benchmark, task.task_id, round_index])
        path = self.output / "candidates" / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())
        eligibility = self._eligibility(a, b)
        proposal = DistillationResult(None, reason="source_not_eligible")
        if self.distill_candidate and eligibility["eligible"]:
            try:
                value = await self._invoke(self.distill_candidate, task, a, b, round_index)
                proposal = value if isinstance(value, DistillationResult) else DistillationResult(value)
                if proposal.record.get("tokens") and not proposal.record["tokens"].get("usage_complete"):
                    proposal.candidate = None
                    proposal.reason = "distillation_usage_incomplete"
                if proposal.candidate is not None:
                    self._check_candidate(proposal.candidate)
            except (ValueError, TypeError, KeyError, ValidationError) as exc:
                proposal = DistillationResult(None, record=proposal.record, reason=f"candidate_check_failed: {exc}")
        row = {"key": key, "benchmark": task.benchmark, "source_task_id": task.task_id, "round": round_index,
               "candidate": proposal.candidate.to_dict() if proposal.candidate else None,
               "record": proposal.record, "reason": proposal.reason or "no_candidate", "eligibility": eligibility}
        atomic_json(path, row)
        return row

    def _statistics(self, policy, observations):
        unique = {o["key"]: o for o in observations}
        observations = list(unique.values())
        n = len(observations)
        if not n:
            return
        policy.utility = sum(max(0., min(1., o["reduction"])) for o in observations)/n
        phat, z = sum(o["success"] for o in observations)/n, 1.96
        policy.confidence = (phat+z*z/(2*n)-z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n)))/(1+z*z/n)
        policy.metadata["paired_observations"] = observations

    async def _validate(self, candidate, tasks, round_index):
        if self.config_payload.get("study_variant") == "without_heldout_verification":
            self._check_candidate(candidate)
            candidate.metadata["experimental_unverified"] = True
            self.library.add(candidate)
            return {"verdict": "experimental_keep", "reason": "static contract only; not verified", "policy": candidate.to_dict()}
        baseline_policies = self._policies(verified=True)
        pairs = []
        for task in tasks:
            for seed in self.config.heldout_seeds:
                a,b = await self._pair(task, baseline_policies, (*baseline_policies,candidate), seed,
                                       round_index, ("candidate_base","candidate_test"), candidate.id)
                pairs.append(PairedExecution(a.evidence,b.evidence))
        for pair in pairs:
            self.library.record_evidence(pair.baseline)
            self.library.record_evidence(pair.candidate)
        admission = PairedPolicyAdmission(self.library, config=self.admission)
        decision = admission.evaluate(candidate, pairs)
        safety_ok = all(not p.candidate.feedback.contract_violations
                        and p.candidate.feedback.metrics.get("usage_complete", True) for p in pairs)
        other_ok = all(sum(getattr(p.candidate.feedback, attr) for p in pairs) <=
                       (1+self.config.other_cost_tolerance)*sum(getattr(p.baseline.feedback, attr) for p in pairs)+1e-9
                       for attr in ("token_cost","latency","total_graph_cost"))
        if not safety_ok or not other_ok:
            decision = replace(decision, verdict=AdmissionVerdict.REJECT,
                               reason="contract/usage check failed" if not safety_ok else "non-target cost tolerance exceeded")
        nearest = self.library.get(decision.nearest_policy_id) if decision.nearest_policy_id else None
        # Merging keeps the old rule's semantics: only transfer verification to
        # a rule with the same structured precondition and operation.
        if decision.verdict == AdmissionVerdict.MERGE and nearest and (
                nearest.precondition != candidate.precondition or nearest.operation != candidate.operation):
            decision = replace(decision, verdict=AdmissionVerdict.ADMIT, reason="similar description but different rule; admit separately")
        observations = []
        for pair in pairs:
            a,b = pair.baseline.feedback, pair.candidate.feedback
            reduction = self._cost_reduction(a,b)
            success = bool(b.valid and not b.contract_violations and b.quality-a.quality >= -self.config.quality_tolerance
                           and reduction >= self.config.minimum_cost_reduction)
            observations.append({"key": digest([candidate.id,candidate.version,pair.pair_key]), "reduction": reduction, "success": success})
            candidate.with_evidence(pair.candidate, positive=success)
        self._statistics(candidate, observations)
        previous = copy.deepcopy(nearest.metadata.get("paired_observations", [])) if nearest else []
        applied = admission.apply(candidate, decision)
        if decision.verdict == AdmissionVerdict.MERGE and applied:
            self._statistics(applied, previous+observations)
        if decision.verdict == AdmissionVerdict.REJECT:
            self.library.add(candidate)
        active = [p for p in self.library.all() if p.status != PolicyStatus.REJECTED]
        evicted = []
        while len(active) > self.config.library_capacity:
            victim = min(active,key=lambda p:(p.confidence,p.utility,p.metadata.get("source_round",0),p.id))
            victim.status = PolicyStatus.REJECTED
            victim.metadata["evicted"] = True
            evicted.append(victim.id)
            active.remove(victim)
        return {**asdict(decision), "verdict":decision.verdict.value, "policy":candidate.to_dict(), "evicted":evicted}

    async def _source(self):
        benchmarks = sorted({t.benchmark for t in self.tasks})
        if not (self.output / "policy_snapshots/bootstrap.json").exists():
            for b in benchmarks:
                for task in self._partition("source",b)[:self.config.bootstrap_tasks]:
                    a,c = await self._pair(task, (), self._policies(), self.config.source_seed, 0, ("no_policy","source_library"))
                    self.library.record_evidence(a.evidence)
                    self.library.record_evidence(c.evidence)
            self._checkpoint()
            self._snapshot("bootstrap")
        for r in range(self._completed_rounds+1,self.config.rounds+1):
            for b in benchmarks:
                start = self.config.bootstrap_tasks+(r-1)*self.config.source_tasks_per_round
                batch = self._partition("source",b)[start:start+self.config.source_tasks_per_round]
                # Freeze the source library for this benchmark-round, even if
                # resume happens after one candidate admission.
                source_path = self.output / "source_libraries" / (digest([b,r])+".json")
                if not source_path.exists():
                    atomic_json(source_path,[p.to_dict() for p in self._policies()])
                source_policies = tuple(CompactnessPolicy.from_dict(p) for p in json.loads(source_path.read_text()))
                proposals = []
                for task in batch:
                    a,c = await self._pair(task, (), source_policies, self.config.source_seed, r, ("no_policy","source_library"))
                    self.library.record_evidence(a.evidence)
                    self.library.record_evidence(c.evidence)
                    proposal = await self._distill(task,a,c,r)
                    proposals.append(proposal)
                retained = 0
                for proposal in proposals:
                    if any(d["key"] == proposal["key"] for d in self._decisions):
                        retained += int(bool(proposal["candidate"]) and not any(
                            d["key"] == proposal["key"] and d["reason"] == "round_candidate_budget" for d in self._decisions))
                        continue
                    candidate_data = proposal["candidate"]
                    result = {"verdict":"reject","reason":proposal["reason"]}
                    if candidate_data:
                        if retained >= self.config.max_candidates_per_round:
                            result["reason"] = "round_candidate_budget"
                        else:
                            retained += 1
                            candidate = CompactnessPolicy.from_dict(candidate_data)
                            candidate.metadata.update(source_round=r,source_benchmark=b,source_task_id=proposal["source_task_id"])
                            try:
                                self._check_candidate(candidate)
                                result = await self._validate(candidate,self._fold(b,r),r)
                            except (ValueError,TypeError,KeyError,ValidationError) as exc:
                                candidate.status = PolicyStatus.REJECTED
                                result = {"verdict":"reject","reason":f"validation_failed: {exc}","policy":candidate.to_dict()}
                    self._decisions.append({k:proposal[k] for k in ("key","benchmark","source_task_id","round")} | {"eligibility":proposal.get("eligibility")} | result)
                    self._checkpoint()
            self._completed_rounds = r
            self._checkpoint()
            print(json.dumps({"phase":"evolution_round", "completed_rounds":r, "decisions":len(self._decisions)}),flush=True)
            self._snapshot(f"round-{r:02d}")
        # Filter before target starts; snapshots retain the earlier evidence.
        for p in list(self.library.all()):
            if p.status != PolicyStatus.VERIFIED and not (self.config_payload.get("study_variant") == "without_heldout_verification" and p.status == PolicyStatus.CANDIDATE):
                self.library.remove(p.id)
        self._snapshot("final")
        self._stage = "frozen"
        self._checkpoint()

    async def _target(self):
        from .paper_phase import assert_target_allowed
        assert_target_allowed(self.config_payload, "target")
        self._stage = "target"
        self._checkpoint()
        final_path = self.output / "policy_snapshots/final.json"
        if not final_path.exists():
            raise ValueError("target requires a final snapshot")
        frozen = json.loads(final_path.read_text())
        if self.library.to_dict() != frozen:
            raise ValueError("target library differs from final snapshot")
        freeze = {"digest":digest(frozen)}
        lock = self.output / "target_library.lock.json"
        if lock.exists() and json.loads(lock.read_text()) != freeze:
            raise ValueError("target library lock mismatch")
        if not lock.exists():
            atomic_json(lock,freeze)
        for task in self._partition("target"):
            for seed in self.config.target_seeds:
                await self._pair(task,(),self._policies(verified=True),seed,None,("base_planner","compactflow"))
                if self.library.to_dict() != frozen or json.loads(final_path.read_text()) != frozen:
                    raise ValueError("target evaluation changed the frozen library")
        self._stage = "complete"
        self._checkpoint()

    def _summary(self):
        datasets = {}
        expected_benchmarks = {d["name"] for d in self.config_payload.get("datasets", [])} or {t.benchmark for t in self.tasks}
        for b in sorted(expected_benchmarks):
            records = [r for r in self._target_records if r["benchmark"] == b]
            expected = len(self._partition("target",b))*len(self.config.target_seeds)*2
            valid = bool(expected) and len(records) == expected and all(
                r["evidence"]["feedback"]["valid"] and r["record"].get("tokens",{}).get("usage_complete",True) for r in records)
            datasets[b] = {"target_records":len(records),"expected_records":expected,
                "tasks":len({r["task_id"] for r in records}), "status":"complete" if valid else "incomplete",
                "quality":{v+"_mean":self._mean_quality(records,v) for v in ("base_planner","compactflow")}}
        incomplete = [k for k,v in self.coverage.items() if isinstance(v,Mapping) and v.get("status") != "complete"]
        complete = bool(datasets) and all(v["status"] == "complete" for v in datasets.values()) and not incomplete
        return {"publication_status":"complete" if complete else "incomplete", "datasets":datasets,
                "source_records":len(self._source_records),"validation_records":len(self._validation_records),
                "target_records":len(self._target_records), "admissions":dict(collections.Counter(d["verdict"] for d in self._decisions)),
                "coverage_incomplete":incomplete, "protocol":"per-benchmark source budgets and disjoint round validation folds",
                "tokens":sum(r["evidence"]["feedback"]["token_cost"] for r in self._records.values()),
                "distillation_tokens":sum(json.loads(p.read_text()).get("record",{}).get("tokens",{}).get("total_tokens",0)
                                         for p in (self.output/"candidates").glob("*.json"))}

    @staticmethod
    def _mean_quality(records, variant):
        values = [r["evidence"]["feedback"]["quality"] for r in records if r["variant"] == variant]
        return sum(values)/len(values) if values else None

    async def run(self, *, resume=False, stage="all"):
        if stage not in {"all", "freeze", "target"}:
            raise ValueError("invalid evolution stage")
        self.output.mkdir(parents=True,exist_ok=True)
        with (self.output/".process.lock").open("a") as process_lock:
            try:
                fcntl.flock(process_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("another process owns this run") from exc
            self._prepare(resume)
            if self._stage == "source":
                if stage == "target":
                    raise ValueError("evolution must freeze before target")
                await self._source()
            if stage != "freeze" and self._stage != "complete":
                await self._target()
            summary = self._summary()
            from .gaia import gaia_summary
            summary["gaia"] = gaia_summary(list(self._records.values()))
            atomic_json(self.output/"coverage.json",self.coverage)
            atomic_json(self.output/"summary.json",summary)
            with (self.output/"tables.csv").open("w") as f:
                writer = csv.DictWriter(f,fieldnames=["benchmark","base_planner_mean","compactflow_mean","status"])
                writer.writeheader()
                writer.writerows({"benchmark":b,**v["quality"],"status":v["status"]} for b,v in summary["datasets"].items())
            return EvolutionResult(self.output,summary,self.coverage)


async def run_evolution(tasks, *, resume=False, **kwargs):
    """Convenience API for a complete run or restart from its durable checkpoint."""
    return await EvolutionRunner(tasks, **kwargs).run(resume=resume)
